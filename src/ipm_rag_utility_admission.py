from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NEEDED_COLUMNS = [
    "dataset",
    "query_id",
    "gold_support",
    "retriever",
    "e5_small_v2_score",
    "bge_m3_score",
    "entity_score",
    "relation_score",
    "source_score",
    "scet_conflict_score",
]


def minmax_by_query(values: pd.Series, query_keys: pd.Series) -> pd.Series:
    def scale(group: pd.Series) -> pd.Series:
        lo = float(group.min())
        hi = float(group.max())
        if hi <= lo:
            return pd.Series(0.0, index=group.index)
        return (group - lo) / (hi - lo)

    return values.groupby(query_keys.astype(str), sort=False).transform(scale)


def load_scet_candidate_rows(detail_path: Path, chunksize: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(detail_path, usecols=lambda col: col in NEEDED_COLUMNS, chunksize=chunksize):
        if "retriever" not in chunk.columns:
            continue
        part = chunk[chunk["retriever"].astype(str).eq("scet")].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=NEEDED_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def add_rag_utility_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["e5_small_v2_score", "bge_m3_score", "entity_score", "relation_score", "source_score", "scet_conflict_score"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    semantic_utility = out[["e5_small_v2_score", "bge_m3_score"]].max(axis=1)
    anchor_utility = 0.60 * out["relation_score"] + 0.40 * out["entity_score"]
    source_utility = out["source_score"].clip(0.0, 1.0)
    raw = 0.62 * semantic_utility + 0.28 * anchor_utility + 0.10 * source_utility - 0.20 * out["scet_conflict_score"]
    query_key = out["dataset"].astype(str) + "::" + out["query_id"].astype(str)
    out["retrieval_score"] = minmax_by_query(raw, query_key)
    out["rank"] = out.groupby(query_key, sort=False)["retrieval_score"].rank(method="first", ascending=False).astype(int)
    out["evidence_supported"] = out["rank"].le(5).astype(int)
    out["retriever"] = "rag_utility"
    out["gold_support"] = pd.to_numeric(out["gold_support"], errors="coerce").fillna(0).astype(int)
    return out


def dcg(relevances: list[int]) -> float:
    return float(sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances)))


def evidence_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    work = rows.copy()
    query_key = work["dataset"].astype(str) + "::" + work["query_id"].astype(str)
    queries = query_key.nunique()
    gold_queries = 0
    hit_queries = 0
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    for _, group in work.groupby(query_key, sort=False):
        ordered = group.sort_values("retrieval_score", ascending=False)
        gold = ordered["gold_support"].astype(int).tolist()
        if any(gold):
            gold_queries += 1
        if any(gold[:5]):
            hit_queries += 1
        first_rank = next((idx + 1 for idx, value in enumerate(gold) if value), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        ideal = sorted(gold, reverse=True)[:10]
        ndcg_values.append(dcg(gold[:10]) / max(dcg(ideal), 1e-12))
    supported_positive = int(((work["evidence_supported"].eq(1)) & (work["gold_support"].eq(1))).sum())
    supported_negative = int(((work["evidence_supported"].eq(1)) & (work["gold_support"].eq(0))).sum())
    supported_total = supported_positive + supported_negative
    return {
        "queries": int(queries),
        "gold_queries": int(gold_queries),
        "recall_at_5": hit_queries / max(gold_queries, 1),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "ndcg_at_10": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "supported_positive": supported_positive,
        "supported_negative": supported_negative,
        "support_precision": supported_positive / max(supported_total, 1),
        "supported_per_100": 100.0 * supported_positive / max(queries, 1),
        "evidence_hit": supported_positive / max(int(work["gold_support"].sum()), 1),
    }


def stress_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    work = rows.copy()
    high_conflict = work["scet_conflict_score"] >= work["scet_conflict_score"].quantile(0.75)
    high_similarity = work["retrieval_score"] >= work["retrieval_score"].quantile(0.75)
    stress_pool = work[high_conflict | (high_similarity & work["gold_support"].eq(0))]
    if stress_pool.empty:
        stress_pool = work
    supported = stress_pool[stress_pool["evidence_supported"].eq(1)]
    true_supported = int(supported["gold_support"].eq(1).sum()) if not supported.empty else 0
    false_supported = int(supported["gold_support"].eq(0).sum()) if not supported.empty else 0
    return {
        "stress_rows": int(len(stress_pool)),
        "stress_supported": int(len(supported)),
        "stress_true_supported": true_supported,
        "stress_false_supported": false_supported,
        "stress_support_precision": true_supported / max(true_supported + false_supported, 1),
        "mean_conflict_score": float(stress_pool["scet_conflict_score"].mean()) if not stress_pool.empty else 0.0,
    }


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    stress_rows = []
    for dataset, group in detail.groupby("dataset", sort=False):
        metrics = evidence_metrics(group)
        metrics.update({"dataset": dataset, "retriever": "rag_utility"})
        summary_rows.append(metrics)
        stress = stress_metrics(group)
        stress.update({"dataset": dataset, "retriever": "rag_utility"})
        stress_rows.append(stress)
    return pd.DataFrame(summary_rows), pd.DataFrame(stress_rows)


def run(detail_path: Path, out: Path, chunksize: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base = load_scet_candidate_rows(detail_path, chunksize)
    if base.empty:
        raise RuntimeError(f"No SCET candidate rows found in {detail_path}")
    detail = add_rag_utility_scores(base)
    summary, stress = summarize(detail)
    keep_detail = ["dataset", "query_id", "gold_support", "retrieval_score", "rank", "evidence_supported", "retriever", "scet_conflict_score"]
    detail[keep_detail].to_csv(out / "rag_utility_detail.csv", index=False)
    summary.to_csv(out / "rag_utility_summary.csv", index=False)
    stress.to_csv(out / "rag_utility_stress_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.detail, args.out, args.chunksize)
