from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ipm_cross_domain_evidence_benchmark import evidence_retrieval_metrics
from ipm_scet_evidence_transport import build_conflict_stress_summary
from ipm_scet_risk_reduction_audit import (
    matched_volume_threshold_audit,
    threshold_sweep,
    unique_scet_candidate_rows,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DETAIL = PROJECT / "results" / "ipm_scet_full_seed13_full_neural" / "scet_detail.csv"
DEFAULT_OUT = PROJECT / "results" / "ipm_scet_nli_verification"
DEFAULT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_DIRECTION = "evidence_premise_to_query_hypothesis_v1"

KEY_COLUMNS = ["dataset", "query_id", "candidate_id"]
NLI_COLUMNS = [
    "nli_pool_rank",
    "nli_entailment_score",
    "nli_contradiction_score",
    "nli_neutral_score",
    "nli_support_score",
]


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _query_rank(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return frame.groupby("query_id", sort=False)[score_col].rank(method="first", ascending=False).astype(int)


def _pair_digest(row: pd.Series) -> str:
    hasher = hashlib.sha1()
    hasher.update(NLI_DIRECTION.encode("utf-8"))
    hasher.update(b"\0")
    for col in [*KEY_COLUMNS, "query_text", "evidence_text"]:
        hasher.update(str(row.get(col, "")).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:20]


def _label_index(id2label: dict[int, str], target: str, fallback: int) -> int:
    target = target.lower()
    for raw_idx, raw_label in id2label.items():
        if target in str(raw_label).lower():
            return int(raw_idx)
    return fallback


def nli_probabilities_to_scores(probabilities: np.ndarray, id2label: dict[int, str]) -> pd.DataFrame:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 3:
        raise ValueError("NLI probabilities must be a two-dimensional array with at least three labels")
    contradiction_idx = _label_index(id2label, "contradiction", 0)
    entailment_idx = _label_index(id2label, "entailment", 1)
    neutral_idx = _label_index(id2label, "neutral", 2)
    out = pd.DataFrame(
        {
            "nli_entailment_score": probs[:, entailment_idx],
            "nli_contradiction_score": probs[:, contradiction_idx],
            "nli_neutral_score": probs[:, neutral_idx],
        }
    )
    out["nli_support_score"] = out["nli_entailment_score"] - out["nli_contradiction_score"]
    return out


def nli_premise_hypothesis_texts(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    premises = frame["evidence_text"].fillna("").astype(str).tolist()
    hypotheses = frame["query_text"].fillna("").astype(str).tolist()
    return premises, hypotheses


def select_nli_candidate_pool(frame: pd.DataFrame, retrieval_col: str = "bge_m3_score", retrieve_k: int = 20) -> pd.DataFrame:
    if retrieval_col not in frame.columns:
        raise ValueError(f"Missing retrieval column: {retrieval_col}")
    work = frame.copy()
    work["_nli_retrieval_score"] = _safe_numeric(work, retrieval_col)
    work["nli_pool_rank"] = _query_rank(work, "_nli_retrieval_score")
    pool = work[work["nli_pool_rank"] <= int(retrieve_k)].copy()
    return pool.drop(columns=["_nli_retrieval_score"], errors="ignore").sort_values(["dataset", "query_id", "nli_pool_rank"]).reset_index(drop=True)


def _filter_max_queries(frame: pd.DataFrame, max_queries_per_dataset: int | None) -> pd.DataFrame:
    if max_queries_per_dataset is None:
        return frame
    parts = []
    for _, group in frame.groupby("dataset", sort=False):
        keep = group["query_id"].drop_duplicates().astype(str).head(int(max_queries_per_dataset))
        parts.append(group[group["query_id"].astype(str).isin(set(keep))].copy())
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def _load_score_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        cached = pd.read_csv(cache_path)
        if "pair_digest" in cached.columns:
            return cached
    return pd.DataFrame()


def _write_score_cache(cache_path: Path, table: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.drop_duplicates("pair_digest", keep="last").to_csv(cache_path, index=False)


def score_nli_candidate_pool(
    pool: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 96,
    max_length: int = 256,
    cache_path: Path | None = None,
    flush_every_batches: int = 25,
) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "pair_digest", *NLI_COLUMNS])
    work = pool.copy()
    work["pair_digest"] = work.apply(_pair_digest, axis=1)
    cache_path = cache_path or (DEFAULT_OUT / "nli_score_cache.csv")
    cached = _load_score_cache(cache_path)
    existing = set(cached["pair_digest"].astype(str)) if not cached.empty else set()
    missing = work[~work["pair_digest"].astype(str).isin(existing)].copy()

    produced_parts = [cached] if not cached.empty else []
    if not missing.empty:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        chunk_rows: list[pd.DataFrame] = []
        with torch.no_grad():
            for batch_no, start in enumerate(range(0, len(missing), int(batch_size)), start=1):
                batch = missing.iloc[start : start + int(batch_size)]
                premises, hypotheses = nli_premise_hypothesis_texts(batch)
                encoded = tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                probabilities = torch.softmax(model(**encoded).logits, dim=-1).detach().cpu().numpy()
                score_cols = nli_probabilities_to_scores(probabilities, id2label)
                scored_batch = pd.concat(
                    [
                        batch[[*KEY_COLUMNS, "query_text", "evidence_text", "evidence_source", "gold_support", "nli_pool_rank", "pair_digest"]]
                        .reset_index(drop=True),
                        score_cols.reset_index(drop=True),
                    ],
                    axis=1,
                )
                chunk_rows.append(scored_batch)
                if batch_no % int(flush_every_batches) == 0:
                    interim = pd.concat([*produced_parts, *chunk_rows], ignore_index=True)
                    _write_score_cache(cache_path, interim)
        if chunk_rows:
            produced_parts.extend(chunk_rows)
            _write_score_cache(cache_path, pd.concat(produced_parts, ignore_index=True))

    scores = _load_score_cache(cache_path)
    needed = work[["pair_digest"]].drop_duplicates()
    scores = needed.merge(scores, on="pair_digest", how="left")
    return scores


def attach_nli_scores(frame: pd.DataFrame, score_rows: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if score_rows.empty:
        for col in NLI_COLUMNS:
            out[col] = np.nan
        out["nli_verifier_score"] = -1.0
        return out
    scores = score_rows.drop_duplicates(KEY_COLUMNS, keep="last")[[*KEY_COLUMNS, *NLI_COLUMNS]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
        scores[col] = scores[col].astype(str)
    out = out.merge(scores, on=KEY_COLUMNS, how="left")
    out["nli_verifier_score"] = pd.to_numeric(out["nli_support_score"], errors="coerce").fillna(-1.0)
    return out


def apply_nli_verification_loop(scored: pd.DataFrame, top_k: int = 5, retriever_name: str = "nli_verification_loop") -> pd.DataFrame:
    if "nli_support_score" not in scored.columns:
        raise ValueError("NLI verification loop requires nli_support_score")
    out = scored.copy()
    out["retrieval_score"] = pd.to_numeric(out["nli_support_score"], errors="coerce").fillna(-1.0)
    if "bge_m3_score" in out.columns:
        out["retrieval_score"] = out["retrieval_score"] + 0.001 * _safe_numeric(out, "bge_m3_score")
    out["rank"] = _query_rank(out, "retrieval_score")
    out["top_k"] = int(top_k)
    in_pool = pd.to_numeric(out.get("nli_pool_rank", pd.Series(np.nan, index=out.index)), errors="coerce").notna()
    out["evidence_supported"] = ((out["rank"] <= int(top_k)) & in_pool).astype(int)
    out["retriever"] = retriever_name
    return out


def candidate_pool_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in frame.groupby("dataset", sort=False):
        per_query = group.groupby("query_id", sort=False).size()
        gold_by_query = group.groupby("query_id", sort=False)["gold_support"].sum()
        rows.append(
            {
                "dataset": dataset,
                "queries": float(group["query_id"].nunique()),
                "candidate_rows": float(len(group)),
                "mean_candidates_per_query": float(per_query.mean()) if len(per_query) else 0.0,
                "median_candidates_per_query": float(per_query.median()) if len(per_query) else 0.0,
                "gold_queries": float((gold_by_query > 0).sum()),
                "gold_support_rows": float(pd.to_numeric(group["gold_support"], errors="coerce").fillna(0).sum()),
                "mean_gold_rows_per_gold_query": float(gold_by_query[gold_by_query > 0].mean()) if (gold_by_query > 0).any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_nli_tables(scored: pd.DataFrame, top_k: int = 5) -> dict[str, pd.DataFrame]:
    loop_detail = apply_nli_verification_loop(scored, top_k=top_k)
    loop_summary = pd.DataFrame(
        [
            {"dataset": dataset, "retriever": "nli_verification_loop", **evidence_retrieval_metrics(group, top_k=top_k)}
            for dataset, group in loop_detail.groupby("dataset", sort=False)
        ]
    )
    loop_stress = build_conflict_stress_summary(loop_detail)
    matched_parts = []
    threshold_parts = []
    for dataset, group in scored.groupby("dataset", sort=False):
        group = group.copy()
        matched_parts.append(matched_volume_threshold_audit(group, baseline_score_cols=["nli_verifier_score"], top_k=top_k))
        threshold_parts.append(threshold_sweep(group, "nli_verifier_score", top_k=top_k))
    matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    threshold = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    return {
        "nli_verification_detail": loop_detail,
        "nli_verification_summary": loop_summary,
        "nli_verification_stress": loop_stress,
        "nli_matched_volume_audit": matched,
        "nli_threshold_curve": threshold,
    }


def run_nli_audit(
    detail_path: Path = DEFAULT_DETAIL,
    out_dir: Path = DEFAULT_OUT,
    datasets: list[str] | None = None,
    retrieval_col: str = "bge_m3_score",
    retrieve_k: int = 20,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 96,
    max_length: int = 256,
    max_queries_per_dataset: int | None = None,
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(detail_path, low_memory=False)
    unique = unique_scet_candidate_rows(detail)
    if datasets:
        wanted = {dataset.strip().lower() for dataset in datasets}
        unique = unique[unique["dataset"].astype(str).str.lower().isin(wanted)].copy()
    unique = _filter_max_queries(unique, max_queries_per_dataset)
    pool_audit = candidate_pool_audit(unique)
    pool = select_nli_candidate_pool(unique, retrieval_col=retrieval_col, retrieve_k=retrieve_k)
    suffix = "full" if max_queries_per_dataset is None else f"q{max_queries_per_dataset}"
    dataset_suffix = "all" if not datasets else "_".join(sorted(dataset.lower() for dataset in datasets))
    cache_path = out_dir / f"nli_score_cache_{dataset_suffix}_{suffix}_k{retrieve_k}.csv"
    scores = score_nli_candidate_pool(
        pool,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        cache_path=cache_path,
    )
    scored = attach_nli_scores(unique, scores)
    tables = build_nli_tables(scored, top_k=top_k)
    tables["nli_candidate_pool_audit"] = pool_audit
    tables["nli_score_rows"] = scores
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def _parse_datasets(value: str) -> list[str] | None:
    if not value.strip() or value.strip().lower() == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLI verification diagnostics for SCET.")
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--retrieval-col", default="bge_m3_score")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-queries-per-dataset", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = run_nli_audit(
        detail_path=args.detail,
        out_dir=args.out,
        datasets=_parse_datasets(args.datasets),
        retrieval_col=args.retrieval_col,
        retrieve_k=args.retrieve_k,
        top_k=args.top_k,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_queries_per_dataset=args.max_queries_per_dataset,
    )
    summary = tables.get("nli_verification_summary", pd.DataFrame())
    print(summary.to_string(index=False) if not summary.empty else "(no summary rows)")


if __name__ == "__main__":
    main()
