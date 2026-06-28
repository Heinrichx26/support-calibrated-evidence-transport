from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ipm_cross_domain_evidence_benchmark import evidence_retrieval_metrics, minmax
from ipm_scet_evidence_transport import build_conflict_stress_summary


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DETAIL = PROJECT / "results" / "ipm_scet_full_seed13_full_neural" / "scet_detail.csv"
DEFAULT_OUT = PROJECT / "results" / "ipm_scet_risk_reduction_audit"

BASELINE_SCORE_COLUMNS = [
    "bm25_score",
    "tfidf_score",
    "dense_score",
    "sbert_mini_lm_score",
    "e5_small_v2_score",
    "bge_m3_score",
    "cross_encoder_score",
]


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _query_rank(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return frame.groupby("query_id", sort=False)[score_col].rank(method="first", ascending=False).astype(int)


def unique_scet_candidate_rows(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail.copy()
    work = detail[detail["retriever"].astype(str).eq("scet")].copy()
    if work.empty:
        work = detail.copy()
    keys = ["dataset", "query_id", "candidate_id"]
    work = work.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)
    work["gold_support"] = _safe_numeric(work, "gold_support").astype(int).clip(0, 1)
    return work


def apply_retrieval_verification_loop(
    frame: pd.DataFrame,
    retrieval_col: str,
    verifier_col: str,
    retrieve_k: int = 20,
    top_k: int = 5,
    name: str | None = None,
) -> pd.DataFrame:
    if retrieval_col not in frame.columns:
        raise ValueError(f"Missing retrieval column: {retrieval_col}")
    if verifier_col not in frame.columns:
        raise ValueError(f"Missing verifier column: {verifier_col}")
    out = frame.copy()
    retrieval_rank_col = "_retrieval_loop_rank"
    out[retrieval_rank_col] = _query_rank(out.assign(_retrieval_score=_safe_numeric(out, retrieval_col)), "_retrieval_score")
    in_pool = out[retrieval_rank_col] <= int(retrieve_k)

    verifier = _safe_numeric(out, verifier_col)
    relation = _safe_numeric(out, "relation_score")
    entity = _safe_numeric(out, "entity_score")
    source = _safe_numeric(out, "source_score", 1.0)
    conflict = _safe_numeric(out, "scet_conflict_score")
    raw = 0.55 * verifier + 0.18 * relation + 0.17 * entity + 0.05 * source - 0.25 * conflict

    loop_score = pd.Series(-1.0, index=out.index, dtype=float)
    for _, group in out.groupby("query_id", sort=False):
        idx = group.index
        values = raw.loc[idx].to_numpy(float)
        normed = minmax(values)
        loop_score.loc[idx] = normed
    loop_score.loc[~in_pool] = -1.0
    out["retrieval_score"] = loop_score
    out["rank"] = _query_rank(out, "retrieval_score")
    out["top_k"] = int(top_k)
    out["evidence_supported"] = ((out["rank"] <= int(top_k)) & in_pool).astype(int)
    out["retriever"] = name or f"{retrieval_col}_verification_loop"
    out = out.drop(columns=[retrieval_rank_col], errors="ignore")
    return out


def _support_metrics(frame: pd.DataFrame, score_col: str, threshold: float, top_k: int) -> dict[str, float]:
    work = frame.copy()
    work["retrieval_score"] = _safe_numeric(work, score_col)
    work["rank"] = _query_rank(work, "retrieval_score")
    work["top_k"] = int(top_k)
    work["evidence_supported"] = ((work["rank"] <= int(top_k)) & (work["retrieval_score"] >= float(threshold))).astype(int)
    return {key: float(value) for key, value in evidence_retrieval_metrics(work, top_k=int(top_k)).items()}


def threshold_sweep(frame: pd.DataFrame, score_col: str, top_k: int = 5, n_grid: int = 101) -> pd.DataFrame:
    scores = _safe_numeric(frame, score_col)
    if scores.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["_audit_score"] = scores
    work["_audit_rank"] = _query_rank(work, "_audit_score")
    work["gold_support"] = _safe_numeric(work, "gold_support").astype(int).clip(0, 1)
    top_rows = work[work["_audit_rank"] <= int(top_k)].copy()
    query_count = int(work["query_id"].nunique())
    gold_total = int(work["gold_support"].sum())
    gold_by_query = work.groupby("query_id", sort=False)["gold_support"].sum()
    top_hit_by_query = top_rows.groupby("query_id", sort=False)["gold_support"].max()
    gold_queries = int((gold_by_query > 0).sum())
    hit_queries = int((top_hit_by_query > 0).sum())
    recall_at_5 = hit_queries / max(gold_queries, 1)
    quantiles = np.linspace(0.0, 1.0, int(n_grid))
    thresholds = sorted({float(np.quantile(scores, q)) for q in quantiles} | {0.0, 0.25, 0.5, 0.75, 0.95, 1.0})
    rows = []
    for threshold in thresholds:
        selected = top_rows[top_rows["_audit_score"] >= float(threshold)]
        supported_positive = int(selected["gold_support"].eq(1).sum())
        supported_negative = int(selected["gold_support"].eq(0).sum())
        supported_total = supported_positive + supported_negative
        rows.append(
            {
                "dataset": str(frame["dataset"].iloc[0]) if "dataset" in frame and not frame.empty else "",
                "score_column": score_col,
                "threshold": float(threshold),
                "support_precision": supported_positive / max(supported_total, 1),
                "supported_per_100": 100.0 * supported_positive / max(query_count, 1),
                "certificate_per_100": 100.0 * supported_total / max(query_count, 1),
                "recall_at_5": float(recall_at_5),
                "evidence_hit": supported_positive / max(gold_total, 1),
                "supported_positive": float(supported_positive),
                "supported_negative": float(supported_negative),
                "certificate_total": float(supported_total),
            }
        )
    return pd.DataFrame(rows)


def _scet_metrics(frame: pd.DataFrame, top_k: int) -> dict[str, float]:
    if "scet_support_score" not in frame.columns:
        return {}
    scet = frame.copy()
    scet["retrieval_score"] = _safe_numeric(scet, "scet_support_score")
    scet["rank"] = _query_rank(scet, "retrieval_score")
    if "evidence_supported" not in scet.columns:
        scet["evidence_supported"] = (scet["rank"] <= top_k).astype(int)
    metrics = {key: float(value) for key, value in evidence_retrieval_metrics(scet, top_k=top_k).items()}
    query_count = int(scet["query_id"].nunique())
    metrics["certificate_per_100"] = 100.0 * float(pd.to_numeric(scet["evidence_supported"], errors="coerce").fillna(0).sum()) / max(query_count, 1)
    return metrics


def matched_volume_threshold_audit(
    frame: pd.DataFrame,
    baseline_score_cols: list[str] | None = None,
    top_k: int = 5,
    min_volume_ratio: float = 0.9,
) -> pd.DataFrame:
    baseline_score_cols = baseline_score_cols or [col for col in BASELINE_SCORE_COLUMNS if col in frame.columns]
    scet_metrics = _scet_metrics(frame, top_k=top_k)
    scet_volume = float(scet_metrics.get("certificate_per_100", 0.0))
    scet_supported = float(scet_metrics.get("supported_per_100", 0.0))
    scet_precision = float(scet_metrics.get("support_precision", 0.0))
    rows = []
    for col in baseline_score_cols:
        if col not in frame.columns:
            continue
        sweep = threshold_sweep(frame, col, top_k=top_k)
        if sweep.empty:
            continue
        lower_volume = scet_volume * float(min_volume_ratio)
        feasible = sweep[(sweep["certificate_per_100"] <= scet_volume + 1e-9) & (sweep["certificate_per_100"] >= lower_volume - 1e-9)].copy()
        if feasible.empty:
            feasible = sweep.assign(_volume_gap=sweep["certificate_per_100"].sub(scet_volume).abs())
            best = feasible.sort_values(["_volume_gap", "support_precision", "recall_at_5"], ascending=[True, False, False]).iloc[0]
        else:
            best = feasible.sort_values(["support_precision", "recall_at_5", "certificate_per_100"], ascending=False).iloc[0]
        rows.append(
            {
                "dataset": str(frame["dataset"].iloc[0]) if "dataset" in frame and not frame.empty else "",
                "score_column": col,
                "scet_support_precision": scet_precision,
                "scet_supported_per_100": scet_supported,
                "scet_certificate_per_100": scet_volume,
                "min_matched_certificate_per_100": lower_volume,
                "matched_support_precision": float(best["support_precision"]),
                "matched_supported_per_100": float(best["supported_per_100"]),
                "matched_certificate_per_100": float(best["certificate_per_100"]),
                "matched_threshold": float(best["threshold"]),
                "precision_margin_vs_matched": scet_precision - float(best["support_precision"]),
            }
        )
    return pd.DataFrame(rows)


def build_audit_tables(detail: pd.DataFrame, top_k: int = 5, retrieve_k: int = 20) -> dict[str, pd.DataFrame]:
    unique = unique_scet_candidate_rows(detail)
    loop_parts = []
    threshold_parts = []
    matched_parts = []
    for dataset, group in unique.groupby("dataset", sort=False):
        group = group.copy()
        if "bge_m3_score" in group and "cross_encoder_score" in group:
            loop_parts.append(
                apply_retrieval_verification_loop(
                    group,
                    retrieval_col="bge_m3_score",
                    verifier_col="cross_encoder_score",
                    retrieve_k=retrieve_k,
                    top_k=top_k,
                    name="retrieval_verification_loop",
                )
            )
        score_cols = [col for col in [*BASELINE_SCORE_COLUMNS, "scet_support_score"] if col in group.columns]
        for col in score_cols:
            threshold_parts.append(threshold_sweep(group, col, top_k=top_k))
        matched_parts.append(matched_volume_threshold_audit(group, [col for col in BASELINE_SCORE_COLUMNS if col in group.columns], top_k=top_k))
    loop_detail = pd.concat(loop_parts, ignore_index=True) if loop_parts else pd.DataFrame()
    loop_summary = pd.DataFrame()
    loop_stress = pd.DataFrame()
    if not loop_detail.empty:
        loop_summary = pd.DataFrame(
            [
                {"dataset": dataset, "retriever": "retrieval_verification_loop", **evidence_retrieval_metrics(group, top_k=top_k)}
                for dataset, group in loop_detail.groupby("dataset", sort=False)
            ]
        )
        loop_stress = build_conflict_stress_summary(loop_detail)
    threshold_curve = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    return {
        "unique_candidates": unique,
        "verification_loop_detail": loop_detail,
        "verification_loop_summary": loop_summary,
        "verification_loop_stress": loop_stress,
        "threshold_curve": threshold_curve,
        "matched_volume_audit": matched,
    }


def run_audit(detail_path: Path = DEFAULT_DETAIL, out_dir: Path = DEFAULT_OUT, top_k: int = 5, retrieve_k: int = 20) -> dict[str, pd.DataFrame]:
    detail = pd.read_csv(detail_path)
    tables = build_audit_tables(detail, top_k=top_k, retrieve_k=retrieve_k)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCET risk-reduction diagnostics.")
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = run_audit(args.detail, args.out, top_k=args.top_k, retrieve_k=args.retrieve_k)
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    main()
