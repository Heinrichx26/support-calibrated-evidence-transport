from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ipm_cross_domain_evidence_benchmark import (
    evidence_retrieval_metrics,
    load_or_build_candidates,
    split_calibration_evaluation_queries,
)
from ipm_scet_evidence_transport import (
    apply_scet_model,
    apply_baseline_retrievers,
    available_scet_views,
    build_conflict_stress_summary,
    compute_conflict_scores,
    fit_scet_model,
    prepare_scet_scored_candidates,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "ipm_scet_open_pool_audit_final"

DATASET_LABELS = {
    "scifact": "SciFact full large-pool audit",
    "fever": "FEVER open evidence-pair large-pool audit",
    "hover": "HoVer sampled large-pool audit",
}


def parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def audit_candidate_mode(dataset: str, mode: str) -> str:
    if mode == "full" and dataset == "hover":
        return "smoke"
    return mode


def audit_scope(dataset: str, mode: str, candidates_per_query: int) -> str:
    if dataset == "hover" and mode == "full":
        return f"sampled_validation_k{candidates_per_query}"
    if dataset == "fever":
        return f"open_evidence_pair_pool_k{candidates_per_query}"
    return f"large_candidate_pool_k{candidates_per_query}"


def _topk_mask(frame: pd.DataFrame, score_col: str, top_k: int) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("query_id", sort=False):
        idx = group.sort_values(score_col, ascending=False).head(top_k).index
        mask.loc[idx] = True
    return mask


def certificate_metrics(frame: pd.DataFrame, top_k: int = 5) -> dict[str, Any]:
    metrics = evidence_retrieval_metrics(frame, top_k=top_k)
    cert_total = int(pd.to_numeric(frame["evidence_supported"], errors="coerce").fillna(0).astype(int).sum())
    queries = int(frame["query_id"].nunique())
    metrics["certificates"] = cert_total
    metrics["cert_per_100"] = 100.0 * cert_total / max(queries, 1)
    return metrics


def stress_precision(frame: pd.DataFrame) -> float:
    stress = build_conflict_stress_summary(frame)
    if stress.empty:
        return 0.0
    return float(stress["stress_support_precision"].iloc[0])


def summarize_detail(
    detail: pd.DataFrame,
    *,
    audit_mode: str,
    candidate_mode: str,
    candidates_per_query: int,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, retriever), group in detail.groupby(["dataset", "retriever"], sort=False):
        metrics = certificate_metrics(group, top_k=top_k)
        metrics.update(
            {
                "dataset": dataset,
                "retriever": retriever,
                "audit_mode": audit_mode,
                "candidate_mode": candidate_mode,
                "candidate_scope": audit_scope(str(dataset), audit_mode, candidates_per_query),
                "candidates_per_query": int(candidates_per_query),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def threshold_grid(scores: pd.Series) -> list[float]:
    values = pd.to_numeric(scores, errors="coerce").fillna(0.0).astype(float)
    fixed = {0.0, 0.25, 0.5, 0.75, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0}
    quantiles = {float(np.quantile(values, q)) for q in np.linspace(0.0, 1.0, 31)}
    return sorted(fixed | quantiles)


def apply_score_threshold(frame: pd.DataFrame, score_col: str, threshold: float, top_k: int) -> pd.DataFrame:
    out = frame.copy()
    out["retrieval_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
    top_mask = _topk_mask(out, score_col, top_k)
    out["evidence_supported"] = (top_mask & (out[score_col] >= threshold)).astype(int)
    return out


def matched_volume_rows(dataset_detail: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scet = dataset_detail[dataset_detail["retriever"].eq("scet")].copy()
    if scet.empty:
        return pd.DataFrame()
    scet_metrics = certificate_metrics(scet, top_k=top_k)
    scet_stress = stress_precision(scet)
    target_volume = float(scet_metrics["cert_per_100"])
    rows.append(
        {
            "dataset": str(scet["dataset"].iloc[0]),
            "method": "SCET",
            "matched_to": "SCET",
            "threshold": float(scet.get("scet_support_threshold", pd.Series([0.0])).iloc[0]),
            "cert_per_100": target_volume,
            "support_precision": float(scet_metrics["support_precision"]),
            "stress_support_precision": scet_stress,
            "volume_gap": 0.0,
        }
    )
    for retriever, group in dataset_detail[~dataset_detail["retriever"].eq("scet")].groupby("retriever", sort=False):
        work = group.copy()
        score_col = "retrieval_score"
        if "scet_conflict_score" not in work.columns:
            views = available_scet_views(work)
            work["scet_conflict_score"] = compute_conflict_scores(work, views)
        best: dict[str, Any] | None = None
        for threshold in threshold_grid(work[score_col]):
            candidate = apply_score_threshold(work, score_col, threshold, top_k)
            metrics = certificate_metrics(candidate, top_k=top_k)
            stress = stress_precision(candidate)
            cert_per_100 = float(metrics["cert_per_100"])
            row = {
                "dataset": str(work["dataset"].iloc[0]),
                "method": str(retriever),
                "matched_to": "SCET",
                "threshold": float(threshold),
                "cert_per_100": cert_per_100,
                "support_precision": float(metrics["support_precision"]),
                "stress_support_precision": float(stress),
                "volume_gap": abs(cert_per_100 - target_volume),
            }
            key = (row["volume_gap"], -row["support_precision"], -row["stress_support_precision"])
            if best is None or key < (
                float(best["volume_gap"]),
                -float(best["support_precision"]),
                -float(best["stress_support_precision"]),
            ):
                best = row
        if best is not None:
            rows.append(best)
    return pd.DataFrame(rows)


def run_dataset(
    dataset: str,
    *,
    mode: str,
    out: Path,
    neural_level: str,
    candidates_per_query: int,
    calibration_query_fraction: float,
    calibration_seed: int,
    top_k: int,
    semantic_batch_size: int,
    fast_text_scores: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_mode = audit_candidate_mode(dataset, mode)
    candidates = load_or_build_candidates(dataset, candidate_mode, candidates_per_query)
    scored = prepare_scet_scored_candidates(
        candidates,
        neural_level=neural_level,
        semantic_batch_size=semantic_batch_size,
        semantic_cache_key=f"open_pool_{dataset}_{candidate_mode}_{candidates_per_query}",
        fast_text_scores=fast_text_scores,
    )
    calibration, evaluation = split_calibration_evaluation_queries(
        scored,
        calibration_fraction=calibration_query_fraction,
        seed=calibration_seed,
    )
    model = fit_scet_model(calibration, top_k=top_k)
    scet_detail = apply_scet_model(evaluation, model)
    scet_detail["evaluation_partition"] = "evaluation"

    retrievers = ["bm25", "mvec_tg"]
    if not fast_text_scores:
        retrievers.extend(["tfidf_svd", "dense"])
    if neural_level in {"light", "full"}:
        retrievers.extend(["sbert_mini_lm", "e5_small_v2", "mvec_tg_semantic"])
    if neural_level == "full":
        retrievers.extend(["bge_m3", "cross_encoder"])
    baseline_detail = apply_baseline_retrievers(evaluation, calibration, retrievers, top_k=top_k)
    if not baseline_detail.empty:
        baseline_detail["scet_conflict_score"] = compute_conflict_scores(
            baseline_detail,
            available_scet_views(baseline_detail),
        )
    detail = pd.concat([baseline_detail, scet_detail], ignore_index=True) if not baseline_detail.empty else scet_detail
    summary = summarize_detail(
        detail,
        audit_mode=mode,
        candidate_mode=candidate_mode,
        candidates_per_query=candidates_per_query,
        top_k=top_k,
    )
    stress = build_conflict_stress_summary(detail)
    if not stress.empty:
        stress["audit_mode"] = mode
        stress["candidate_mode"] = candidate_mode
        stress["candidate_scope"] = audit_scope(dataset, mode, candidates_per_query)
        stress["candidates_per_query"] = int(candidates_per_query)
    volume = matched_volume_rows(detail, top_k=top_k)
    if not volume.empty:
        volume["audit_mode"] = mode
        volume["candidate_mode"] = candidate_mode
        volume["candidate_scope"] = audit_scope(dataset, mode, candidates_per_query)
        volume["candidates_per_query"] = int(candidates_per_query)
    return summary, volume, stress


def run_open_pool_audit(
    *,
    mode: str,
    datasets: list[str],
    out: Path,
    neural_level: str = "none",
    candidates_per_query: int = 500,
    calibration_query_fraction: float = 0.3,
    calibration_seed: int = 13,
    top_k: int = 5,
    semantic_batch_size: int = 64,
    fast_text_scores: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    summary_parts: list[pd.DataFrame] = []
    volume_parts: list[pd.DataFrame] = []
    stress_parts: list[pd.DataFrame] = []
    for dataset in datasets:
        summary, volume, stress = run_dataset(
            dataset,
            mode=mode,
            out=out,
            neural_level=neural_level,
            candidates_per_query=candidates_per_query,
            calibration_query_fraction=calibration_query_fraction,
            calibration_seed=calibration_seed,
            top_k=top_k,
            semantic_batch_size=semantic_batch_size,
            fast_text_scores=fast_text_scores,
        )
        summary_parts.append(summary)
        volume_parts.append(volume)
        stress_parts.append(stress)
    summary_out = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    volume_out = pd.concat(volume_parts, ignore_index=True) if volume_parts else pd.DataFrame()
    stress_out = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
    summary_out.to_csv(out / "open_pool_summary.csv", index=False)
    volume_out.to_csv(out / "open_pool_volume_audit.csv", index=False)
    stress_out.to_csv(out / "open_pool_stress_summary.csv", index=False)
    return summary_out, volume_out, stress_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--datasets", default="scifact,fever,hover")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--neural-level", choices=["none", "light", "full"], default="none")
    parser.add_argument("--candidates-per-query", type=int, default=500)
    parser.add_argument("--calibration-query-fraction", type=float, default=0.3)
    parser.add_argument("--calibration-seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--fast-text-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, volume, stress = run_open_pool_audit(
        mode=args.mode,
        datasets=parse_list(args.datasets),
        out=args.out,
        neural_level=args.neural_level,
        candidates_per_query=args.candidates_per_query,
        calibration_query_fraction=args.calibration_query_fraction,
        calibration_seed=args.calibration_seed,
        top_k=args.top_k,
        semantic_batch_size=args.semantic_batch_size,
        fast_text_scores=args.fast_text_scores,
    )
    print(summary.to_string(index=False) if not summary.empty else "(no summary rows)")
    if not volume.empty:
        print(volume.to_string(index=False))
    if not stress.empty:
        print(stress.to_string(index=False))


if __name__ == "__main__":
    main()
