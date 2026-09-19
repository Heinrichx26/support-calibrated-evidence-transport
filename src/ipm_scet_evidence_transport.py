from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ipm_cross_domain_evidence_benchmark import (
    SEMANTIC_RETRIEVERS,
    TEXT_RETRIEVERS,
    add_semantic_score_columns,
    apply_retriever,
    evidence_retrieval_metrics,
    load_bts_summary,
    load_or_build_candidates,
    minmax,
    score_text_candidates,
    select_mvec_tg_rule,
    split_calibration_evaluation_queries,
    summarize_text_detail,
)
from ipm_evidence_dataset_adapters import UNIFIED_COLUMNS, validate_unified_schema


PROJECT = Path(__file__).resolve().parents[1]
OUT_SMOKE = PROJECT / "results" / "ipm_scet_smoke"
OUT_FULL = PROJECT / "results" / "ipm_scet_full"
SCET_CACHE = PROJECT / "data" / "interim" / "evidence_retrieval" / "scet_cache"

BASE_VIEW_COLUMNS = [
    "bm25_score",
    "tfidf_score",
    "dense_score",
    "entity_score",
    "relation_score",
    "sbert_mini_lm_score",
    "e5_small_v2_score",
    "bge_m3_score",
    "cross_encoder_score",
    "bts_view_margin_score",
]

SCET_OUTPUT_FILES = [
    "scet_detail.csv",
    "scet_summary.csv",
    "scet_support_certificates.jsonl",
    "scet_ablation_summary.csv",
    "scet_stress_summary.csv",
]

NEGATION_TERMS = {
    "no",
    "not",
    "without",
    "never",
    "none",
    "lack",
    "lacks",
    "lacking",
    "failed",
    "fails",
    "failure",
    "inhibit",
    "inhibits",
    "inhibited",
    "decrease",
    "decreases",
    "decreased",
    "reduced",
    "reduces",
    "absence",
}

CONFLICT_TERMS = {
    "contradict",
    "contradicts",
    "contradicted",
    "opposite",
    "refute",
    "refutes",
    "refuted",
    "inconsistent",
    "unrelated",
}

SEMANTIC_SCORE_COLUMNS = {
    "dense_score",
    "sbert_mini_lm_score",
    "e5_small_v2_score",
    "bge_m3_score",
    "cross_encoder_score",
}


@dataclass(frozen=True)
class SCETModel:
    view_columns: list[str]
    view_reliability: dict[str, float]
    support_threshold: float
    penalty_weight: float
    top_k: int
    selection_mode: str
    calibration_query_ids: list[str]
    calibration_metrics: dict[str, float]
    threshold_grid: list[dict[str, float]]
    portfolio_transport: SupportPortfolioModel | None = None
    semantic_portfolio_mean_weight: float = 0.0


@dataclass(frozen=True)
class SupportPortfolioModel:
    view_columns: list[str]
    feature_weights: dict[str, float]
    feature_intercept: float
    feature_means: dict[str, float]
    feature_scales: dict[str, float]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _term_overlap(left: str, right: str) -> float:
    left_terms = set(_tokenize(left))
    right_terms = set(_tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / math.sqrt(len(left_terms) * len(right_terms))


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _group_minmax(values: pd.Series, groups: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=values.index, dtype=float)
    for _, idx in groups.groupby(groups, sort=False).groups.items():
        arr = values.loc[idx].to_numpy(float)
        out.loc[idx] = minmax(arr)
    return out.clip(0.0, 1.0)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.5
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    pos_rank_sum = float(ranks[pos].sum())
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    if not np.isfinite(auc):
        return 0.5
    return float(np.clip(auc, 0.0, 1.0))


def _topk_mask(frame: pd.DataFrame, score_col: str, top_k: int) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("query_id", sort=False):
        idx = group.sort_values(score_col, ascending=False).head(top_k).index
        mask.loc[idx] = True
    return mask


def _support_metrics_for_threshold(frame: pd.DataFrame, score_col: str, threshold: float, top_k: int) -> dict[str, Any]:
    work = frame.copy()
    top_mask = _topk_mask(work, score_col, top_k)
    work["retrieval_score"] = work[score_col]
    work["evidence_supported"] = (top_mask & (work[score_col] >= threshold)).astype(int)
    return evidence_retrieval_metrics(work, top_k=top_k)


def available_scet_views(frame: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    candidates = list(requested) if requested is not None else BASE_VIEW_COLUMNS
    views = []
    for col in candidates:
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().any():
            views.append(col)
    return views


def validate_scet_schema(frame: pd.DataFrame) -> pd.DataFrame:
    out = validate_unified_schema(frame)
    if out["query_id"].nunique() == 0:
        raise ValueError("SCET requires at least one query_id")
    if out["candidate_id"].nunique() == 0:
        raise ValueError("SCET requires at least one candidate_id")
    return out


def add_bge_m3_scores(
    scored: pd.DataFrame,
    batch_size: int = 32,
    cache_key: str = "scet_bge_m3",
    cache_dir: Path = SCET_CACHE,
) -> pd.DataFrame:
    if scored.empty or "bge_m3_score" in scored.columns:
        return scored
    try:
        from ipm_cross_domain_evidence_benchmark import _encode_texts
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError("BGE-M3 score generation needs the benchmark embedding helper") from exc
    out = scored.copy()
    query_lookup = out.drop_duplicates("query_id")[["query_id", "query_text"]].reset_index(drop=True)
    doc_texts = pd.Series(out["evidence_text"].astype(str).unique()).tolist()
    doc_index = {text: idx for idx, text in enumerate(doc_texts)}
    query_ids = query_lookup["query_id"].astype(str).tolist()
    query_index = {query_id: idx for idx, query_id in enumerate(query_ids)}
    query_embeddings = _encode_texts(
        query_lookup["query_text"].astype(str).tolist(),
        "BAAI/bge-m3",
        "",
        batch_size,
        cache_dir,
        f"{cache_key}_query",
    )
    doc_embeddings = _encode_texts(
        doc_texts,
        "BAAI/bge-m3",
        "",
        batch_size,
        cache_dir,
        f"{cache_key}_evidence",
    )
    row_query_idx = out["query_id"].astype(str).map(query_index).to_numpy(dtype=int)
    row_doc_idx = out["evidence_text"].astype(str).map(doc_index).to_numpy(dtype=int)
    scores = (query_embeddings[row_query_idx] * doc_embeddings[row_doc_idx]).sum(axis=1)
    out["bge_m3_score"] = _group_minmax(pd.Series(scores, index=out.index), out["query_id"].astype(str))
    return out


def add_cross_encoder_scores(
    scored: pd.DataFrame,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    batch_size: int = 32,
    cache_key: str = "scet_cross_encoder",
    cache_dir: Path = SCET_CACHE,
) -> pd.DataFrame:
    if scored.empty or "cross_encoder_score" in scored.columns:
        return scored
    pair_hasher = hashlib.sha1()
    pair_hasher.update(model_name.encode("utf-8"))
    for query, evidence in zip(scored["query_text"].astype(str), scored["evidence_text"].astype(str)):
        pair_hasher.update(query.encode("utf-8", errors="ignore"))
        pair_hasher.update(b"\0")
        pair_hasher.update(evidence.encode("utf-8", errors="ignore"))
        pair_hasher.update(b"\0")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}_{safe_model}_{pair_hasher.hexdigest()[:16]}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        scores = cached["scores"]
        if len(scores) == len(scored):
            out = scored.copy()
            out["cross_encoder_score"] = _group_minmax(pd.Series(scores, index=out.index), out["query_id"].astype(str))
            return out
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cross-encoder scoring needs sentence-transformers") from exc
    model = CrossEncoder(model_name)
    pairs = list(zip(scored["query_text"].astype(str).tolist(), scored["evidence_text"].astype(str).tolist()))
    scores = np.asarray(model.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=float)
    np.savez_compressed(cache_path, scores=scores.astype("float32"))
    out = scored.copy()
    out["cross_encoder_score"] = _group_minmax(pd.Series(scores, index=out.index), out["query_id"].astype(str))
    return out


def prepare_scet_scored_candidates(
    candidates: pd.DataFrame,
    neural_level: str = "none",
    semantic_batch_size: int = 64,
    semantic_cache_key: str = "scet",
    fast_text_scores: bool = False,
) -> pd.DataFrame:
    base_retrievers = ["bm25", "tfidf_svd", "dense"]
    if neural_level in {"light", "full"}:
        base_retrievers.extend(["sbert_mini_lm", "e5_small_v2"])
    scored = score_text_candidates(
        validate_scet_schema(candidates),
        base_retrievers,
        semantic_batch_size=semantic_batch_size,
        semantic_cache_key=semantic_cache_key,
        fast_text_scores=fast_text_scores,
    )
    if neural_level == "full":
        try:
            scored = add_bge_m3_scores(scored, batch_size=max(8, semantic_batch_size // 2), cache_key=semantic_cache_key)
        except Exception as exc:
            scored["bge_m3_error"] = str(exc)
        try:
            scored = add_cross_encoder_scores(scored, batch_size=max(8, semantic_batch_size // 2), cache_key=semantic_cache_key)
        except Exception as exc:
            scored["cross_encoder_error"] = str(exc)
    return scored


def compute_conflict_scores(frame: pd.DataFrame, view_columns: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    view_frame = pd.DataFrame({col: _safe_numeric(frame, col) for col in view_columns}, index=frame.index)
    if view_frame.empty:
        view_frame = pd.DataFrame({"fallback": pd.Series(0.0, index=frame.index)})
    view_disagreement = view_frame.std(axis=1).fillna(0.0).clip(0.0, 1.0)
    semantic_cols = [col for col in ["dense_score", "sbert_mini_lm_score", "e5_small_v2_score", "bge_m3_score", "cross_encoder_score"] if col in frame]
    if semantic_cols:
        semantic_strength = pd.concat([_safe_numeric(frame, col) for col in semantic_cols], axis=1).max(axis=1)
    else:
        semantic_strength = _safe_numeric(frame, "dense_score")
    structural_cols = [col for col in ["entity_score", "relation_score"] if col in frame]
    if structural_cols:
        structural_strength = pd.concat([_safe_numeric(frame, col) for col in structural_cols], axis=1).max(axis=1)
    else:
        structural_strength = pd.Series(0.0, index=frame.index)
    high_similarity_unanchored = (semantic_strength - structural_strength).clip(0.0, 1.0)

    query_neg = frame["query_text"].astype(str).map(lambda text: len(set(_tokenize(text)) & NEGATION_TERMS))
    evidence_terms = frame["evidence_text"].astype(str).map(lambda text: set(_tokenize(text)))
    evidence_neg = evidence_terms.map(lambda terms: len(terms & NEGATION_TERMS))
    negation_mismatch = ((query_neg > 0) ^ (evidence_neg > 0)).astype(float)
    conflict_word = evidence_terms.map(lambda terms: float(bool(terms & CONFLICT_TERMS)))
    raw = (
        0.35 * view_disagreement
        + 0.35 * high_similarity_unanchored
        + 0.15 * negation_mismatch
        + 0.15 * conflict_word
    )
    return raw.clip(0.0, 1.0)


def _family_columns(view_columns: list[str], family: set[str]) -> list[str]:
    return [col for col in view_columns if col in family]


def _row_second_largest(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(0.0, index=frame.index, dtype=float)
    values = frame.to_numpy(float)
    if values.shape[1] == 1:
        return pd.Series(values[:, 0], index=frame.index, dtype=float)
    sorted_values = np.sort(values, axis=1)
    return pd.Series(sorted_values[:, -2], index=frame.index, dtype=float)


def build_support_portfolio_features(frame: pd.DataFrame, view_columns: list[str]) -> pd.DataFrame:
    usable = available_scet_views(frame, view_columns)
    if not usable:
        raise ValueError("Support portfolio transport requires at least one view column")
    features = pd.DataFrame(index=frame.index)
    for col in usable:
        features[f"view::{col}"] = _safe_numeric(frame, col)

    semantic_cols = _family_columns(usable, SEMANTIC_SCORE_COLUMNS)
    lexical_cols = _family_columns(usable, {"bm25_score", "tfidf_score", "dense_score"})
    anchor_cols = [col for col in ["entity_score", "relation_score", "source_score", "bts_view_margin_score"] if col in frame.columns]

    semantic_frame = pd.DataFrame({col: _safe_numeric(frame, col) for col in semantic_cols}, index=frame.index)
    lexical_frame = pd.DataFrame({col: _safe_numeric(frame, col) for col in lexical_cols}, index=frame.index)
    anchor_frame = pd.DataFrame({col: _safe_numeric(frame, col) for col in anchor_cols}, index=frame.index)
    all_view_frame = pd.DataFrame({col: _safe_numeric(frame, col) for col in usable}, index=frame.index)

    if semantic_frame.empty:
        semantic_frame = all_view_frame
    if lexical_frame.empty:
        lexical_frame = all_view_frame
    if anchor_frame.empty:
        anchor_frame = pd.DataFrame({"anchor_fallback": pd.Series(0.0, index=frame.index)})

    semantic_max = semantic_frame.max(axis=1)
    semantic_mean = semantic_frame.mean(axis=1)
    semantic_second = _row_second_largest(semantic_frame)
    lexical_max = lexical_frame.max(axis=1)
    anchor_mean = anchor_frame.mean(axis=1)
    anchor_max = anchor_frame.max(axis=1)
    anchor_min = anchor_frame.min(axis=1)
    conflict = compute_conflict_scores(frame, usable)
    disagreement = all_view_frame.std(axis=1).fillna(0.0)

    features["semantic_max"] = semantic_max
    features["semantic_mean"] = semantic_mean
    features["semantic_second"] = semantic_second
    features["lexical_max"] = lexical_max
    features["anchor_mean"] = anchor_mean
    features["anchor_max"] = anchor_max
    features["anchor_min"] = anchor_min
    features["view_disagreement"] = disagreement
    features["conflict_score"] = conflict
    features["semantic_anchor_product"] = semantic_max * anchor_mean
    features["semantic_second_anchor_product"] = semantic_second * anchor_mean
    features["semantic_conflict_residual"] = semantic_max * (1.0 - conflict)
    features["anchor_conflict_residual"] = anchor_mean * (1.0 - conflict)
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)


def _standardize_features(features: pd.DataFrame, means: dict[str, float], scales: dict[str, float]) -> pd.DataFrame:
    ordered = list(means)
    aligned = pd.DataFrame(index=features.index)
    for col in ordered:
        aligned[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0) if col in features else 0.0
        aligned[col] = (aligned[col].astype(float) - float(means[col])) / max(float(scales.get(col, 1.0)), 1e-9)
    return aligned


def fit_support_portfolio_transport(calibration: pd.DataFrame, view_columns: list[str]) -> SupportPortfolioModel:
    usable = available_scet_views(calibration, view_columns)
    features = build_support_portfolio_features(calibration, usable)
    y = pd.to_numeric(calibration["gold_support"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    means = {col: float(features[col].mean()) for col in features.columns}
    scales = {col: float(features[col].std(ddof=0) or 1.0) for col in features.columns}
    standardized = _standardize_features(features, means, scales)
    if y.nunique() < 2:
        fallback_weights = {col: 0.0 for col in standardized.columns}
        return SupportPortfolioModel(usable, fallback_weights, 0.0, means, scales)
    try:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=13, solver="liblinear")
        classifier.fit(standardized.to_numpy(float), y.to_numpy(int))
        weights = {col: float(value) for col, value in zip(standardized.columns, classifier.coef_[0])}
        intercept = float(classifier.intercept_[0])
    except Exception:
        reliability = learn_view_reliability(calibration, usable, top_k=5)
        weights = {col: 0.0 for col in standardized.columns}
        for col, value in reliability.items():
            feature = f"view::{col}"
            if feature in weights:
                weights[feature] = float(value)
        intercept = -0.5
    return SupportPortfolioModel(usable, weights, intercept, means, scales)


def apply_support_portfolio_transport(
    frame: pd.DataFrame,
    portfolio: SupportPortfolioModel,
    score_column: str = "scet_support_score",
) -> pd.DataFrame:
    out = frame.copy()
    features = build_support_portfolio_features(out, portfolio.view_columns)
    standardized = _standardize_features(features, portfolio.feature_means, portfolio.feature_scales)
    logits = pd.Series(float(portfolio.feature_intercept), index=out.index, dtype=float)
    for col in standardized.columns:
        logits = logits + standardized[col] * float(portfolio.feature_weights.get(col, 0.0))
    logits = logits.clip(-30.0, 30.0)
    out["scet_conflict_score"] = features["conflict_score"].clip(0.0, 1.0)
    out[score_column] = 1.0 / (1.0 + np.exp(-logits))
    return out


def apply_semantic_portfolio_transport(
    frame: pd.DataFrame,
    view_columns: list[str],
    mean_weight: float = 0.0,
    score_column: str = "scet_support_score",
) -> pd.DataFrame:
    out = frame.copy()
    semantic_cols = [col for col in view_columns if col in SEMANTIC_SCORE_COLUMNS and col in out.columns]
    if not semantic_cols:
        semantic_cols = [col for col in view_columns if col in out.columns]
    if not semantic_cols:
        raise ValueError("Semantic portfolio transport requires at least one score column")
    semantic = pd.DataFrame({col: _safe_numeric(out, col) for col in semantic_cols}, index=out.index)
    semantic_max = semantic.max(axis=1)
    semantic_mean = semantic.mean(axis=1)
    semantic_second = _row_second_largest(semantic)
    conflict = compute_conflict_scores(out, available_scet_views(out, view_columns))
    raw = semantic_max + float(mean_weight) * semantic_mean + 0.05 * semantic_second - 0.03 * conflict
    out["scet_conflict_score"] = conflict
    out[score_column] = _group_minmax(raw, out["query_id"].astype(str))
    return out


def learn_view_reliability(calibration: pd.DataFrame, view_columns: list[str], top_k: int = 5) -> dict[str, float]:
    if not view_columns:
        raise ValueError("SCET requires at least one view column")
    y = pd.to_numeric(calibration["gold_support"], errors="coerce").fillna(0).astype(int).to_numpy()
    qualities: dict[str, float] = {}
    for col in view_columns:
        scores = _safe_numeric(calibration, col).to_numpy(float)
        pos_scores = scores[y == 1]
        neg_scores = scores[y == 0]
        separation = float(pos_scores.mean() - neg_scores.mean()) if len(pos_scores) and len(neg_scores) else 0.0
        auc = _safe_auc(y, scores)
        top_support = _topk_mask(calibration.assign(_tmp_score=scores), "_tmp_score", top_k)
        top_total = int(top_support.sum())
        top_precision = float(((top_support.to_numpy(bool)) & (y == 1)).sum() / max(top_total, 1))
        quality = 0.05 + max(separation, 0.0) + max(auc - 0.5, 0.0) * 1.5 + top_precision
        qualities[col] = float(max(quality, 1e-6))
    total = sum(qualities.values())
    return {col: qualities[col] / max(total, 1e-12) for col in view_columns}


def apply_support_transport(
    frame: pd.DataFrame,
    view_columns: list[str],
    view_reliability: dict[str, float],
    penalty_weight: float,
    score_column: str = "scet_support_score",
) -> pd.DataFrame:
    out = frame.copy()
    if not view_columns:
        raise ValueError("SCET support transport requires view columns")
    weighted = pd.Series(0.0, index=out.index, dtype=float)
    total = 0.0
    for col in view_columns:
        weight = float(view_reliability.get(col, 0.0))
        if weight <= 0:
            continue
        weighted = weighted + weight * _safe_numeric(out, col)
        total += weight
    weighted = weighted / max(total, 1e-12)
    out["scet_conflict_score"] = compute_conflict_scores(out, view_columns)
    raw = weighted - float(penalty_weight) * out["scet_conflict_score"]
    out[score_column] = _group_minmax(raw, out["query_id"].astype(str))
    return out


def select_support_threshold(
    calibration: pd.DataFrame,
    score_col: str = "scet_support_score",
    top_k: int = 5,
    precision_weight: float = 2.0,
    min_supported_per_100: float = 50.0,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    scores = pd.to_numeric(calibration[score_col], errors="coerce").fillna(0.0)
    quantiles = np.linspace(0.0, 1.0, 31)
    fixed_thresholds = {0.0, 0.25, 0.5, 0.75, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0}
    grid = sorted({float(np.quantile(scores, q)) for q in quantiles} | fixed_thresholds)
    best_threshold = 0.0
    best_metrics: dict[str, float] = {}
    best_key: tuple[int, float, float, float, float] | None = None
    rows: list[dict[str, float]] = []
    for threshold in grid:
        metrics = _support_metrics_for_threshold(calibration, score_col, threshold, top_k)
        supported_positive = float(metrics["supported_positive"])
        support_precision = float(metrics["support_precision"])
        evidence_hit = float(metrics["evidence_hit"])
        recall_at_5 = float(metrics.get("recall_at_5", 0.0))
        mrr = float(metrics.get("mrr", 0.0))
        ndcg_at_10 = float(metrics.get("ndcg_at_10", 0.0))
        supported_per_100 = float(metrics["supported_per_100"])
        if supported_positive <= 0:
            objective = -1.0
        else:
            objective = evidence_hit + precision_weight * support_precision + 0.01 * supported_per_100
        row = {
            "threshold": float(threshold),
            "objective": float(objective),
            "support_precision": support_precision,
            "evidence_hit": evidence_hit,
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "ndcg_at_10": ndcg_at_10,
            "supported_per_100": supported_per_100,
            "supported_positive": supported_positive,
        }
        rows.append(row)
        feasible = int(supported_positive > 0 and supported_per_100 >= min_supported_per_100)
        key = (feasible, support_precision, evidence_hit, supported_per_100, objective)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {k: float(v) for k, v in row.items() if k != "threshold"}
    return best_threshold, best_metrics, rows


def fit_scet_model(
    calibration: pd.DataFrame,
    view_columns: list[str] | None = None,
    top_k: int = 5,
    penalty_weight: float = 0.25,
    precision_weight: float = 2.0,
    max_calibration_recall_drop: float = 0.015,
) -> SCETModel:
    usable_views = available_scet_views(calibration, view_columns)
    reliability = learn_view_reliability(calibration, usable_views, top_k=top_k)
    query_ids = sorted(calibration["query_id"].astype(str).unique().tolist())
    if not usable_views:
        raise RuntimeError("SCET could not select a support transport model")

    def candidate_specs(frame: pd.DataFrame) -> list[dict[str, Any]]:
        frame_reliability = learn_view_reliability(frame, usable_views, top_k=top_k)
        specs: list[dict[str, Any]] = [
            {
                "mode": "support_portfolio_transport",
                "cols": usable_views,
                "weights": frame_reliability,
                "penalty": 0.0,
                "portfolio": fit_support_portfolio_transport(frame, usable_views),
            },
            {
                "mode": "calibrated_transport",
                "cols": usable_views,
                "weights": frame_reliability,
                "penalty": float(penalty_weight),
                "portfolio": None,
            },
            {
                "mode": "no_conflict_transport",
                "cols": usable_views,
                "weights": frame_reliability,
                "penalty": 0.0,
                "portfolio": None,
            },
        ]
        specs.append(
            {
                "mode": "equal_transport",
                "cols": usable_views,
                "weights": {col: 1.0 / len(usable_views) for col in usable_views},
                "penalty": float(penalty_weight),
                "portfolio": None,
                "semantic_mean_weight": 0.0,
            }
        )
        semantic_views = [col for col in usable_views if col in SEMANTIC_SCORE_COLUMNS]
        if len(semantic_views) >= 2:
            semantic_reliability = learn_view_reliability(frame, semantic_views, top_k=top_k)
            specs.append(
                {
                    "mode": "semantic_reliability_transport",
                    "cols": semantic_views,
                    "weights": semantic_reliability,
                    "penalty": 0.0,
                    "portfolio": None,
                    "semantic_mean_weight": 0.0,
                }
            )
            specs.append(
                {
                    "mode": "semantic_equal_transport",
                    "cols": semantic_views,
                    "weights": {col: 1.0 / len(semantic_views) for col in semantic_views},
                    "penalty": 0.0,
                    "portfolio": None,
                    "semantic_mean_weight": 0.0,
                }
            )
        for col in usable_views:
            if col not in SEMANTIC_SCORE_COLUMNS:
                continue
            single_penalty = 0.0 if col in SEMANTIC_SCORE_COLUMNS else float(penalty_weight)
            specs.append(
                {
                    "mode": f"single_view_{col}",
                    "cols": [col],
                    "weights": {col: 1.0},
                    "penalty": single_penalty,
                    "portfolio": None,
                    "semantic_mean_weight": 0.0,
                }
            )
        return specs

    if len(query_ids) >= 6:
        selection_train, selection_valid = split_calibration_evaluation_queries(calibration, calibration_fraction=0.6, seed=7919)
    else:
        selection_train, selection_valid = calibration.copy(), calibration.copy()

    selected_spec: dict[str, Any] | None = None
    selection_rows: list[tuple[int, dict[str, Any], dict[str, float]]] = []
    for offset, spec in enumerate(candidate_specs(selection_train)):
        mode = str(spec["mode"])
        cols = list(spec["cols"])
        weights = dict(spec["weights"])
        candidate_penalty = float(spec["penalty"])
        portfolio = spec.get("portfolio")
        semantic_mean_weight = float(spec.get("semantic_mean_weight", 0.0))
        if mode == "support_portfolio_transport" and portfolio is not None:
            train_transport = apply_support_portfolio_transport(selection_train, portfolio)
        elif mode.startswith("semantic_portfolio_"):
            train_transport = apply_semantic_portfolio_transport(selection_train, cols, mean_weight=semantic_mean_weight)
        else:
            train_transport = apply_support_transport(selection_train, cols, weights, candidate_penalty)
        threshold, _, _ = select_support_threshold(
            train_transport,
            score_col="scet_support_score",
            top_k=top_k,
            precision_weight=precision_weight,
        )
        valid_model = SCETModel(
            view_columns=cols,
            view_reliability=weights,
            support_threshold=threshold,
            penalty_weight=float(candidate_penalty),
            top_k=int(top_k),
            selection_mode=mode,
            calibration_query_ids=sorted(selection_train["query_id"].astype(str).unique().tolist()),
            calibration_metrics={},
            threshold_grid=[],
            portfolio_transport=portfolio,
            semantic_portfolio_mean_weight=semantic_mean_weight,
        )
        valid_detail = apply_scet_model(selection_valid, valid_model)
        valid_metrics = {key: float(value) for key, value in evidence_retrieval_metrics(valid_detail, top_k=top_k).items()}
        selection_rows.append((offset, spec, valid_metrics))

    best_recall = max(float(metrics.get("recall_at_5", 0.0)) for *_, metrics in selection_rows)
    selected_key: tuple[int, float, float, float, float, int] | None = None
    for offset, spec, metrics in selection_rows:
        recall = float(metrics.get("recall_at_5", 0.0))
        support_precision = float(metrics.get("support_precision", 0.0))
        evidence_hit = float(metrics.get("evidence_hit", 0.0))
        supported_per_100 = float(metrics.get("supported_per_100", 0.0))
        recall_feasible = int(recall >= best_recall - float(max_calibration_recall_drop))
        key = (recall_feasible, support_precision, recall, evidence_hit, supported_per_100, -offset)
        if selected_key is None or key > selected_key:
            selected_spec = spec
            selected_key = key
    if selected_spec is None:
        raise RuntimeError("SCET could not select a support transport model")
    selected_mode = str(selected_spec["mode"])
    selected_cols = list(selected_spec["cols"])
    selected_penalty = float(selected_spec["penalty"])
    selected_semantic_mean_weight = float(selected_spec.get("semantic_mean_weight", 0.0))
    final_weights = reliability
    if selected_mode == "equal_transport":
        final_weights = {col: 1.0 / len(usable_views) for col in usable_views}
    elif selected_mode == "semantic_equal_transport":
        final_weights = {col: 1.0 / len(selected_cols) for col in selected_cols}
    elif selected_mode == "semantic_reliability_transport":
        final_weights = learn_view_reliability(calibration, selected_cols, top_k=top_k)
    elif selected_mode.startswith("single_view_"):
        selected_view = selected_mode.removeprefix("single_view_")
        selected_cols = [selected_view]
        final_weights = {selected_view: 1.0}
    final_portfolio = None
    if selected_mode == "support_portfolio_transport":
        selected_cols = usable_views
        final_portfolio = fit_support_portfolio_transport(calibration, selected_cols)
        final_transport = apply_support_portfolio_transport(calibration, final_portfolio)
    elif selected_mode.startswith("semantic_portfolio_"):
        final_transport = apply_semantic_portfolio_transport(calibration, selected_cols, mean_weight=selected_semantic_mean_weight)
    else:
        final_transport = apply_support_transport(calibration, selected_cols, final_weights, selected_penalty)
    threshold, metrics, grid = select_support_threshold(
        final_transport,
        score_col="scet_support_score",
        top_k=top_k,
        precision_weight=precision_weight,
        min_supported_per_100=25.0,
    )
    return SCETModel(
        view_columns=selected_cols,
        view_reliability=final_weights,
        support_threshold=threshold,
        penalty_weight=float(selected_penalty),
        top_k=int(top_k),
        selection_mode=selected_mode,
        calibration_query_ids=query_ids,
        calibration_metrics=metrics,
        threshold_grid=grid,
        portfolio_transport=final_portfolio,
        semantic_portfolio_mean_weight=selected_semantic_mean_weight,
    )


def apply_scet_model(frame: pd.DataFrame, model: SCETModel, retriever_name: str = "scet") -> pd.DataFrame:
    if model.selection_mode == "support_portfolio_transport" and model.portfolio_transport is not None:
        out = apply_support_portfolio_transport(frame, model.portfolio_transport)
    elif model.selection_mode.startswith("semantic_portfolio_"):
        out = apply_semantic_portfolio_transport(frame, model.view_columns, mean_weight=model.semantic_portfolio_mean_weight)
    else:
        out = apply_support_transport(frame, model.view_columns, model.view_reliability, model.penalty_weight)
    out["retrieval_score"] = out["scet_support_score"]
    rank = out.groupby("query_id")["retrieval_score"].rank(method="first", ascending=False).astype(int)
    out["rank"] = rank
    out["top_k"] = model.top_k
    out["evidence_supported"] = ((rank <= model.top_k) & (out["scet_support_score"] >= model.support_threshold)).astype(int)
    out["retriever"] = retriever_name
    out["scet_support_threshold"] = model.support_threshold
    out["scet_view_reliability_json"] = json.dumps(model.view_reliability, sort_keys=True)
    out["scet_selection_mode"] = model.selection_mode
    return out


def fit_and_apply_scet(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    view_columns: list[str] | None = None,
    top_k: int = 5,
    penalty_weight: float = 0.25,
    precision_weight: float = 2.0,
) -> tuple[pd.DataFrame, SCETModel]:
    model = fit_scet_model(
        calibration,
        view_columns=view_columns,
        top_k=top_k,
        penalty_weight=penalty_weight,
        precision_weight=precision_weight,
    )
    return apply_scet_model(evaluation, model), model


def _mvec_rule_lookup(calibration: pd.DataFrame, retrievers: list[str]) -> dict[str, dict[str, Any]]:
    rule_lookup: dict[str, dict[str, Any]] = {}
    base_views = ["bm25_score", "tfidf_score", "dense_score", "entity_score", "relation_score"]
    semantic_views = [str(spec["score_column"]) for spec in SEMANTIC_RETRIEVERS.values() if str(spec["score_column"]) in calibration.columns]
    if "mvec_tg" in retrievers:
        rule_lookup["mvec_tg"] = select_mvec_tg_rule(
            calibration,
            None,
            [col for col in base_views if col in calibration.columns],
            allow_weighted_consensus=True,
        )
    if "mvec_tg_semantic" in retrievers:
        rule_lookup["mvec_tg_semantic"] = select_mvec_tg_rule(
            calibration,
            None,
            [col for col in [*base_views, *semantic_views] if col in calibration.columns],
            allow_weighted_consensus=True,
        )
    return rule_lookup


def apply_baseline_retrievers(evaluation: pd.DataFrame, calibration: pd.DataFrame, retrievers: list[str], top_k: int) -> pd.DataFrame:
    rule_lookup = _mvec_rule_lookup(calibration, retrievers)
    parts = []
    for retriever in retrievers:
        if retriever not in TEXT_RETRIEVERS:
            continue
        if retriever == "tfidf_svd" and "tfidf_score" not in evaluation.columns:
            continue
        if retriever == "dense" and "dense_score" not in evaluation.columns:
            continue
        if retriever == "bge_m3" and "bge_m3_score" not in evaluation.columns:
            continue
        if retriever == "cross_encoder" and "cross_encoder_score" not in evaluation.columns:
            continue
        if retriever == "nca_metric" and not {"tfidf_score", "dense_score", "entity_score"}.issubset(evaluation.columns):
            continue
        if retriever == "calibrated_fusion" and not {"bm25_score", "tfidf_score", "dense_score", "relation_score"}.issubset(
            evaluation.columns
        ):
            continue
        if retriever in SEMANTIC_RETRIEVERS:
            score_col = str(SEMANTIC_RETRIEVERS[retriever]["score_column"])
            if score_col not in evaluation.columns:
                continue
        part = apply_retriever(evaluation, retriever, preselected_rule=rule_lookup.get(retriever))
        part["rank"] = part.groupby("query_id")["retrieval_score"].rank(method="first", ascending=False).astype(int)
        part["top_k"] = top_k
        part["evaluation_partition"] = "evaluation"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_scet_ablation_summary(calibration: pd.DataFrame, evaluation: pd.DataFrame, model: SCETModel) -> pd.DataFrame:
    variants: list[tuple[str, SCETModel]] = [("scet", model)]
    if model.view_columns:
        equal = {col: 1.0 / len(model.view_columns) for col in model.view_columns}
        equal_cal = apply_support_transport(calibration, model.view_columns, equal, model.penalty_weight)
        tau, metrics, grid = select_support_threshold(equal_cal, top_k=model.top_k)
        variants.append(
            (
                "scet_equal_views",
                SCETModel(
                    view_columns=model.view_columns,
                    view_reliability=equal,
                    support_threshold=tau,
                    penalty_weight=model.penalty_weight,
                    top_k=model.top_k,
                    selection_mode="equal_transport",
                    calibration_query_ids=model.calibration_query_ids,
                    calibration_metrics=metrics,
                    threshold_grid=grid,
                ),
            )
        )
        no_conflict_cal = apply_support_transport(calibration, model.view_columns, model.view_reliability, 0.0)
        tau, metrics, grid = select_support_threshold(no_conflict_cal, top_k=model.top_k)
        variants.append(
            (
                "scet_no_conflict_penalty",
                SCETModel(
                    view_columns=model.view_columns,
                    view_reliability=model.view_reliability,
                    support_threshold=tau,
                    penalty_weight=0.0,
                    top_k=model.top_k,
                    selection_mode="no_conflict_transport",
                    calibration_query_ids=model.calibration_query_ids,
                    calibration_metrics=metrics,
                    threshold_grid=grid,
                ),
            )
        )
        best_view = max(model.view_reliability, key=model.view_reliability.get)
        single_cal = apply_support_transport(calibration, [best_view], {best_view: 1.0}, model.penalty_weight)
        tau, metrics, grid = select_support_threshold(single_cal, top_k=model.top_k)
        variants.append(
            (
                f"scet_best_single_{best_view}",
                SCETModel(
                    view_columns=[best_view],
                    view_reliability={best_view: 1.0},
                    support_threshold=tau,
                    penalty_weight=model.penalty_weight,
                    top_k=model.top_k,
                    selection_mode=f"single_view_{best_view}",
                    calibration_query_ids=model.calibration_query_ids,
                    calibration_metrics=metrics,
                    threshold_grid=grid,
                ),
            )
        )
    rows = []
    for name, variant in variants:
        detail = apply_scet_model(evaluation, variant, retriever_name=name)
        metrics = evidence_retrieval_metrics(detail, top_k=variant.top_k)
        metrics.update(
            {
                "dataset": str(evaluation["dataset"].iloc[0]) if not evaluation.empty else "",
                "variant": name,
                "selection_mode": variant.selection_mode,
                "support_threshold": variant.support_threshold,
                "penalty_weight": variant.penalty_weight,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_conflict_stress_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty or "scet_conflict_score" not in detail.columns:
        return pd.DataFrame()
    rows = []
    for (dataset, retriever), group in detail.groupby(["dataset", "retriever"], sort=False):
        work = group.copy()
        work["gold_support"] = pd.to_numeric(work["gold_support"], errors="coerce").fillna(0).astype(int)
        work["evidence_supported"] = pd.to_numeric(work["evidence_supported"], errors="coerce").fillna(0).astype(int)
        high_conflict = work["scet_conflict_score"] >= work["scet_conflict_score"].quantile(0.75)
        high_similarity = pd.to_numeric(work.get("retrieval_score", 0.0), errors="coerce").fillna(0.0) >= work["retrieval_score"].quantile(0.75)
        stress_pool = work[high_conflict | (high_similarity & work["gold_support"].eq(0))]
        if stress_pool.empty:
            stress_pool = work
        supported = stress_pool[stress_pool["evidence_supported"].eq(1)]
        false_supported = int((supported["gold_support"].eq(0)).sum()) if not supported.empty else 0
        true_supported = int((supported["gold_support"].eq(1)).sum()) if not supported.empty else 0
        rows.append(
            {
                "dataset": dataset,
                "retriever": retriever,
                "stress_rows": int(len(stress_pool)),
                "stress_supported": int(len(supported)),
                "stress_true_supported": true_supported,
                "stress_false_supported": false_supported,
                "stress_support_precision": true_supported / max(true_supported + false_supported, 1),
                "mean_conflict_score": float(stress_pool["scet_conflict_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def support_certificates(detail: pd.DataFrame, model: SCETModel, per_query: int = 1, max_rows: int = 500) -> list[dict[str, Any]]:
    if detail.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, group in detail[detail["retriever"].eq("scet")].groupby("query_id", sort=False):
        selected = group.sort_values(["evidence_supported", "scet_support_score"], ascending=[False, False]).head(per_query)
        for row in selected.itertuples(index=False):
            view_scores = {col: float(getattr(row, col, 0.0)) for col in model.view_columns if hasattr(row, col)}
            rows.append(
                {
                    "dataset": str(getattr(row, "dataset")),
                    "split": str(getattr(row, "split")),
                    "query_id": str(getattr(row, "query_id")),
                    "candidate_id": str(getattr(row, "candidate_id")),
                    "query_text": str(getattr(row, "query_text")),
                    "top_evidence": str(getattr(row, "evidence_text")),
                    "gold_support": int(getattr(row, "gold_support")),
                    "evidence_supported": int(getattr(row, "evidence_supported")),
                    "rank": int(getattr(row, "rank")),
                    "calibrated_support_score": float(getattr(row, "scet_support_score")),
                    "support_threshold": float(model.support_threshold),
                    "conflict_score": float(getattr(row, "scet_conflict_score")),
                    "view_reliability": model.view_reliability,
                    "view_scores": view_scores,
                }
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_scet_detail(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    summary = summarize_text_detail(detail)
    rows = []
    for dataset, group in summary.groupby("dataset", sort=False):
        baselines = group[~group["retriever"].astype(str).str.startswith("scet")]
        scet = group[group["retriever"].eq("scet")]
        if baselines.empty or scet.empty:
            rows.append(group)
            continue
        best_recall = float(baselines["recall_at_5"].max())
        best_precision = float(baselines["support_precision"].max())
        best_supported = float(baselines["supported_per_100"].max())
        adjusted = group.copy()
        mask = adjusted["retriever"].eq("scet")
        adjusted.loc[mask, "recall_at_5_delta_vs_best_baseline"] = adjusted.loc[mask, "recall_at_5"] - best_recall
        adjusted.loc[mask, "support_precision_delta_vs_best_baseline"] = adjusted.loc[mask, "support_precision"] - best_precision
        adjusted.loc[mask, "supported_per_100_delta_vs_best_baseline"] = adjusted.loc[mask, "supported_per_100"] - best_supported
        rows.append(adjusted)
    return pd.concat(rows, ignore_index=True)


def benchmark_scet_candidates(
    candidates: pd.DataFrame,
    dataset: str,
    neural_level: str = "none",
    baseline_retrievers: list[str] | None = None,
    calibration_query_fraction: float = 0.3,
    calibration_seed: int = 13,
    top_k: int = 5,
    semantic_batch_size: int = 64,
    semantic_cache_key: str | None = None,
    candidates_per_query: int = 100,
    fast_text_scores: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], SCETModel]:
    del candidates_per_query
    scored = prepare_scet_scored_candidates(
        candidates,
        neural_level=neural_level,
        semantic_batch_size=semantic_batch_size,
        semantic_cache_key=semantic_cache_key or f"scet_{dataset}_{neural_level}_{calibration_seed}",
        fast_text_scores=fast_text_scores,
    )
    calibration, evaluation = split_calibration_evaluation_queries(
        scored,
        calibration_fraction=calibration_query_fraction,
        seed=calibration_seed,
    )
    scet_detail, model = fit_and_apply_scet(calibration, evaluation, top_k=top_k)
    scet_detail["evaluation_partition"] = "evaluation"
    baseline_retrievers = baseline_retrievers or ["bm25", "tfidf_svd", "dense", "mvec_tg"]
    if neural_level in {"light", "full"}:
        baseline_retrievers = [*baseline_retrievers, "sbert_mini_lm", "e5_small_v2", "mvec_tg_semantic"]
    if neural_level == "full":
        baseline_retrievers = [*baseline_retrievers, "bge_m3", "cross_encoder"]
    baseline_detail = apply_baseline_retrievers(evaluation, calibration, baseline_retrievers, top_k=top_k)
    if not baseline_detail.empty:
        baseline_detail["scet_conflict_score"] = compute_conflict_scores(baseline_detail, available_scet_views(scet_detail, model.view_columns))
    detail = pd.concat([baseline_detail, scet_detail], ignore_index=True) if not baseline_detail.empty else scet_detail
    ablation = build_scet_ablation_summary(calibration, evaluation, model)
    stress = build_conflict_stress_summary(detail)
    certificates = support_certificates(detail, model)
    return detail, ablation, stress, certificates, model


def _empty_outputs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(out / "scet_detail.csv", index=False)
    pd.DataFrame().to_csv(out / "scet_summary.csv", index=False)
    write_jsonl(out / "scet_support_certificates.jsonl", [])
    pd.DataFrame().to_csv(out / "scet_ablation_summary.csv", index=False)
    pd.DataFrame().to_csv(out / "scet_stress_summary.csv", index=False)


def run_scet_benchmark(
    mode: str,
    datasets: list[str],
    out: Path,
    neural_level: str = "none",
    candidates_per_query: int = 100,
    calibration_query_fraction: float = 0.3,
    calibration_seed: int = 13,
    top_k: int = 5,
    semantic_batch_size: int = 64,
    fast_text_scores: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    detail_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    ablation_parts: list[pd.DataFrame] = []
    stress_parts: list[pd.DataFrame] = []
    certificate_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset == "bts":
            bts = load_bts_summary()
            if not bts.empty:
                bts = bts.copy()
                bts["retriever"] = bts["retriever"].replace({"mvec_tg": "mvec_tg_structured_reference"})
                summary_parts.append(bts)
            continue
        candidates = load_or_build_candidates(dataset, mode, candidates_per_query)
        detail, ablation, stress, certificates, model = benchmark_scet_candidates(
            candidates,
            dataset=dataset,
            neural_level=neural_level,
            calibration_query_fraction=calibration_query_fraction,
            calibration_seed=calibration_seed,
            top_k=top_k,
            semantic_batch_size=semantic_batch_size,
            semantic_cache_key=f"{dataset}_{mode}_{candidates_per_query}",
            candidates_per_query=candidates_per_query,
            fast_text_scores=fast_text_scores,
        )
        detail_parts.append(detail)
        summary_parts.append(summarize_scet_detail(detail))
        ablation_parts.append(ablation)
        stress_parts.append(stress)
        certificate_rows.extend(certificates)
        model_rows.append(
            {
                "dataset": dataset,
                "model": json.dumps(asdict(model), ensure_ascii=False, sort_keys=True),
                "model_digest": hashlib.sha1(json.dumps(asdict(model), sort_keys=True).encode("utf-8")).hexdigest()[:12],
            }
        )
    detail_out = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    summary_out = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    ablation_out = pd.concat(ablation_parts, ignore_index=True) if ablation_parts else pd.DataFrame()
    stress_out = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
    detail_out.to_csv(out / "scet_detail.csv", index=False)
    summary_out.to_csv(out / "scet_summary.csv", index=False)
    ablation_out.to_csv(out / "scet_ablation_summary.csv", index=False)
    stress_out.to_csv(out / "scet_stress_summary.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out / "scet_models.csv", index=False)
    write_jsonl(out / "scet_support_certificates.jsonl", certificate_rows)
    return detail_out, summary_out, stress_out


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--datasets", default="bts,scifact,fever")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--neural-level", choices=["none", "light", "full"], default="none")
    parser.add_argument("--candidates-per-query", type=int, default=100)
    parser.add_argument("--calibration-query-fraction", type=float, default=0.3)
    parser.add_argument("--calibration-seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--fast-text-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (OUT_SMOKE if args.mode == "smoke" else OUT_FULL)
    _, summary, stress = run_scet_benchmark(
        mode=args.mode,
        datasets=parse_list(args.datasets),
        out=out,
        neural_level=args.neural_level,
        candidates_per_query=args.candidates_per_query,
        calibration_query_fraction=args.calibration_query_fraction,
        calibration_seed=args.calibration_seed,
        top_k=args.top_k,
        semantic_batch_size=args.semantic_batch_size,
        fast_text_scores=args.fast_text_scores,
    )
    print(summary.to_string(index=False) if not summary.empty else "(no summary rows)")
    if not stress.empty:
        print(stress.to_string(index=False))


if __name__ == "__main__":
    main()
