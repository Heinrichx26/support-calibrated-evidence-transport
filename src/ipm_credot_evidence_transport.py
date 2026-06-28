from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ipm_cross_domain_evidence_benchmark import (
    evidence_retrieval_metrics,
    load_or_build_candidates,
    minmax,
    split_calibration_evaluation_queries,
    summarize_text_detail,
)
from ipm_scet_evidence_transport import (
    BASE_VIEW_COLUMNS,
    SCETModel,
    apply_baseline_retrievers,
    apply_scet_model,
    apply_support_portfolio_transport,
    available_scet_views,
    build_conflict_stress_summary,
    compute_conflict_scores,
    fit_scet_model,
    fit_support_portfolio_transport,
    learn_view_reliability,
    prepare_scet_scored_candidates,
    select_support_threshold,
    write_jsonl,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT_SMOKE = PROJECT / "results" / "ipm_credot_smoke"
OUT_FULL = PROJECT / "results" / "ipm_credot_full"

CREDOT_OUTPUT_FILES = [
    "credot_detail.csv",
    "credot_summary.csv",
    "credot_support_certificates.jsonl",
    "credot_ablation_summary.csv",
    "credot_stress_summary.csv",
]

DIMENSIONS = [
    "semantic",
    "lexical",
    "entity",
    "relation",
    "source_margin",
    "coverage",
    "conflict_complement",
]

SEMANTIC_COLUMNS = {
    "dense_score",
    "sbert_mini_lm_score",
    "e5_small_v2_score",
    "bge_m3_score",
    "cross_encoder_score",
}

LEXICAL_COLUMNS = {"bm25_score", "tfidf_score", "dense_score"}
ENTITY_COLUMNS = {"entity_score"}
RELATION_COLUMNS = {"relation_score"}
SOURCE_COLUMNS = {"source_score", "bts_view_margin_score"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "over",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "under",
    "was",
    "were",
    "which",
    "with",
}


@dataclass(frozen=True)
class CREDOTTransportConfig:
    epsilon: float = 0.08
    tau_mu: float = 0.55
    tau_nu: float = 0.55
    conflict_weight: float = 0.48
    robust_radius: float = 0.16
    mass_reward: float = 0.02
    base_support_weight: float = 1.0
    transport_cost_weight: float = 0.04
    obligation_gap_weight: float = 0.02
    conflict_support_penalty: float = 0.01
    max_iter: int = 80


@dataclass(frozen=True)
class CREDOTModel:
    scet_model: SCETModel
    view_columns: list[str]
    support_threshold: float
    top_k: int
    calibration_query_ids: list[str]
    calibration_metrics: dict[str, float]
    threshold_grid: list[dict[str, float]]
    dimension_reliability: dict[str, float]
    reliability_lower: dict[str, float]
    reliability_upper: dict[str, float]
    config: CREDOTTransportConfig
    transport_mode: str = "credal_transport"
    dual_feature_weights: dict[str, float] = field(default_factory=dict)
    dual_feature_intercept: float = 0.0
    dual_feature_means: dict[str, float] = field(default_factory=dict)
    dual_feature_scales: dict[str, float] = field(default_factory=dict)


DUAL_FEATURES = [
    "scet_support_score",
    "credot_evidence_mass",
    "credot_obligation_gap",
    "credot_robust_transport_cost",
    "credot_conflict_score",
    "scet_mass_product",
    "scet_conflict_residual",
    "mass_gap_balance",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _stem(token: str) -> str:
    for suffix in ("ization", "ations", "ation", "ingly", "edly", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _terms(text: str) -> set[str]:
    terms = set()
    for token in _tokenize(text):
        stemmed = _stem(token)
        if len(stemmed) >= 3 and stemmed not in STOPWORDS:
            terms.add(stemmed)
    return terms


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _query_rank(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return frame.groupby("query_id", sort=False)[score_col].rank(method="first", ascending=False).astype(int)


def _column_max(frame: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> pd.Series:
    usable = [col for col in columns if col in frame.columns]
    if not usable:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.concat([_safe_numeric(frame, col) for col in usable], axis=1).max(axis=1).astype(float)


def _column_mean(frame: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> pd.Series:
    usable = [col for col in columns if col in frame.columns]
    if not usable:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.concat([_safe_numeric(frame, col) for col in usable], axis=1).mean(axis=1).astype(float)


def _coverage_scores(frame: pd.DataFrame) -> pd.Series:
    values = []
    for row in frame.itertuples(index=False):
        query_terms = _terms(str(getattr(row, "query_text", "")))
        evidence_terms = _terms(str(getattr(row, "evidence_text", "")))
        if not query_terms:
            values.append(0.0)
        else:
            values.append(len(query_terms & evidence_terms) / len(query_terms))
    return pd.Series(values, index=frame.index, dtype=float).clip(0.0, 1.0)


def _dimension_reliability(view_reliability: dict[str, float], view_columns: list[str]) -> dict[str, float]:
    mean_reliability = float(np.mean(list(view_reliability.values()))) if view_reliability else 1.0 / max(len(DIMENSIONS), 1)

    def family_reliability(columns: set[str], fallback: float = mean_reliability) -> float:
        usable = [col for col in view_columns if col in columns and col in view_reliability]
        if not usable:
            return float(fallback)
        return float(np.mean([view_reliability[col] for col in usable]))

    semantic = family_reliability(SEMANTIC_COLUMNS)
    lexical = family_reliability(LEXICAL_COLUMNS)
    entity = family_reliability(ENTITY_COLUMNS, fallback=mean_reliability)
    relation = family_reliability(RELATION_COLUMNS, fallback=mean_reliability)
    source = family_reliability(SOURCE_COLUMNS, fallback=0.5 * (entity + relation))
    coverage = float(np.mean([semantic, lexical, entity, relation]))
    conflict = float(np.mean([semantic, entity, relation]))
    raw = {
        "semantic": semantic,
        "lexical": lexical,
        "entity": entity,
        "relation": relation,
        "source_margin": source,
        "coverage": coverage,
        "conflict_complement": conflict,
    }
    total = sum(max(value, 1e-6) for value in raw.values())
    return {key: max(value, 1e-6) / max(total, 1e-12) for key, value in raw.items()}


def _credal_intervals(reliability: dict[str, float], radius: float) -> tuple[dict[str, float], dict[str, float]]:
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for dim, value in reliability.items():
        value = float(np.clip(value, 0.0, 1.0))
        width = float(radius) * (0.5 + 0.5 * (1.0 - value))
        lower[dim] = float(np.clip(value - width, 0.01, 1.0))
        upper[dim] = float(np.clip(value + width, 0.01, 1.0))
    return lower, upper


def _obligation_distribution(model: CREDOTModel) -> np.ndarray:
    base = {
        "semantic": 0.24,
        "lexical": 0.13,
        "entity": 0.17,
        "relation": 0.17,
        "source_margin": 0.08,
        "coverage": 0.11,
        "conflict_complement": 0.10,
    }
    values = []
    for dim in DIMENSIONS:
        reliability = float(model.dimension_reliability.get(dim, 1.0 / len(DIMENSIONS)))
        values.append(float(base[dim]) * (0.75 + reliability))
    arr = np.asarray(values, dtype=float)
    return arr / max(float(arr.sum()), 1e-12)


def _evidence_supply(frame: pd.DataFrame, view_columns: list[str]) -> pd.DataFrame:
    del view_columns
    conflict = compute_conflict_scores(frame, available_scet_views(frame, BASE_VIEW_COLUMNS))
    semantic = _column_max(frame, SEMANTIC_COLUMNS, default=_column_mean(frame, BASE_VIEW_COLUMNS, default=0.0).mean())
    lexical = _column_max(frame, LEXICAL_COLUMNS, default=0.0)
    entity = _safe_numeric(frame, "entity_score", default=0.0)
    relation = _safe_numeric(frame, "relation_score", default=0.0)
    source = _column_max(frame, SOURCE_COLUMNS, default=float(np.nan))
    anchor_mean = pd.concat([entity, relation], axis=1).mean(axis=1)
    source = source.where(source.notna(), 0.30 + 0.40 * anchor_mean)
    coverage = _coverage_scores(frame)
    supply = pd.DataFrame(
        {
            "semantic": semantic,
            "lexical": lexical,
            "entity": entity,
            "relation": relation,
            "source_margin": source,
            "coverage": coverage,
            "conflict_complement": 1.0 - conflict,
        },
        index=frame.index,
    )
    return supply.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 1.0)


def _base_cost_matrix(reliability: dict[str, float], conflict: float, config: CREDOTTransportConfig) -> np.ndarray:
    n = len(DIMENSIONS)
    matrix = np.full((n, n), 0.58, dtype=float)
    for idx, dim in enumerate(DIMENSIONS):
        rel = float(reliability.get(dim, 1.0 / n))
        matrix[idx, idx] = 0.04 + 0.18 * (1.0 - rel)
    compatible = {
        ("semantic", "lexical"): 0.20,
        ("lexical", "semantic"): 0.20,
        ("semantic", "coverage"): 0.28,
        ("coverage", "semantic"): 0.28,
        ("lexical", "coverage"): 0.24,
        ("coverage", "lexical"): 0.24,
        ("entity", "relation"): 0.22,
        ("relation", "entity"): 0.22,
        ("source_margin", "entity"): 0.35,
        ("source_margin", "relation"): 0.35,
        ("entity", "source_margin"): 0.35,
        ("relation", "source_margin"): 0.35,
    }
    index = {dim: idx for idx, dim in enumerate(DIMENSIONS)}
    for (left, right), value in compatible.items():
        matrix[index[left], index[right]] = min(matrix[index[left], index[right]], value)

    liability = np.ones((n, n), dtype=float) * 0.35
    conflict_idx = index["conflict_complement"]
    liability[:, conflict_idx] = 1.15
    liability[conflict_idx, :] = 0.85
    liability[conflict_idx, conflict_idx] = 1.30
    return matrix + float(config.conflict_weight) * float(conflict) * liability


def _generalized_kl(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = np.clip(left, 1e-12, None)
    right = np.clip(right, 1e-12, None)
    return float(np.sum(left * np.log(left / right) - left + right))


def _unbalanced_transport(
    mu: np.ndarray,
    nu: np.ndarray,
    cost: np.ndarray,
    config: CREDOTTransportConfig,
) -> tuple[float, np.ndarray]:
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    nu = np.clip(np.asarray(nu, dtype=float), 1e-9, None)
    eps = max(float(config.epsilon), 1e-4)
    tau_mu = max(float(config.tau_mu), 1e-4)
    tau_nu = max(float(config.tau_nu), 1e-4)
    kernel = np.exp(-np.asarray(cost, dtype=float) / eps)
    kernel = np.clip(kernel, 1e-300, None)
    u = np.ones_like(mu)
    v = np.ones_like(nu)
    phi_mu = tau_mu / (tau_mu + eps)
    phi_nu = tau_nu / (tau_nu + eps)
    for _ in range(max(int(config.max_iter), 1)):
        u = (mu / np.clip(kernel @ v, 1e-300, None)) ** phi_mu
        v = (nu / np.clip(kernel.T @ u, 1e-300, None)) ** phi_nu
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            u = np.ones_like(mu)
            v = np.ones_like(nu)
            break
    plan = (u[:, None] * kernel) * v[None, :]
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    row_mass = plan.sum(axis=1)
    col_mass = plan.sum(axis=0)
    objective = float(np.sum(plan * cost))
    objective += tau_mu * _generalized_kl(row_mass, mu)
    objective += tau_nu * _generalized_kl(col_mass, nu)
    prior = np.outer(mu, nu / max(float(nu.sum()), 1e-12))
    objective += eps * _generalized_kl(plan, np.clip(prior, 1e-12, None))
    return float(objective), plan


def _dual_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    scet = _safe_numeric(frame, "scet_support_score")
    mass = _safe_numeric(frame, "credot_evidence_mass")
    gap = _safe_numeric(frame, "credot_obligation_gap")
    cost = _safe_numeric(frame, "credot_robust_transport_cost")
    conflict = _safe_numeric(frame, "credot_conflict_score")
    features["scet_support_score"] = scet
    features["credot_evidence_mass"] = mass
    features["credot_obligation_gap"] = gap
    features["credot_robust_transport_cost"] = cost
    features["credot_conflict_score"] = conflict
    features["scet_mass_product"] = scet * mass
    features["scet_conflict_residual"] = scet * (1.0 - conflict)
    features["mass_gap_balance"] = mass - gap
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _standardize_dual_features(features: pd.DataFrame, means: dict[str, float], scales: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    for col in DUAL_FEATURES:
        values = pd.to_numeric(features[col], errors="coerce").fillna(0.0) if col in features else pd.Series(0.0, index=features.index)
        out[col] = (values.astype(float) - float(means.get(col, 0.0))) / max(float(scales.get(col, 1.0)), 1e-9)
    return out


def _with_fitted_dual_potential(model: CREDOTModel, scored_calibration: pd.DataFrame, mode: str) -> CREDOTModel:
    y = pd.to_numeric(scored_calibration["gold_support"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    features = _dual_features(scored_calibration)
    means = {col: float(features[col].mean()) for col in DUAL_FEATURES}
    scales = {col: float(features[col].std(ddof=0) or 1.0) for col in DUAL_FEATURES}
    if y.nunique() < 2:
        return replace(
            model,
            transport_mode=mode,
            dual_feature_weights={col: 0.0 for col in DUAL_FEATURES},
            dual_feature_intercept=0.0,
            dual_feature_means=means,
            dual_feature_scales=scales,
        )
    standardized = _standardize_dual_features(features, means, scales)
    try:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=29, solver="liblinear")
        classifier.fit(standardized.to_numpy(float), y.to_numpy(int))
        weights = {col: float(value) for col, value in zip(DUAL_FEATURES, classifier.coef_[0])}
        intercept = float(classifier.intercept_[0])
    except Exception:
        weights = {col: 0.0 for col in DUAL_FEATURES}
        weights["scet_support_score"] = 1.0
        intercept = 0.0
    return replace(
        model,
        transport_mode=mode,
        dual_feature_weights=weights,
        dual_feature_intercept=intercept,
        dual_feature_means=means,
        dual_feature_scales=scales,
    )


def _apply_dual_potential(out: pd.DataFrame, model: CREDOTModel) -> pd.DataFrame:
    if not model.dual_feature_weights:
        out["credot_support_score"] = out.groupby("query_id", sort=False)["credot_raw_score"].transform(lambda values: minmax(values.to_numpy()))
        return out
    features = _dual_features(out)
    standardized = _standardize_dual_features(features, model.dual_feature_means, model.dual_feature_scales)
    logits = pd.Series(float(model.dual_feature_intercept), index=out.index, dtype=float)
    for col in DUAL_FEATURES:
        logits = logits + standardized[col] * float(model.dual_feature_weights.get(col, 0.0))
    logits = logits.clip(-30.0, 30.0)
    out["credot_dual_logit"] = logits
    out["credot_support_score"] = 1.0 / (1.0 + np.exp(-logits))
    return out


def fit_scet_portfolio_prior(calibration: pd.DataFrame, view_columns: list[str] | None = None, top_k: int = 5) -> SCETModel:
    usable_views = available_scet_views(calibration, view_columns or BASE_VIEW_COLUMNS)
    reliability = learn_view_reliability(calibration, usable_views, top_k=top_k)
    portfolio = fit_support_portfolio_transport(calibration, usable_views)
    scored = apply_support_portfolio_transport(calibration, portfolio)
    threshold, metrics, grid = select_support_threshold_fast(
        scored,
        score_col="scet_support_score",
        top_k=top_k,
        precision_weight=2.0,
        min_supported_per_100=25.0,
    )
    return SCETModel(
        view_columns=usable_views,
        view_reliability=reliability,
        support_threshold=float(threshold),
        penalty_weight=0.0,
        top_k=int(top_k),
        selection_mode="support_portfolio_transport",
        calibration_query_ids=sorted(calibration["query_id"].astype(str).unique().tolist()),
        calibration_metrics={key: float(value) for key, value in metrics.items()},
        threshold_grid=grid,
        portfolio_transport=portfolio,
    )


def select_support_threshold_fast(
    calibration: pd.DataFrame,
    score_col: str = "credot_support_score",
    top_k: int = 5,
    precision_weight: float = 2.0,
    min_supported_per_100: float = 25.0,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    work = calibration.copy()
    scores = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0).astype(float)
    work["retrieval_score"] = scores
    work["rank"] = work.groupby("query_id", sort=False)["retrieval_score"].rank(method="first", ascending=False).astype(int)
    top_mask = work["rank"].le(int(top_k)).to_numpy(bool)
    gold = pd.to_numeric(work["gold_support"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
    base = work.copy()
    base["evidence_supported"] = top_mask.astype(int)
    base_metrics = evidence_retrieval_metrics(base, top_k=top_k)
    queries = int(base_metrics["queries"])
    total_gold = int(gold.sum())
    fixed_thresholds = {0.0, 0.25, 0.5, 0.75, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0}
    grid = sorted({float(np.quantile(scores, q)) for q in np.linspace(0.0, 1.0, 31)} | fixed_thresholds)
    rows: list[dict[str, float]] = []
    best_threshold = 0.0
    best_metrics: dict[str, float] = {}
    best_key: tuple[int, float, float, float, float] | None = None
    score_values = scores.to_numpy(float)
    recall_at_5 = float(base_metrics.get("recall_at_5", 0.0))
    for threshold in grid:
        selected = top_mask & (score_values >= float(threshold))
        supported_positive = int((selected & (gold == 1)).sum())
        supported_negative = int((selected & (gold == 0)).sum())
        support_precision = supported_positive / max(supported_positive + supported_negative, 1)
        supported_per_100 = 100.0 * supported_positive / max(queries, 1)
        evidence_hit = supported_positive / max(total_gold, 1)
        objective = (
            -1.0
            if supported_positive <= 0
            else evidence_hit + float(precision_weight) * support_precision + 0.01 * supported_per_100
        )
        row = {
            "threshold": float(threshold),
            "objective": float(objective),
            "support_precision": float(support_precision),
            "evidence_hit": float(evidence_hit),
            "recall_at_5": recall_at_5,
            "mrr": float(base_metrics.get("mrr", 0.0)),
            "ndcg_at_10": float(base_metrics.get("ndcg_at_10", 0.0)),
            "supported_per_100": float(supported_per_100),
            "supported_positive": float(supported_positive),
        }
        rows.append(row)
        feasible = int(supported_positive > 0 and supported_per_100 >= float(min_supported_per_100))
        key = (feasible, support_precision, recall_at_5, supported_per_100, objective)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {k: float(v) for k, v in row.items() if k != "threshold"}
    return best_threshold, best_metrics, rows


def _score_credot_compact(frame: pd.DataFrame, model: CREDOTModel) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    scet_detail = apply_scet_model(frame, model.scet_model, retriever_name="scet")
    out = scet_detail.copy()
    conflict = compute_conflict_scores(out, available_scet_views(out, model.view_columns))
    supply_frame = _evidence_supply(out, model.view_columns)
    supply = supply_frame.loc[:, DIMENSIONS].to_numpy(float)
    mu = _obligation_distribution(model)
    nu = np.clip(supply * mu.reshape(1, -1), 1e-9, None)
    evidence_mass = nu.sum(axis=1)
    obligation_gap = (mu.reshape(1, -1) * (1.0 - supply)).sum(axis=1)
    center_rel = np.asarray([model.dimension_reliability.get(dim, 1.0 / len(DIMENSIONS)) for dim in DIMENSIONS], dtype=float)
    adverse_rel = np.asarray([model.reliability_lower.get(dim, center_rel[pos]) for pos, dim in enumerate(DIMENSIONS)], dtype=float)
    center_diag = 0.04 + 0.18 * (1.0 - center_rel)
    adverse_diag = 0.04 + 0.18 * (1.0 - adverse_rel)
    deficit = np.clip(mu.reshape(1, -1) - nu, 0.0, None)
    center_cost = (deficit * center_diag.reshape(1, -1)).sum(axis=1) / np.clip(evidence_mass, 1e-9, None)
    adverse_cost = (deficit * adverse_diag.reshape(1, -1)).sum(axis=1) / np.clip(evidence_mass, 1e-9, None)
    conflict_supply = supply_frame["conflict_complement"].to_numpy(float)
    conflict_liability = float(model.config.conflict_weight) * conflict.to_numpy(float) * (1.0 - conflict_supply)
    center_cost = center_cost + conflict_liability
    adverse_cost = adverse_cost + conflict_liability
    robust_cost = np.maximum(center_cost, adverse_cost)
    scet_score = _safe_numeric(out, "scet_support_score").to_numpy(float)
    raw = (
        float(model.config.base_support_weight) * scet_score
        + float(model.config.mass_reward) * evidence_mass
        - float(model.config.transport_cost_weight) * robust_cost
        - float(model.config.obligation_gap_weight) * obligation_gap
        - float(model.config.conflict_support_penalty) * conflict.to_numpy(float)
    )
    out["credot_transport_cost"] = center_cost
    out["credot_robust_transport_cost"] = robust_cost
    out["credot_evidence_mass"] = evidence_mass
    out["credot_obligation_gap"] = obligation_gap
    out["credot_conflict_score"] = conflict
    out["scet_conflict_score"] = conflict
    out["credot_raw_score"] = raw
    return _apply_dual_potential(out, model)


def _score_credot_frame(frame: pd.DataFrame, model: CREDOTModel, include_transport_artifacts: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if not include_transport_artifacts:
        return _score_credot_compact(frame, model)
    scet_detail = apply_scet_model(frame, model.scet_model, retriever_name="scet")
    out = scet_detail.copy()
    conflict = compute_conflict_scores(out, available_scet_views(out, model.view_columns))
    supply_frame = _evidence_supply(out, model.view_columns)
    mu = _obligation_distribution(model)
    center_reliability = model.dimension_reliability
    adverse_reliability = model.reliability_lower if model.config.robust_radius > 0 else model.dimension_reliability

    transport_costs: list[float] = []
    robust_costs: list[float] = []
    evidence_masses: list[float] = []
    obligation_gaps: list[float] = []
    plans: list[dict[str, dict[str, float]]] = []
    obligation_json: list[str] = []
    supply_json: list[str] = []
    raw_scores: list[float] = []
    for idx, row in out.iterrows():
        supply_values = supply_frame.loc[idx, DIMENSIONS].to_numpy(float)
        nu = np.clip(mu * supply_values, 1e-9, None)
        center_cost_matrix = _base_cost_matrix(center_reliability, float(conflict.loc[idx]), model.config)
        adverse_cost_matrix = _base_cost_matrix(adverse_reliability, float(conflict.loc[idx]), model.config)
        center_objective, center_plan = _unbalanced_transport(mu, nu, center_cost_matrix, model.config)
        adverse_objective, adverse_plan = _unbalanced_transport(mu, nu, adverse_cost_matrix, model.config)
        evidence_mass = float(np.sum(nu))
        center_cost = center_objective / max(evidence_mass, 1e-9)
        adverse_cost = adverse_objective / max(evidence_mass, 1e-9)
        robust_cost = max(center_cost, adverse_cost)
        plan = adverse_plan if adverse_cost >= center_cost else center_plan
        gap = float(np.sum(mu * (1.0 - supply_values)))
        scet_score = float(row.get("scet_support_score", 0.0))
        raw = (
            float(model.config.base_support_weight) * scet_score
            + float(model.config.mass_reward) * evidence_mass
            - float(model.config.transport_cost_weight) * robust_cost
            - float(model.config.obligation_gap_weight) * gap
            - float(model.config.conflict_support_penalty) * float(conflict.loc[idx])
        )
        transport_costs.append(float(center_cost))
        robust_costs.append(float(robust_cost))
        evidence_masses.append(evidence_mass)
        obligation_gaps.append(gap)
        if include_transport_artifacts:
            plans.append(
                {
                    DIMENSIONS[r]: {DIMENSIONS[c]: float(plan[r, c]) for c in range(len(DIMENSIONS)) if plan[r, c] > 1e-6}
                    for r in range(len(DIMENSIONS))
                }
            )
            obligation_json.append(json.dumps({dim: float(mu[pos]) for pos, dim in enumerate(DIMENSIONS)}, sort_keys=True))
            supply_json.append(json.dumps({dim: float(supply_values[pos]) for pos, dim in enumerate(DIMENSIONS)}, sort_keys=True))
        raw_scores.append(float(raw))

    out["credot_transport_cost"] = transport_costs
    out["credot_robust_transport_cost"] = robust_costs
    out["credot_evidence_mass"] = evidence_masses
    out["credot_obligation_gap"] = obligation_gaps
    out["credot_conflict_score"] = conflict
    out["scet_conflict_score"] = conflict
    if include_transport_artifacts:
        out["credot_obligation_json"] = obligation_json
        out["credot_supply_json"] = supply_json
        out["credot_transport_plan_json"] = [json.dumps(plan, sort_keys=True) for plan in plans]
    out["credot_reliability_interval_json"] = json.dumps(
            {
                dim: [float(model.reliability_lower.get(dim, 0.0)), float(model.reliability_upper.get(dim, 0.0))]
                for dim in DIMENSIONS
            },
            sort_keys=True,
    )
    out["credot_raw_score"] = raw_scores
    return _apply_dual_potential(out, model)


def _metrics_for_threshold(frame: pd.DataFrame, threshold: float, top_k: int) -> dict[str, float]:
    work = frame.copy()
    work["retrieval_score"] = _safe_numeric(work, "credot_support_score")
    work["rank"] = _query_rank(work, "retrieval_score")
    work["top_k"] = int(top_k)
    work["evidence_supported"] = ((work["rank"] <= int(top_k)) & (work["credot_support_score"] >= float(threshold))).astype(int)
    return {key: float(value) for key, value in evidence_retrieval_metrics(work, top_k=int(top_k)).items()}


def fit_credot_model(
    calibration: pd.DataFrame,
    view_columns: list[str] | None = None,
    top_k: int = 5,
    config: CREDOTTransportConfig | None = None,
    fast_prior: bool = False,
) -> CREDOTModel:
    config = config or CREDOTTransportConfig()
    usable_views = available_scet_views(calibration, view_columns or BASE_VIEW_COLUMNS)
    if fast_prior:
        scet_model = fit_scet_portfolio_prior(calibration, view_columns=usable_views, top_k=top_k)
    else:
        scet_model = fit_scet_model(calibration, view_columns=usable_views, top_k=top_k)
    reliability = _dimension_reliability(scet_model.view_reliability, scet_model.view_columns)
    query_ids = sorted(calibration["query_id"].astype(str).unique().tolist())
    candidate_configs = [
        ("credal_transport", config),
        (
            "transport_without_conflict",
            replace(config, conflict_weight=0.0, conflict_support_penalty=0.0),
        ),
        (
            "evidence_potential",
            replace(
                config,
                mass_reward=0.0,
                transport_cost_weight=0.0,
                obligation_gap_weight=0.0,
                conflict_support_penalty=0.0,
            ),
        ),
        (
            "mass_certified_potential",
            replace(
                config,
                mass_reward=0.01,
                transport_cost_weight=0.0,
                obligation_gap_weight=0.0,
                conflict_support_penalty=0.0,
            ),
        ),
    ]
    best_key: tuple[int, float, float, float, float, int] | None = None
    best_model: CREDOTModel | None = None
    best_recall = -1.0
    evaluated: list[tuple[int, CREDOTModel, dict[str, float]]] = []
    offset = 0
    for mode, candidate_config in candidate_configs:
        lower, upper = _credal_intervals(reliability, candidate_config.robust_radius)
        provisional = CREDOTModel(
            scet_model=scet_model,
            view_columns=scet_model.view_columns,
            support_threshold=0.0,
            top_k=int(top_k),
            calibration_query_ids=query_ids,
            calibration_metrics={},
            threshold_grid=[],
            dimension_reliability=reliability,
            reliability_lower=lower,
            reliability_upper=upper,
            config=candidate_config,
            transport_mode=mode,
        )
        scored_calibration = _score_credot_frame(calibration, provisional, include_transport_artifacts=False)
        candidate_models = [(mode, provisional, scored_calibration)]
        dual_model = _with_fitted_dual_potential(provisional, scored_calibration, f"{mode}_dual")
        dual_scored = _score_credot_frame(calibration, dual_model, include_transport_artifacts=False)
        candidate_models.append((f"{mode}_dual", dual_model, dual_scored))
        for _, candidate_model, scored in candidate_models:
            threshold, metrics, grid = select_support_threshold_fast(
                scored,
                score_col="credot_support_score",
                top_k=top_k,
                precision_weight=2.2,
                min_supported_per_100=25.0,
            )
            calibrated = replace(
                candidate_model,
                support_threshold=float(threshold),
                calibration_metrics={key: float(value) for key, value in metrics.items()},
                threshold_grid=grid,
            )
            evaluated.append((offset, calibrated, {key: float(value) for key, value in metrics.items()}))
            best_recall = max(best_recall, float(metrics.get("recall_at_5", 0.0)))
            offset += 1
    for offset, candidate_model, metrics in evaluated:
        recall = float(metrics.get("recall_at_5", 0.0))
        support_precision = float(metrics.get("support_precision", 0.0))
        evidence_hit = float(metrics.get("evidence_hit", 0.0))
        supported_per_100 = float(metrics.get("supported_per_100", 0.0))
        recall_feasible = int(recall >= best_recall - 0.01)
        key = (recall_feasible, support_precision, recall, evidence_hit, supported_per_100, -offset)
        if best_key is None or key > best_key:
            best_key = key
            best_model = candidate_model
    if best_model is None:
        raise RuntimeError("CREDOT could not select a calibrated transport mode")
    return best_model


def apply_credot_model(
    frame: pd.DataFrame,
    model: CREDOTModel,
    retriever_name: str = "credot",
    include_transport_artifacts: bool = True,
) -> pd.DataFrame:
    out = _score_credot_frame(frame, model, include_transport_artifacts=include_transport_artifacts)
    out["retrieval_score"] = out["credot_support_score"]
    out["rank"] = _query_rank(out, "retrieval_score")
    out["top_k"] = int(model.top_k)
    out["evidence_supported"] = ((out["rank"] <= int(model.top_k)) & (out["credot_support_score"] >= model.support_threshold)).astype(int)
    out["retriever"] = retriever_name
    out["credot_support_threshold"] = float(model.support_threshold)
    out["credot_selection_mode"] = model.transport_mode
    out["credot_dimension_reliability_json"] = json.dumps(model.dimension_reliability, sort_keys=True)
    return out


def fit_and_apply_credot(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    view_columns: list[str] | None = None,
    top_k: int = 5,
    config: CREDOTTransportConfig | None = None,
    fast_prior: bool = False,
) -> tuple[pd.DataFrame, CREDOTModel]:
    model = fit_credot_model(calibration, view_columns=view_columns, top_k=top_k, config=config, fast_prior=fast_prior)
    return apply_credot_model(evaluation, model), model


def _calibrated_variant(model: CREDOTModel, calibration: pd.DataFrame, config: CREDOTTransportConfig) -> CREDOTModel:
    reliability = model.dimension_reliability
    lower, upper = _credal_intervals(reliability, config.robust_radius)
    provisional = replace(model, config=config, reliability_lower=lower, reliability_upper=upper, support_threshold=0.0)
    scored = _score_credot_frame(calibration, provisional, include_transport_artifacts=False)
    threshold, metrics, grid = select_support_threshold_fast(
        scored,
        score_col="credot_support_score",
        top_k=model.top_k,
        precision_weight=2.2,
        min_supported_per_100=25.0,
    )
    return replace(provisional, support_threshold=float(threshold), calibration_metrics=metrics, threshold_grid=grid)


def build_credot_ablation_summary(calibration: pd.DataFrame, evaluation: pd.DataFrame, model: CREDOTModel) -> pd.DataFrame:
    variants = [
        ("credot", model),
        ("credot_no_robustness", _calibrated_variant(model, calibration, replace(model.config, robust_radius=0.0))),
        (
            "credot_no_conflict",
            _calibrated_variant(
                model,
                calibration,
                replace(model.config, conflict_weight=0.0, conflict_support_penalty=0.0),
            ),
        ),
    ]
    rows = []
    for name, variant in variants:
        detail = apply_credot_model(evaluation, variant, retriever_name=name, include_transport_artifacts=False)
        metrics = evidence_retrieval_metrics(detail, top_k=variant.top_k)
        metrics.update(
            {
                "dataset": str(evaluation["dataset"].iloc[0]) if not evaluation.empty else "",
                "variant": name,
                "support_threshold": float(variant.support_threshold),
                "robust_radius": float(variant.config.robust_radius),
                "conflict_weight": float(variant.config.conflict_weight),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def credot_support_certificates(detail: pd.DataFrame, model: CREDOTModel, per_query: int = 1, max_rows: int = 500) -> list[dict[str, Any]]:
    if detail.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, group in detail[detail["retriever"].eq("credot")].groupby("query_id", sort=False):
        selected = group.sort_values(["evidence_supported", "credot_support_score"], ascending=[False, False]).head(per_query)
        if "credot_transport_plan_json" not in selected.columns:
            artifacts = _score_credot_frame(selected.copy(), model, include_transport_artifacts=True)
            for col in ["credot_transport_plan_json", "credot_obligation_json", "credot_supply_json"]:
                selected[col] = artifacts[col].to_numpy()
        for row in selected.itertuples(index=False):
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
                    "calibrated_support_score": float(getattr(row, "credot_support_score")),
                    "support_threshold": float(model.support_threshold),
                    "robust_transport_cost": float(getattr(row, "credot_robust_transport_cost")),
                    "transport_cost": float(getattr(row, "credot_transport_cost")),
                    "evidence_mass": float(getattr(row, "credot_evidence_mass")),
                    "obligation_gap": float(getattr(row, "credot_obligation_gap")),
                    "conflict_score": float(getattr(row, "credot_conflict_score")),
                    "transport_plan": json.loads(str(getattr(row, "credot_transport_plan_json"))),
                    "obligation_distribution": json.loads(str(getattr(row, "credot_obligation_json"))),
                    "supply_distribution": json.loads(str(getattr(row, "credot_supply_json"))),
                    "credal_reliability_interval": {
                        dim: [float(model.reliability_lower.get(dim, 0.0)), float(model.reliability_upper.get(dim, 0.0))]
                        for dim in DIMENSIONS
                    },
                    "dimension_reliability": model.dimension_reliability,
                }
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def summarize_credot_detail(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    summary = summarize_text_detail(detail)
    rows = []
    for dataset, group in summary.groupby("dataset", sort=False):
        adjusted = group.copy()
        credot = adjusted[adjusted["retriever"].eq("credot")]
        scet = adjusted[adjusted["retriever"].eq("scet")]
        baselines = adjusted[~adjusted["retriever"].isin(["credot", "scet"])]
        if not credot.empty and not scet.empty:
            mask = adjusted["retriever"].eq("credot")
            scet_row = scet.iloc[0]
            adjusted.loc[mask, "recall_at_5_delta_vs_scet"] = adjusted.loc[mask, "recall_at_5"] - float(scet_row["recall_at_5"])
            adjusted.loc[mask, "support_precision_delta_vs_scet"] = adjusted.loc[mask, "support_precision"] - float(scet_row["support_precision"])
            adjusted.loc[mask, "supported_per_100_delta_vs_scet"] = adjusted.loc[mask, "supported_per_100"] - float(scet_row["supported_per_100"])
        if not credot.empty and not baselines.empty:
            mask = adjusted["retriever"].eq("credot")
            adjusted.loc[mask, "recall_at_5_delta_vs_best_baseline"] = adjusted.loc[mask, "recall_at_5"] - float(baselines["recall_at_5"].max())
            adjusted.loc[mask, "support_precision_delta_vs_best_baseline"] = adjusted.loc[mask, "support_precision"] - float(
                baselines["support_precision"].max()
            )
            adjusted.loc[mask, "supported_per_100_delta_vs_best_baseline"] = adjusted.loc[mask, "supported_per_100"] - float(
                baselines["supported_per_100"].max()
            )
        rows.append(adjusted)
    return pd.concat(rows, ignore_index=True)


def benchmark_credot_candidates(
    candidates: pd.DataFrame,
    dataset: str,
    neural_level: str = "none",
    baseline_retrievers: list[str] | None = None,
    calibration_query_fraction: float = 0.3,
    calibration_seed: int = 13,
    top_k: int = 5,
    semantic_batch_size: int = 64,
    semantic_cache_key: str | None = None,
    fast_text_scores: bool = False,
    fast_prior: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], CREDOTModel]:
    scored = prepare_scet_scored_candidates(
        candidates,
        neural_level=neural_level,
        semantic_batch_size=semantic_batch_size,
        semantic_cache_key=semantic_cache_key or f"credot_{dataset}_{neural_level}_{calibration_seed}",
        fast_text_scores=fast_text_scores,
    )
    calibration, evaluation = split_calibration_evaluation_queries(
        scored,
        calibration_fraction=calibration_query_fraction,
        seed=calibration_seed,
    )
    model = fit_credot_model(calibration, top_k=top_k, fast_prior=fast_prior)
    credot_detail = apply_credot_model(evaluation, model, include_transport_artifacts=False)
    credot_detail["evaluation_partition"] = "evaluation"
    scet_detail = apply_scet_model(evaluation, model.scet_model, retriever_name="scet")
    scet_detail["evaluation_partition"] = "evaluation"
    if baseline_retrievers is None:
        baseline_retrievers = ["bm25", "tfidf_svd", "dense", "mvec_tg"]
    if baseline_retrievers and neural_level in {"light", "full"}:
        baseline_retrievers = [*baseline_retrievers, "sbert_mini_lm", "e5_small_v2", "mvec_tg_semantic"]
    if baseline_retrievers and neural_level == "full":
        baseline_retrievers = [*baseline_retrievers, "bge_m3", "cross_encoder"]
    baseline_detail = apply_baseline_retrievers(evaluation, calibration, baseline_retrievers, top_k=top_k) if baseline_retrievers else pd.DataFrame()
    if not baseline_detail.empty:
        baseline_detail["scet_conflict_score"] = compute_conflict_scores(baseline_detail, available_scet_views(baseline_detail, model.view_columns))
    detail = pd.concat([baseline_detail, scet_detail, credot_detail], ignore_index=True) if not baseline_detail.empty else pd.concat(
        [scet_detail, credot_detail], ignore_index=True
    )
    ablation = build_credot_ablation_summary(calibration, evaluation, model)
    stress = build_conflict_stress_summary(detail)
    certificates = credot_support_certificates(detail, model)
    return detail, ablation, stress, certificates, model


def run_credot_benchmark(
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
    method_only: bool = False,
    fast_prior: bool = False,
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
            continue
        candidates = load_or_build_candidates(dataset, mode, candidates_per_query)
        detail, ablation, stress, certificates, model = benchmark_credot_candidates(
            candidates,
            dataset=dataset,
            neural_level=neural_level,
            calibration_query_fraction=calibration_query_fraction,
            calibration_seed=calibration_seed,
            top_k=top_k,
            semantic_batch_size=semantic_batch_size,
            semantic_cache_key=f"{dataset}_{mode}_{candidates_per_query}",
            fast_text_scores=fast_text_scores,
            baseline_retrievers=[] if method_only else None,
            fast_prior=fast_prior,
        )
        detail_parts.append(detail)
        summary_parts.append(summarize_credot_detail(detail))
        ablation_parts.append(ablation)
        stress_parts.append(stress)
        certificate_rows.extend(certificates)
        model_payload = json.dumps(asdict(model), ensure_ascii=False, sort_keys=True, default=str)
        model_rows.append(
            {
                "dataset": dataset,
                "model": model_payload,
                "model_digest": hashlib.sha1(model_payload.encode("utf-8")).hexdigest()[:12],
            }
        )
    detail_out = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    summary_out = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    ablation_out = pd.concat(ablation_parts, ignore_index=True) if ablation_parts else pd.DataFrame()
    stress_out = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
    detail_out.to_csv(out / "credot_detail.csv", index=False)
    summary_out.to_csv(out / "credot_summary.csv", index=False)
    ablation_out.to_csv(out / "credot_ablation_summary.csv", index=False)
    stress_out.to_csv(out / "credot_stress_summary.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out / "credot_models.csv", index=False)
    write_jsonl(out / "credot_support_certificates.jsonl", certificate_rows)
    return detail_out, summary_out, stress_out


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run credal-robust evidence optimal transport.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--datasets", default="scifact,fever,hover")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--neural-level", choices=["none", "light", "full"], default="none")
    parser.add_argument("--candidates-per-query", type=int, default=100)
    parser.add_argument("--calibration-query-fraction", type=float, default=0.3)
    parser.add_argument("--calibration-seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--fast-text-scores", action="store_true")
    parser.add_argument("--method-only", action="store_true")
    parser.add_argument("--fast-prior", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (OUT_SMOKE if args.mode == "smoke" else OUT_FULL)
    _, summary, stress = run_credot_benchmark(
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
        method_only=args.method_only,
        fast_prior=args.fast_prior,
    )
    print(summary.to_string(index=False) if not summary.empty else "(no summary rows)")
    if not stress.empty:
        print(stress.to_string(index=False))


if __name__ == "__main__":
    main()
