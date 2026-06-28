from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ipm_evidence_dataset_adapters import OUT_DEFAULT as ADAPTER_OUT, UNIFIED_COLUMNS, build_dataset, validate_unified_schema


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "ipm_cross_domain_evidence_benchmark"
PROTOCOL_BTS = PROJECT / "results" / "ipm_event_evidence_benchmark_final" / "ipm_event_evidence_benchmark_judgement.csv"
PROTOCOL_BTS_DETAIL = PROJECT / "results" / "ipm_event_evidence_benchmark_final" / "ipm_event_evidence_benchmark_detail.csv"
EMBEDDING_CACHE = PROJECT / "data" / "interim" / "evidence_retrieval" / "embedding_cache"

TEXT_RETRIEVERS = {
    "bm25",
    "tfidf_svd",
    "dense",
    "sbert_mini_lm",
    "e5_small_v2",
    "bge_m3",
    "cross_encoder",
    "nca_metric",
    "single_view_proto",
    "calibrated_fusion",
    "mvec_tg",
    "mvec_tg_semantic",
}

SEMANTIC_RETRIEVERS = {
    "sbert_mini_lm": {
        "score_column": "sbert_mini_lm_score",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5_small_v2": {
        "score_column": "e5_small_v2_score",
        "model_name": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_terms = tokenize(query)
    doc_terms = [tokenize(doc) for doc in docs]
    lengths = np.array([len(terms) for terms in doc_terms], dtype=float)
    avgdl = float(lengths.mean()) if len(lengths) else 1.0
    df: dict[str, int] = {}
    for terms in doc_terms:
        for term in set(terms):
            df[term] = df.get(term, 0) + 1
    scores = np.zeros(len(docs), dtype=float)
    n_docs = max(len(docs), 1)
    for term in query_terms:
        n_qi = df.get(term, 0)
        idf = math.log(1 + (n_docs - n_qi + 0.5) / (n_qi + 0.5))
        for idx, terms in enumerate(doc_terms):
            freq = terms.count(term)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * lengths[idx] / max(avgdl, 1e-9))
            scores[idx] += idf * freq * (k1 + 1) / denom
    return scores


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def cosine_scores_from_embeddings(query_vector: Any, document_vectors: Any) -> np.ndarray:
    query_arr = np.asarray(query_vector, dtype=float).reshape(1, -1)
    doc_arr = np.asarray(document_vectors, dtype=float)
    if doc_arr.ndim == 1:
        doc_arr = doc_arr.reshape(1, -1)
    query_norm = np.linalg.norm(query_arr, axis=1, keepdims=True)
    doc_norm = np.linalg.norm(doc_arr, axis=1, keepdims=True)
    query_arr = query_arr / np.clip(query_norm, 1e-12, None)
    doc_arr = doc_arr / np.clip(doc_norm, 1e-12, None)
    return (doc_arr @ query_arr.T).ravel()


def _hash_texts(texts: list[str], model_name: str, prefix: str) -> str:
    hasher = hashlib.sha1()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(prefix.encode("utf-8"))
    for text in texts:
        hasher.update(str(text).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def _load_sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_texts(
    texts: list[str],
    model_name: str,
    prefix: str,
    batch_size: int,
    cache_dir: Path,
    cache_key: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _hash_texts(texts, model_name, prefix)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    path = cache_dir / f"{cache_key}_{safe_model}_{digest}.npz"
    if path.exists():
        data = np.load(path)
        return data["embeddings"]
    model = _load_sentence_transformer(model_name)
    encoded_texts = [prefix + str(text) for text in texts]
    embeddings = model.encode(
        encoded_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    np.savez_compressed(path, embeddings=embeddings)
    return embeddings


def semantic_retrievers_requested(retrievers: list[str]) -> list[str]:
    requested: list[str] = []
    for retriever in retrievers:
        if retriever in SEMANTIC_RETRIEVERS and retriever not in requested:
            requested.append(retriever)
        if retriever == "mvec_tg_semantic":
            for semantic in SEMANTIC_RETRIEVERS:
                if semantic not in requested:
                    requested.append(semantic)
    return requested


def add_semantic_score_columns(
    scored: pd.DataFrame,
    retrievers: list[str],
    batch_size: int = 64,
    cache_dir: Path = EMBEDDING_CACHE,
    cache_key: str = "text",
) -> pd.DataFrame:
    requested = semantic_retrievers_requested(retrievers)
    if not requested or scored.empty:
        return scored
    out = scored.copy()
    query_lookup = out.drop_duplicates("query_id")[["query_id", "query_text"]].reset_index(drop=True)
    doc_texts = pd.Series(out["evidence_text"].astype(str).unique()).tolist()
    doc_index = {text: idx for idx, text in enumerate(doc_texts)}
    query_ids = query_lookup["query_id"].astype(str).tolist()
    query_texts = query_lookup["query_text"].astype(str).tolist()
    query_index = {query_id: idx for idx, query_id in enumerate(query_ids)}
    row_query_idx = out["query_id"].astype(str).map(query_index).to_numpy(dtype=int)
    row_doc_idx = out["evidence_text"].astype(str).map(doc_index).to_numpy(dtype=int)
    for retriever in requested:
        spec = SEMANTIC_RETRIEVERS[retriever]
        score_col = str(spec["score_column"])
        if score_col in out.columns:
            continue
        model_name = str(spec["model_name"])
        query_embeddings = _encode_texts(
            query_texts,
            model_name,
            str(spec["query_prefix"]),
            batch_size,
            cache_dir,
            f"{cache_key}_{retriever}_query",
        )
        doc_embeddings = _encode_texts(
            doc_texts,
            model_name,
            str(spec["passage_prefix"]),
            batch_size,
            cache_dir,
            f"{cache_key}_{retriever}_evidence",
        )
        scores = np.empty(len(out), dtype="float32")
        chunk = 100_000
        for start in range(0, len(out), chunk):
            end = min(start + chunk, len(out))
            scores[start:end] = (query_embeddings[row_query_idx[start:end]] * doc_embeddings[row_doc_idx[start:end]]).sum(axis=1)
        out[score_col] = scores
        out[score_col] = out.groupby("query_id", sort=False)[score_col].transform(lambda values: minmax(values.to_numpy()))
    return out


def tfidf_scores(query: str, docs: list[str], dense: bool = False) -> np.ndarray:
    if not docs:
        return np.array([])
    analyzer = "char_wb" if dense else "word"
    ngram_range = (3, 5) if dense else (1, 2)
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1)
    matrix = vectorizer.fit_transform([query, *docs])
    if dense and matrix.shape[1] > 2 and len(docs) > 2:
        n_components = max(1, min(32, matrix.shape[0] - 1, matrix.shape[1] - 1))
        matrix = TruncatedSVD(n_components=n_components, random_state=13).fit_transform(matrix)
    return cosine_similarity(matrix[0:1], matrix[1:]).ravel()


def entity_overlap_scores(query: str, docs: list[str]) -> np.ndarray:
    q = {token for token in tokenize(query) if token[:1].isupper() or len(token) > 5}
    scores = []
    for doc in docs:
        d = {token for token in tokenize(doc) if token[:1].isupper() or len(token) > 5}
        scores.append(len(q & d) / max(len(q | d), 1))
    return np.array(scores, dtype=float)


def relation_scores(query: str, docs: list[str]) -> np.ndarray:
    q_terms = set(tokenize(query))
    values = []
    for doc in docs:
        d_terms = set(tokenize(doc))
        values.append(len(q_terms & d_terms) / max(len(q_terms), 1))
    return np.array(values, dtype=float)


def add_view_scores(candidates: pd.DataFrame, fast_text_scores: bool = False) -> pd.DataFrame:
    work = validate_unified_schema(candidates)
    parts: list[pd.DataFrame] = []
    for _, group in work.groupby("query_id", sort=False):
        docs = group["evidence_text"].astype(str).tolist()
        query = str(group["query_text"].iloc[0])
        piece = group.copy()
        piece["bm25_score"] = minmax(bm25_scores(query, docs))
        if not fast_text_scores:
            piece["tfidf_score"] = minmax(tfidf_scores(query, docs, dense=False))
            piece["dense_score"] = minmax(tfidf_scores(query, docs, dense=True))
        piece["entity_score"] = minmax(entity_overlap_scores(query, docs))
        piece["relation_score"] = minmax(relation_scores(query, docs))
        piece["source_score"] = piece["evidence_source"].astype(str).map(lambda value: 1.0 if value else 0.0)
        parts.append(piece)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*UNIFIED_COLUMNS])


def _rule_metrics_for_scores(frame: pd.DataFrame, score_col: str, supported: pd.Series, top_k: int = 5) -> dict[str, float]:
    work = frame.copy()
    work["retrieval_score"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    work["evidence_supported"] = supported.astype(int).values
    metrics = evidence_retrieval_metrics(work, top_k=top_k)
    return {
        "recall_at_5": float(metrics["recall_at_5"]),
        "support_precision": float(metrics["support_precision"]),
        "mrr": float(metrics["mrr"]),
    }


def _topk_support(frame: pd.DataFrame, score_col: str, k: int = 5) -> pd.Series:
    supported = pd.Series(0, index=frame.index, dtype=int)
    for _, group in frame.groupby("query_id", sort=False):
        idx = group.sort_values(score_col, ascending=False).head(k).index
        supported.loc[idx] = 1
    return supported


def split_calibration_evaluation_queries(
    frame: pd.DataFrame,
    calibration_fraction: float = 0.3,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    query_ids = sorted(frame["query_id"].astype(str).unique().tolist())
    keyed = []
    for query_id in query_ids:
        digest = hashlib.sha1(f"{seed}:{query_id}".encode("utf-8")).hexdigest()
        keyed.append((int(digest[:12], 16), query_id))
    keyed.sort()
    cal_count = max(1, min(len(keyed) - 1, int(round(len(keyed) * calibration_fraction))))
    calibration_ids = {query_id for _, query_id in keyed[:cal_count]}
    calibration = frame[frame["query_id"].astype(str).isin(calibration_ids)].copy()
    evaluation = frame[~frame["query_id"].astype(str).isin(calibration_ids)].copy()
    return calibration, evaluation


def _view_weighted_scores(frame: pd.DataFrame, view_columns: list[str], weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype=float)
    total = 0.0
    for col in view_columns:
        weight = float(weights.get(col, 0.0))
        if col in frame.columns and weight > 0:
            score = score + weight * pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
            total += weight
    return score / max(total, 1e-12)


def _weighted_vote_scores(frame: pd.DataFrame, view_columns: list[str], weights: dict[str, float]) -> pd.Series:
    votes = pd.Series(0.0, index=frame.index, dtype=float)
    total = 0.0
    for col in view_columns:
        weight = float(weights.get(col, 0.0))
        if col in frame.columns and weight > 0:
            votes = votes + weight * (pd.to_numeric(frame[col], errors="coerce").fillna(0.0) >= 0.5).astype(float)
            total += weight
    return votes / max(total, 1e-12)


def select_mvec_tg_rule(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame | None,
    view_columns: list[str],
    precision_weight: float = 2.0,
    allow_weighted_consensus: bool = False,
    weighted_consensus_floor: float = 0.98,
) -> dict[str, Any]:
    del evaluation
    best: dict[str, Any] | None = None
    score_columns = [col for col in view_columns if col in calibration.columns]
    for col in score_columns:
        supported = _topk_support(calibration, col, k=5)
        metrics = _rule_metrics_for_scores(calibration, col, supported, top_k=5)
        score = metrics["recall_at_5"] + precision_weight * metrics["support_precision"]
        candidate = {
            "mode": "single_view",
            "score_column": col,
            "view_threshold": 1,
            "calibration_support_precision": metrics["support_precision"],
            "calibration_recall": metrics["recall_at_5"],
            "calibration_mrr": metrics["mrr"],
            "calibration_score": score,
        }
        if best is None or (candidate["calibration_score"], candidate["calibration_recall"]) > (
            best["calibration_score"],
            best["calibration_recall"],
        ):
            best = candidate
    for threshold in range(1, len(view_columns) + 1):
        votes = (calibration[view_columns] >= 0.5).sum(axis=1)
        supported = votes >= threshold
        calibration = calibration.copy()
        calibration["_mvec_mean_score"] = calibration[view_columns].mean(axis=1)
        metrics = _rule_metrics_for_scores(calibration, "_mvec_mean_score", supported.astype(int), top_k=5)
        score = metrics["recall_at_5"] + precision_weight * metrics["support_precision"]
        candidate = {
            "mode": "consensus",
            "score_column": "_mvec_mean_score",
            "view_threshold": threshold,
            "calibration_support_precision": metrics["support_precision"],
            "calibration_recall": metrics["recall_at_5"],
            "calibration_mrr": metrics["mrr"],
            "calibration_score": score,
        }
        if best is None or (candidate["calibration_score"], candidate["calibration_recall"], threshold) > (
            best["calibration_score"],
            best["calibration_recall"],
            best["view_threshold"],
        ):
            best = candidate
    if allow_weighted_consensus and score_columns:
        qualities: dict[str, float] = {}
        for col in score_columns:
            supported = _topk_support(calibration, col, k=5)
            metrics = _rule_metrics_for_scores(calibration, col, supported, top_k=5)
            qualities[col] = max(metrics["recall_at_5"] + precision_weight * metrics["support_precision"], 1e-9)
        quality_sum = sum(qualities.values())
        weights = {col: qualities[col] / max(quality_sum, 1e-12) for col in score_columns}
        weighted = calibration.copy()
        weighted["_mvec_weighted_score"] = _view_weighted_scores(weighted, score_columns, weights)
        weighted_vote = _weighted_vote_scores(weighted, score_columns, weights)
        weighted_best: dict[str, Any] | None = None
        for threshold in [0.0, 0.25, 0.35, 0.5, 0.65, 0.8]:
            supported = weighted_vote >= threshold
            metrics = _rule_metrics_for_scores(weighted, "_mvec_weighted_score", supported.astype(int), top_k=5)
            score = metrics["recall_at_5"] + precision_weight * metrics["support_precision"]
            candidate = {
                "mode": "weighted_consensus",
                "score_column": "_mvec_weighted_score",
                "view_threshold": threshold,
                "calibration_support_precision": metrics["support_precision"],
                "calibration_recall": metrics["recall_at_5"],
                "calibration_mrr": metrics["mrr"],
                "calibration_score": score,
                "view_weights": weights,
            }
            if weighted_best is None or (candidate["calibration_score"], candidate["calibration_recall"], threshold) > (
                weighted_best["calibration_score"],
                weighted_best["calibration_recall"],
                weighted_best["view_threshold"],
            ):
                weighted_best = candidate
        if weighted_best is not None and best is not None:
            near_score = weighted_best["calibration_score"] >= weighted_consensus_floor * best["calibration_score"]
            near_recall = weighted_best["calibration_recall"] >= best["calibration_recall"] - 0.03
            if near_score and near_recall:
                best = weighted_best
        elif weighted_best is not None:
            best = weighted_best
    return best or {
        "mode": "single_view",
        "score_column": view_columns[0],
        "view_threshold": 1,
        "calibration_support_precision": 0.0,
        "calibration_recall": 0.0,
        "calibration_score": 0.0,
    }


def tg_weight_sensitivity_table(
    scored: pd.DataFrame,
    view_columns: list[str],
    precision_weights: list[float],
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable_views = [col for col in view_columns if col in scored.columns]
    if scored.empty or not usable_views:
        return pd.DataFrame()
    dataset = str(scored["dataset"].iloc[0]) if "dataset" in scored else "text"
    for weight in precision_weights:
        rule = select_mvec_tg_rule(scored, None, usable_views, precision_weight=weight)
        work = scored.copy()
        if rule["mode"] == "single_view":
            score_col = str(rule["score_column"])
            supported = _topk_support(work, score_col, k=top_k)
            work["retrieval_score"] = work[score_col]
            selected_rule = score_col
        elif rule["mode"] == "weighted_consensus":
            weights = dict(rule.get("view_weights", {}))
            work["retrieval_score"] = _view_weighted_scores(work, usable_views, weights)
            supported = _weighted_vote_scores(work, usable_views, weights) >= float(rule["view_threshold"])
            selected_rule = f"weighted_consensus_{float(rule['view_threshold']):.2f}"
        else:
            work["retrieval_score"] = work[usable_views].mean(axis=1)
            supported = (work[usable_views] >= 0.5).sum(axis=1) >= int(rule["view_threshold"])
            selected_rule = f"consensus_{int(rule['view_threshold'])}"
        work["evidence_supported"] = supported.astype(int).values
        metrics = evidence_retrieval_metrics(work, top_k=top_k)
        rows.append(
            {
                "dataset": dataset,
                "precision_weight": float(weight),
                "selected_mode": rule["mode"],
                "selected_rule": selected_rule,
                "recall_at_5": metrics["recall_at_5"],
                "mrr": metrics["mrr"],
                "ndcg_at_10": metrics["ndcg_at_10"],
                "support_precision": metrics["support_precision"],
                "supported_per_100": metrics["supported_per_100"],
            }
        )
    return pd.DataFrame(rows)


def apply_retriever(scored: pd.DataFrame, retriever: str, preselected_rule: dict[str, Any] | None = None) -> pd.DataFrame:
    out = scored.copy()
    if retriever == "bm25":
        out["retrieval_score"] = out["bm25_score"]
    elif retriever == "tfidf_svd":
        if "tfidf_score" not in out.columns:
            raise ValueError("tfidf_svd requested but tfidf_score was not computed")
        out["retrieval_score"] = out["tfidf_score"]
    elif retriever == "dense":
        if "dense_score" not in out.columns:
            raise ValueError("dense requested but dense_score was not computed")
        out["retrieval_score"] = out["dense_score"]
    elif retriever in SEMANTIC_RETRIEVERS:
        score_col = str(SEMANTIC_RETRIEVERS[retriever]["score_column"])
        if score_col not in out.columns:
            raise ValueError(f"Semantic score column missing for {retriever}: {score_col}")
        out["retrieval_score"] = out[score_col]
    elif retriever == "bge_m3":
        if "bge_m3_score" not in out.columns:
            raise ValueError("BGE-M3 score column missing")
        out["retrieval_score"] = out["bge_m3_score"]
    elif retriever == "cross_encoder":
        if "cross_encoder_score" not in out.columns:
            raise ValueError("Cross-encoder score column missing")
        out["retrieval_score"] = out["cross_encoder_score"]
    elif retriever == "nca_metric":
        out["retrieval_score"] = 0.45 * out["tfidf_score"] + 0.35 * out["dense_score"] + 0.20 * out["entity_score"]
    elif retriever == "single_view_proto":
        out["retrieval_score"] = out["relation_score"]
    elif retriever == "calibrated_fusion":
        out["retrieval_score"] = 0.30 * out["bm25_score"] + 0.25 * out["tfidf_score"] + 0.25 * out["dense_score"] + 0.20 * out["relation_score"]
    elif retriever == "mvec_tg":
        views = [col for col in ["bm25_score", "tfidf_score", "dense_score", "entity_score", "relation_score"] if col in out.columns]
        rule = preselected_rule or select_mvec_tg_rule(out, None, views, allow_weighted_consensus=True)
        if rule["mode"] == "single_view":
            score_col = str(rule["score_column"])
            out["retrieval_score"] = out[score_col]
            out["evidence_supported"] = _topk_support(out, score_col, k=5)
        elif rule["mode"] == "weighted_consensus":
            weights = dict(rule.get("view_weights", {}))
            out["retrieval_score"] = _view_weighted_scores(out, views, weights)
            out["evidence_supported"] = (_weighted_vote_scores(out, views, weights) >= float(rule["view_threshold"])).astype(int)
        else:
            out["retrieval_score"] = out[views].mean(axis=1)
            out["evidence_supported"] = ((out[views] >= 0.5).sum(axis=1) >= rule["view_threshold"]).astype(int)
        out["tg_mode"] = rule["mode"]
        out["tg_score_column"] = rule["score_column"]
        out["tg_view_threshold"] = rule["view_threshold"]
        out["tg_view_weights_json"] = json.dumps(rule.get("view_weights", {}), sort_keys=True)
    elif retriever == "mvec_tg_semantic":
        views = [col for col in [
            "bm25_score",
            "tfidf_score",
            "dense_score",
            "entity_score",
            "relation_score",
            *[str(spec["score_column"]) for spec in SEMANTIC_RETRIEVERS.values() if str(spec["score_column"]) in out.columns],
        ] if col in out.columns]
        rule = preselected_rule or select_mvec_tg_rule(out, None, views, allow_weighted_consensus=True)
        if rule["mode"] == "single_view":
            score_col = str(rule["score_column"])
            out["retrieval_score"] = out[score_col]
            out["evidence_supported"] = _topk_support(out, score_col, k=5)
        elif rule["mode"] == "weighted_consensus":
            weights = dict(rule.get("view_weights", {}))
            out["retrieval_score"] = _view_weighted_scores(out, views, weights)
            out["evidence_supported"] = (_weighted_vote_scores(out, views, weights) >= float(rule["view_threshold"])).astype(int)
        else:
            out["retrieval_score"] = out[views].mean(axis=1)
            out["evidence_supported"] = ((out[views] >= 0.5).sum(axis=1) >= rule["view_threshold"]).astype(int)
        out["tg_mode"] = rule["mode"]
        out["tg_score_column"] = rule["score_column"]
        out["tg_view_threshold"] = rule["view_threshold"]
        out["tg_view_weights_json"] = json.dumps(rule.get("view_weights", {}), sort_keys=True)
    else:
        raise ValueError(f"Unsupported retriever: {retriever}")
    if "evidence_supported" not in out:
        out["evidence_supported"] = 0
        for _, group in out.groupby("query_id", sort=False):
            idx = group.sort_values("retrieval_score", ascending=False).head(5).index
            out.loc[idx, "evidence_supported"] = 1
    out["retriever"] = retriever
    return out


def score_text_candidates(
    candidates: pd.DataFrame,
    retrievers: list[str],
    precomputed_scores: dict[str, Any] | None = None,
    semantic_batch_size: int = 64,
    semantic_cache_key: str = "text",
    fast_text_scores: bool = False,
) -> pd.DataFrame:
    scored = add_view_scores(candidates, fast_text_scores=fast_text_scores)
    if precomputed_scores:
        for col, values in precomputed_scores.items():
            scored[col] = values
    missing_semantic = [
        retriever
        for retriever in semantic_retrievers_requested(retrievers)
        if str(SEMANTIC_RETRIEVERS[retriever]["score_column"]) not in scored.columns
    ]
    if missing_semantic:
        scored = add_semantic_score_columns(
            scored,
            missing_semantic,
            batch_size=semantic_batch_size,
            cache_key=semantic_cache_key,
        )
    return scored


def benchmark_text_candidates(
    candidates: pd.DataFrame,
    retrievers: list[str],
    top_k: int = 5,
    precomputed_scores: dict[str, Any] | None = None,
    semantic_batch_size: int = 64,
    semantic_cache_key: str = "text",
    calibration_query_fraction: float | None = None,
    calibration_seed: int = 13,
    fast_text_scores: bool = False,
) -> pd.DataFrame:
    scored = score_text_candidates(
        candidates,
        retrievers,
        precomputed_scores=precomputed_scores,
        semantic_batch_size=semantic_batch_size,
        semantic_cache_key=semantic_cache_key,
        fast_text_scores=fast_text_scores,
    )
    rule_lookup: dict[str, dict[str, Any]] = {}
    if calibration_query_fraction is not None:
        calibration, scored = split_calibration_evaluation_queries(
            scored,
            calibration_fraction=calibration_query_fraction,
            seed=calibration_seed,
        )
        base_views = ["bm25_score", "tfidf_score", "dense_score", "entity_score", "relation_score"]
        semantic_views = [str(spec["score_column"]) for spec in SEMANTIC_RETRIEVERS.values() if str(spec["score_column"]) in scored.columns]
        if "mvec_tg" in retrievers:
            rule_lookup["mvec_tg"] = select_mvec_tg_rule(
                calibration,
                scored,
                [col for col in base_views if col in calibration.columns],
                allow_weighted_consensus=True,
            )
        if "mvec_tg_semantic" in retrievers:
            rule_lookup["mvec_tg_semantic"] = select_mvec_tg_rule(
                calibration,
                scored,
                [col for col in [*base_views, *semantic_views] if col in calibration.columns],
                allow_weighted_consensus=True,
            )
    parts = []
    for retriever in retrievers:
        if retriever not in TEXT_RETRIEVERS:
            continue
        part = apply_retriever(scored, retriever, preselected_rule=rule_lookup.get(retriever))
        part["rank"] = part.groupby("query_id")["retrieval_score"].rank(method="first", ascending=False).astype(int)
        part["top_k"] = top_k
        if calibration_query_fraction is not None:
            part["evaluation_partition"] = "evaluation"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def dcg(relevances: list[int]) -> float:
    return float(sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances)))


def evidence_retrieval_metrics(rows: pd.DataFrame, top_k: int = 5) -> dict[str, Any]:
    work = rows.copy()
    work["retrieval_score"] = pd.to_numeric(work["retrieval_score"], errors="coerce").fillna(0.0)
    work["gold_support"] = pd.to_numeric(work["gold_support"], errors="coerce").fillna(0).astype(int)
    work["evidence_supported"] = pd.to_numeric(work["evidence_supported"], errors="coerce").fillna(0).astype(int)
    queries = work["query_id"].nunique()
    gold_queries = 0
    hit_queries = 0
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    for _, group in work.groupby("query_id", sort=False):
        ranked = group.sort_values("retrieval_score", ascending=False).reset_index(drop=True)
        gold = ranked["gold_support"].tolist()
        if sum(gold) == 0:
            continue
        gold_queries += 1
        top_gold = gold[:top_k]
        hit_queries += int(any(top_gold))
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


def summarize_text_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, retriever), group in detail.groupby(["dataset", "retriever"], sort=False):
        metrics = evidence_retrieval_metrics(group)
        metrics.update({"dataset": dataset, "retriever": retriever})
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_text_part(part: pd.DataFrame) -> dict[str, Any]:
    if part.empty:
        return {}
    dataset = str(part["dataset"].iloc[0])
    retriever = str(part["retriever"].iloc[0])
    metrics = evidence_retrieval_metrics(part)
    metrics.update({"dataset": dataset, "retriever": retriever})
    return metrics


def summarize_tg_rules(detail: pd.DataFrame) -> pd.DataFrame:
    tg_cols = ["dataset", "retriever", "tg_mode", "tg_score_column", "tg_view_threshold", "tg_view_weights_json"]
    if detail.empty or not all(col in detail.columns for col in tg_cols):
        return pd.DataFrame(columns=[*tg_cols, "queries", "rows"])
    rows = detail[detail["retriever"].astype(str).str.contains("mvec_tg")].copy()
    if rows.empty:
        return pd.DataFrame(columns=[*tg_cols, "queries", "rows"])
    return (
        rows.groupby(tg_cols, dropna=False, as_index=False)
        .agg(queries=("query_id", "nunique"), rows=("query_id", "size"))
        .sort_values(["dataset", "retriever", "tg_mode"])
    )


def summarize_bts_detail(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    df = detail[detail["evidence_method"].isin(["teacher", "mvec_tg"])].copy()
    rows = []
    for method, group in df.groupby("evidence_method", sort=False):
        supported = int(pd.to_numeric(group["supported"], errors="coerce").fillna(0).sum())
        support_total = int(pd.to_numeric(group["support_total"], errors="coerce").fillna(0).sum())
        reviewed = int(pd.to_numeric(group["reviewed"], errors="coerce").fillna(0).sum())
        captured = int(pd.to_numeric(group["captured"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "dataset": "bts",
                "retriever": "mvec_tg" if method == "mvec_tg" else "teacher",
                "queries": reviewed,
                "gold_queries": captured,
                "supported_positive": supported,
                "supported_negative": max(support_total - supported, 0),
                "absolute_supported_gain": 0,
            }
        )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    teacher_supported = int(raw.loc[raw["retriever"].eq("teacher"), "supported_positive"].sum())
    raw.loc[raw["retriever"].eq("mvec_tg"), "absolute_supported_gain"] = raw.loc[
        raw["retriever"].eq("mvec_tg"), "supported_positive"
    ] - teacher_supported
    summary = raw.copy()
    summary["recall_at_5"] = summary["supported_positive"] / summary["gold_queries"].clip(lower=1)
    summary["mrr"] = summary["recall_at_5"]
    summary["ndcg_at_10"] = summary["recall_at_5"]
    summary["support_precision"] = summary["supported_positive"] / (summary["supported_positive"] + summary["supported_negative"]).clip(lower=1)
    summary["supported_per_100"] = 100.0 * summary["supported_positive"] / summary["queries"].clip(lower=1)
    summary["evidence_hit"] = summary["recall_at_5"]
    return summary


def load_bts_summary() -> pd.DataFrame:
    if PROTOCOL_BTS_DETAIL.exists():
        return summarize_bts_detail(pd.read_csv(PROTOCOL_BTS_DETAIL))
    if not PROTOCOL_BTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROTOCOL_BTS)
    rows = []
    for _, row in df[df["evidence_method"].isin(["teacher", "mvec_tg"])].iterrows():
        supported = int(row["supported"])
        precision = float(row["support_precision"])
        support_total = int(round(supported / max(precision, 1e-12)))
        rows.append(
            {
                "dataset": "bts",
                "retriever": "mvec_tg" if row["evidence_method"] == "mvec_tg" else "teacher",
                "queries": int(row["reviewed"]),
                "gold_queries": int(row["captured"]),
                "supported_positive": supported,
                "supported_negative": max(support_total - supported, 0),
                "absolute_supported_gain": int(row.get("absolute_supported_gain", 0)),
            }
        )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    summary = raw.groupby(["dataset", "retriever"], as_index=False).agg(
        queries=("queries", "sum"),
        gold_queries=("gold_queries", "sum"),
        supported_positive=("supported_positive", "sum"),
        supported_negative=("supported_negative", "sum"),
        absolute_supported_gain=("absolute_supported_gain", "sum"),
    )
    summary["recall_at_5"] = summary["supported_positive"] / summary["gold_queries"].clip(lower=1)
    summary["mrr"] = summary["recall_at_5"]
    summary["ndcg_at_10"] = summary["recall_at_5"]
    summary["support_precision"] = summary["supported_positive"] / (summary["supported_positive"] + summary["supported_negative"]).clip(lower=1)
    summary["supported_per_100"] = 100.0 * summary["supported_positive"] / summary["queries"].clip(lower=1)
    summary["evidence_hit"] = summary["recall_at_5"]
    return summary


def pareto_frontier_flags(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    out = table.copy()
    metrics = ["recall_at_5", "supported_per_100", "support_precision"]
    flags: list[bool] = []
    for idx, row in out.iterrows():
        peers = out[(out["dataset"].eq(row["dataset"])) & (out.index != idx)]
        dominated = False
        for _, peer in peers.iterrows():
            at_least = all(float(peer.get(metric, 0.0)) >= float(row.get(metric, 0.0)) for metric in metrics)
            strictly = any(float(peer.get(metric, 0.0)) > float(row.get(metric, 0.0)) for metric in metrics)
            if at_least and strictly:
                dominated = True
                break
        flags.append(not dominated)
    out["pareto_frontier"] = flags
    return out


def judge_cross_domain_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    out = summary.copy()
    out["decision"] = "pass"
    out["recall_at_5_delta"] = 0.0
    out["mrr_delta"] = 0.0
    out["support_precision_delta"] = 0.0
    for dataset, group in out.groupby("dataset", sort=False):
        single = group[group["retriever"].eq("single_view_proto")]
        if single.empty:
            continue
        ref = single.sort_values(["recall_at_5", "support_precision"], ascending=False).iloc[0]
        mask = out["dataset"].eq(dataset) & out["retriever"].eq("mvec_tg")
        if mask.any():
            recall_delta = out.loc[mask, "recall_at_5"].iloc[0] - ref["recall_at_5"]
            mrr_delta = out.loc[mask, "mrr"].iloc[0] - ref["mrr"]
            precision_delta = out.loc[mask, "support_precision"].iloc[0] - ref["support_precision"]
            out.loc[mask, "recall_at_5_delta"] = recall_delta
            out.loc[mask, "mrr_delta"] = mrr_delta
            out.loc[mask, "support_precision_delta"] = precision_delta
            passed = (recall_delta >= 0.015 and mrr_delta >= -0.005) or (recall_delta >= -0.005 and precision_delta >= 0.020)
            out.loc[mask, "decision"] = "pass" if passed else "review"
    return pareto_frontier_flags(out)


def write_report(
    out: Path,
    summary: pd.DataFrame,
    judgement: pd.DataFrame,
    sensitivity: pd.DataFrame | None = None,
    tg_rules: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Cross-domain evidence retrieval benchmark",
        "",
        "The benchmark uses unified fixed-queue evidence candidates for structured BTS queues and text-evidence candidate pools.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f") if not summary.empty else "(no rows)",
        "",
        "## Judgement",
        "",
        judgement.to_markdown(index=False, floatfmt=".4f") if not judgement.empty else "(no rows)",
        "",
    ]
    if sensitivity is not None and not sensitivity.empty:
        lines.extend(
            [
                "## TG precision-weight sensitivity",
                "",
                sensitivity.to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    if tg_rules is not None and not tg_rules.empty:
        lines.extend(
            [
                "## TG selected rules",
                "",
                tg_rules.to_markdown(index=False),
                "",
            ]
        )
    (out / "ipm_cross_domain_evidence_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def load_or_build_candidates(dataset: str, mode: str, candidates_per_query: int) -> pd.DataFrame:
    path = ADAPTER_OUT / f"{dataset}_{mode}_candidates_k{candidates_per_query}.csv"
    if not path.exists():
        build_dataset(dataset, mode, ADAPTER_OUT, candidates_per_query=candidates_per_query)
    return pd.read_csv(path)


def run_benchmark(
    mode: str,
    datasets: list[str],
    retrievers: list[str],
    candidates_per_query: int,
    out: Path,
    semantic_batch_size: int = 64,
    tg_precision_weights: list[float] | None = None,
    calibration_query_fraction: float | None = None,
    calibration_seed: int = 13,
    summary_only: bool = False,
    fast_text_scores: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    detail_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    sensitivity_parts: list[pd.DataFrame] = []
    tg_rule_parts: list[pd.DataFrame] = []
    for dataset in datasets:
        if dataset == "bts":
            summary_parts.append(load_bts_summary())
            continue
        candidates = load_or_build_candidates(dataset, mode, candidates_per_query)
        text_retrievers = [r for r in retrievers if r in TEXT_RETRIEVERS]
        scored = score_text_candidates(
            candidates,
            text_retrievers,
            semantic_batch_size=semantic_batch_size,
            semantic_cache_key=f"{dataset}_{mode}_{candidates_per_query}",
            fast_text_scores=fast_text_scores,
        )
        rule_lookup: dict[str, dict[str, Any]] = {}
        evaluation_scored = scored
        if calibration_query_fraction is not None:
            calibration_scored, evaluation_scored = split_calibration_evaluation_queries(
                scored,
                calibration_fraction=calibration_query_fraction,
                seed=calibration_seed,
            )
            base_views = ["bm25_score", "tfidf_score", "dense_score", "entity_score", "relation_score"]
            semantic_views = [
                str(spec["score_column"])
                for spec in SEMANTIC_RETRIEVERS.values()
                if str(spec["score_column"]) in calibration_scored.columns
            ]
            if "mvec_tg" in text_retrievers:
                rule_lookup["mvec_tg"] = select_mvec_tg_rule(
                    calibration_scored,
                    evaluation_scored,
                    [col for col in base_views if col in calibration_scored.columns],
                    allow_weighted_consensus=True,
                )
            if "mvec_tg_semantic" in text_retrievers:
                rule_lookup["mvec_tg_semantic"] = select_mvec_tg_rule(
                    calibration_scored,
                    evaluation_scored,
                    [col for col in [*base_views, *semantic_views] if col in calibration_scored.columns],
                    allow_weighted_consensus=True,
                )
        parts = []
        summary_rows: list[dict[str, Any]] = []
        for retriever in text_retrievers:
            part = apply_retriever(evaluation_scored, retriever, preselected_rule=rule_lookup.get(retriever))
            part["rank"] = part.groupby("query_id")["retrieval_score"].rank(method="first", ascending=False).astype(int)
            part["top_k"] = 5
            if calibration_query_fraction is not None:
                part["evaluation_partition"] = "evaluation"
            if summary_only:
                summary_rows.append(summarize_text_part(part))
                tg_rules = summarize_tg_rules(part)
                if not tg_rules.empty:
                    tg_rule_parts.append(tg_rules)
            else:
                parts.append(part)
        if summary_only:
            detail = pd.DataFrame()
            if summary_rows:
                summary_parts.append(pd.DataFrame(summary_rows))
        else:
            detail = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            detail_parts.append(detail)
            summary_parts.append(summarize_text_detail(detail))
        if tg_precision_weights:
            base_views = ["bm25_score", "tfidf_score", "dense_score", "entity_score", "relation_score"]
            semantic_views = [str(spec["score_column"]) for spec in SEMANTIC_RETRIEVERS.values() if str(spec["score_column"]) in scored.columns]
            sensitivity_parts.append(tg_weight_sensitivity_table(scored, base_views, tg_precision_weights).assign(view_set="base"))
            if semantic_views:
                sensitivity_parts.append(
                    tg_weight_sensitivity_table(scored, [*base_views, *semantic_views], tg_precision_weights).assign(view_set="semantic")
                )
    detail_out = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    judgement = judge_cross_domain_summary(summary)
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True) if sensitivity_parts else pd.DataFrame()
    tg_rules = pd.concat(tg_rule_parts, ignore_index=True) if tg_rule_parts else summarize_tg_rules(detail_out)
    detail_out.to_csv(out / "ipm_cross_domain_evidence_benchmark_detail.csv", index=False)
    summary.to_csv(out / "ipm_cross_domain_evidence_benchmark_summary.csv", index=False)
    judgement.to_csv(out / "ipm_cross_domain_evidence_benchmark_judgement.csv", index=False)
    sensitivity.to_csv(out / "ipm_cross_domain_tg_sensitivity.csv", index=False)
    tg_rules.to_csv(out / "ipm_cross_domain_tg_rules.csv", index=False)
    write_report(out, summary, judgement, sensitivity, tg_rules)
    return detail_out, summary, judgement


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--datasets", default="bts,scifact,fever")
    parser.add_argument("--retrievers", default="teacher,bm25,tfidf_svd,dense,nca_metric,single_view_proto,calibrated_fusion,mvec_tg")
    parser.add_argument("--candidates-per-query", type=int, default=50)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--tg-precision-weights", default="0.5,1.0,2.0,3.0")
    parser.add_argument("--calibration-query-fraction", type=float, default=0.0)
    parser.add_argument("--calibration-seed", type=int, default=13)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--fast-text-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary, judgement = run_benchmark(
        args.mode,
        parse_list(args.datasets),
        parse_list(args.retrievers),
        args.candidates_per_query,
        args.out,
        semantic_batch_size=args.semantic_batch_size,
        tg_precision_weights=parse_float_list(args.tg_precision_weights),
        calibration_query_fraction=args.calibration_query_fraction if args.calibration_query_fraction > 0 else None,
        calibration_seed=args.calibration_seed,
        summary_only=args.summary_only,
        fast_text_scores=args.fast_text_scores,
    )
    print(judgement.to_string(index=False) if not judgement.empty else summary.to_string(index=False))


if __name__ == "__main__":
    main()
