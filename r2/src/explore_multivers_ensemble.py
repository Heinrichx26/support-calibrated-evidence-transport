"""Exploratory, leakage-controlled MultiVerS ensemble search for SciFact."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule

R2 = Path(__file__).resolve().parents[1]
ORIGINAL_VIEWS = list(admission.VIEWS)
OUT = R2 / "results/multivers_extension/ensemble_exploration"


def candidate_scores(x, rationale_rank=None, document_rank=None):
    """A fixed, small family of score combinations defined before test labels."""
    mv = x[:, 12]
    ce = x[:, 8]
    e5 = x[:, 6]
    bge = x[:, 7]
    nli_entail = x[:, 11]
    semantic = (ce + e5 + bge) / 3.0
    scores = {
        "multivers": mv,
        "mv_ce_geometric": np.sqrt(np.clip(mv * ce, 0, 1)),
        "mv_e5_geometric": np.sqrt(np.clip(mv * e5, 0, 1)),
        "mv_nli_geometric": np.sqrt(np.clip(mv * nli_entail, 0, 1)),
        "mv_min_ce": np.minimum(mv, ce),
        "mv_min_e5": np.minimum(mv, e5),
        "mv_min_semantic": np.minimum.reduce([mv, ce, e5, bge]),
        "mv_mean_ce": (mv + ce) / 2.0,
        "mv_mean_semantic": (mv + semantic) / 2.0,
        "mv_gate_ce": mv * (0.5 + 0.5 * ce),
        "mv_gate_nli": mv * (0.5 + 0.5 * nli_entail),
        "mv_gate_semantic": mv * (0.5 + 0.5 * semantic),
    }
    if rationale_rank is not None and document_rank is not None:
        scores.update({
            "mv_rationale_rank": rationale_rank,
            "mv_document_rank": document_rank,
            "mv_rationale_document_geometric": np.sqrt(np.clip(rationale_rank * document_rank, 0, 1)),
            "mv_rationale_document_min": np.minimum(rationale_rank, document_rank),
            "mv_rationale_heavy_document": 0.7 * rationale_rank + 0.3 * document_rank,
            "mv_document_heavy_rationale": 0.3 * rationale_rank + 0.7 * document_rank,
        })
    return scores


def scalar_rank(train, values):
    probs = np.linspace(0, 1, 1001)
    quantiles = np.quantile(train, probs)
    unique, first = np.unique(quantiles, return_index=True)
    last = np.searchsorted(quantiles, unique, side="right") - 1
    mids = (first + last) / (2 * (len(quantiles) - 1))
    return np.interp(values, unique, mids, left=0, right=1)


def frame_score(frame, fit_frame):
    scale = admission.fit_scale(fit_frame)
    x = admission.transform(frame, scale)
    if {"mv_rationale_raw", "mv_document_raw"}.issubset(frame.columns):
        rationale_rank = scalar_rank(fit_frame.mv_rationale_raw.to_numpy(float), frame.mv_rationale_raw.to_numpy(float))
        document_rank = scalar_rank(fit_frame.mv_document_raw.to_numpy(float), frame.mv_document_raw.to_numpy(float))
        return candidate_scores(x, rationale_rank, document_rank)
    return candidate_scores(x)


def precision_at_budget(frame, score, queries):
    rule = budget_rule(frame, score, len(queries))
    rank = admission.ranks(frame, score)
    admitted, _ = admission.apply_threshold(frame, score, rule)
    correct = int((admitted & frame.gold_evidence.to_numpy(bool)).sum())
    return correct / max(int(admitted.sum()), 1), rule


def inner_cv_select(fitting, seed):
    """Choose a formula with query-disjoint folds inside the fitting partition."""
    qids = sorted(fitting.query_id.unique(), key=lambda q: hashlib.sha256(f"{seed}:ensemble:{q}".encode()).hexdigest())
    fold = {q: i % 4 for i, q in enumerate(qids)}
    probe = frame_score(fitting.iloc[:1].copy(), fitting)
    scores = {name: [] for name in probe}
    for f in range(4):
        train = fitting[fitting.query_id.map(fold).to_numpy() != f].copy()
        valid = fitting[fitting.query_id.map(fold).to_numpy() == f].copy()
        for name, score in frame_score(valid, train).items():
            value, _ = precision_at_budget(valid, score, valid.query_id.unique())
            scores[name].append(value)
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    selected = max(means, key=lambda name: (means[name], name))
    return selected, means, scores


def load_frame(pool=100):
    frame = pd.read_parquet(R2 / f"data/scored/scifact_full_k{pool}.parquet")
    mv = pd.read_parquet(R2 / "results/strengthening/multivers/candidate_scores.parquet")
    frame = frame.merge(mv, on=["query_id", "candidate_id"], how="left", validate="one_to_one", sort=False)
    doc_path = R2 / "results/strengthening/multivers/document_scores.jsonl"
    documents = pd.read_json(doc_path, lines=True)
    records = []
    for row in documents.itertuples(index=False):
        probs = np.asarray(row.label_probabilities, dtype=float)
        for sentence_id, rationale in enumerate(row.rationale_probabilities):
            records.append({
                "query_id": str(row.query_id),
                "doc_id": str(row.doc_id),
                "sentence_id": str(sentence_id),
                "mv_rationale_raw": float(rationale),
                "mv_document_raw": float(max(probs[0], probs[2])),
            })
    components = pd.DataFrame(records)
    frame["doc_id"] = frame.candidate_id.str.split("::").str[0]
    frame["sentence_id"] = frame.candidate_id.str.split("::").str[1]
    frame = frame.merge(components, on=["query_id", "doc_id", "sentence_id"], how="left", validate="one_to_one", sort=False)
    assert frame.mv_rationale_raw.notna().all() and frame.mv_document_raw.notna().all()
    assert frame.raw_multivers.notna().all() and np.isfinite(frame.raw_multivers).all()
    return frame


def run(smoke=False, pool=100):
    frame = load_frame(pool)
    out = OUT / (f"k{pool}_smoke" if smoke else f"k{pool}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with_view = ORIGINAL_VIEWS + ["raw_multivers"]
    old_views = admission.VIEWS
    admission.VIEWS = with_view
    try:
        for seed in [13] if smoke else [13, 17, 29, 7]:
            model = json.loads((R2 / f"results/experiments/scifact_full_k{pool}/model_seed{seed}.json").read_text())
            fitids = model["fitting_query_ids"]
            calids = model["threshold_query_ids"]
            testids = model["evaluation_query_ids"]
            if smoke:
                testids = sorted(testids)[:12]
            fitting = frame[frame.query_id.isin(fitids)].copy().reset_index(drop=True)
            calibration = frame[frame.query_id.isin(calids)].copy().reset_index(drop=True)
            test = frame[frame.query_id.isin(testids)].copy().reset_index(drop=True)
            selected, cv_means, cv_values = inner_cv_select(fitting, seed)
            fit_scores = frame_score(fitting, fitting)
            cal_scores = frame_score(calibration, fitting)
            test_scores = frame_score(test, fitting)
            for name in fit_scores:
                fit_precision, _ = precision_at_budget(fitting, fit_scores[name], fitting.query_id.unique())
                test_precision, rule = precision_at_budget(test, test_scores[name], test.query_id.unique())
                rows.append({
                    "seed": seed,
                    "method": name,
                    "selected_by_inner_cv": int(name == selected),
                    "inner_cv_precision": cv_means[name],
                    "fit_precision": fit_precision,
                    "test_precision": test_precision,
                    "test_items": int(len(testids)),
                    "threshold": float(rule["threshold"]),
                })
            print("seed", seed, "selected", selected, "cv", cv_means[selected], flush=True)
            (out / f"cv_seed{seed}.json").write_text(json.dumps({"selected": selected, "means": cv_means, "folds": cv_values}, indent=2), encoding="utf-8")
    finally:
        admission.VIEWS = old_views
    data = pd.DataFrame(rows)
    data.to_csv(out / "candidate_results.csv", index=False)
    selected = data[data.selected_by_inner_cv == 1]
    selected.to_csv(out / "selected_results.csv", index=False)
    audit = {
        "smoke": smoke,
        "pool": pool,
        "candidate_family": list(candidate_scores(np.zeros((1, 13)), np.zeros(1), np.zeros(1)).keys()),
        "selection": "four query-disjoint folds inside fitting partition",
        "test_labels_used_for_selection": False,
        "exact_budget": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(selected[["seed", "method", "inner_cv_precision", "test_precision"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pool", type=int, default=100, choices=[100])
    args = parser.parse_args()
    run(args.smoke, args.pool)
