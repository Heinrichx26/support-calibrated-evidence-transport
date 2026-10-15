"""Leakage-controlled SCET extension with a frozen MultiVerS rationale view."""
from pathlib import Path
import argparse
import hashlib
import json
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
from run_multivers_extension import paired_intervals

R2 = Path(__file__).resolve().parents[1]
BASE_VIEWS = list(admission.VIEWS)
OUT = R2 / "results/multivers_extension/rationale_extension"
SEEDS = [13, 17, 29, 7]


@contextmanager
def views():
    old = admission.VIEWS
    admission.VIEWS = BASE_VIEWS + ["raw_multivers", "raw_multivers_rationale"]
    try:
        yield
    finally:
        admission.VIEWS = old


def load_frame(pool=100):
    frame = pd.read_parquet(R2 / f"data/scored/scifact_full_k{pool}.parquet")
    mv = pd.read_parquet(R2 / "results/strengthening/multivers/candidate_scores.parquet")
    frame = frame.merge(mv, on=["query_id", "candidate_id"], how="left", validate="one_to_one", sort=False)
    docs = pd.read_json(R2 / "results/strengthening/multivers/document_scores.jsonl", lines=True)
    rationale = []
    for row in docs.itertuples(index=False):
        for sid, value in enumerate(row.rationale_probabilities):
            rationale.append({"query_id": str(row.query_id), "doc_id": str(row.doc_id),
                              "sentence_id": str(sid), "raw_multivers_rationale": float(value)})
    component = pd.DataFrame(rationale)
    frame["doc_id"] = frame.candidate_id.str.split("::").str[0]
    frame["sentence_id"] = frame.candidate_id.str.split("::").str[1]
    frame = frame.merge(component, on=["query_id", "doc_id", "sentence_id"], how="left",
                        validate="one_to_one", sort=False)
    assert frame.raw_multivers.notna().all()
    assert frame.raw_multivers_rationale.notna().all()
    return frame


def evaluate(smoke=False, pool=100):
    frame = load_frame(pool)
    out = OUT / (f"k{pool}_smoke" if smoke else f"k{pool}")
    out.mkdir(parents=True, exist_ok=True)
    queries = json.loads((R2 / "data/processed/scifact_full_queries.json").read_text(encoding="utf-8"))
    rows, choices, allstats = [], [], {}
    with views():
        for seed in [13] if smoke else SEEDS:
            model_path = R2 / f"results/experiments/scifact_full_k{pool}/model_seed{seed}.json"
            original = json.loads(model_path.read_text())
            fitids, calids, testids = [original[k] for k in ["fitting_query_ids", "threshold_query_ids", "evaluation_query_ids"]]
            eval_ids = sorted(testids)[:12] if smoke else testids
            fitting = frame[frame.query_id.isin(fitids)].copy().reset_index(drop=True)
            calibration = frame[frame.query_id.isin(calids)].copy().reset_index(drop=True)
            test = frame[frame.query_id.isin(eval_ids)].copy().reset_index(drop=True)
            assert set(fitids).isdisjoint(calids) and set(fitids).isdisjoint(testids) and set(calids).isdisjoint(testids)
            fitted = admission.fit_pipeline(fitting, calibration, seed)
            scores_fit, _, _ = admission.score_pipeline(fitting, fitted)
            scores_cal, _, _ = admission.score_pipeline(calibration, fitted)
            scores_test, stress, _ = admission.score_pipeline(test, fitted)
            qtest = {q: queries[q] for q in eval_ids}
            methods = {
                "scet_rationale_extension": scores_test["scet"],
                "multivers": test.raw_multivers.to_numpy(float),
                "multivers_rationale": test.raw_multivers_rationale.to_numpy(float),
            }
            stats = {}
            for name, score in methods.items():
                if name == "scet_rationale_extension":
                    cal_score = scores_cal["scet"]
                elif name == "multivers":
                    cal_score = calibration.raw_multivers.to_numpy(float)
                else:
                    cal_score = calibration.raw_multivers_rationale.to_numpy(float)
                threshold_rule = admission.fit_threshold(calibration, cal_score, 100)
                stat = query_statistics(test, score, threshold_rule, stress, qtest)
                stat.to_csv(out / f"query_seed{seed}_{name}_prospective.csv", index=False)
                rows.append({"seed": seed, "method": name, "analysis": "prospective_100",
                             **admission.summarize(stat)})
                batch_rule = budget_rule(test, score, len(eval_ids))
                batch_stat = query_statistics(test, score, batch_rule, stress, qtest)
                assert batch_stat.admitted.sum() == len(eval_ids)
                batch_stat.to_csv(out / f"query_seed{seed}_{name}_exact_budget.csv", index=False)
                rows.append({"seed": seed, "method": name, "analysis": "exact_budget",
                             **admission.summarize(batch_stat)})
                stats[name] = batch_stat
            choices.append({"seed": seed, "selected_form": fitted["selected_form"],
                            "penalty": fitted["penalty"], "primary_single": fitted["primary_single"]})
            allstats[seed] = stats
            print("seed", seed, "selected", fitted["selected_form"], "single", fitted["primary_single"], flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(choices).to_csv(out / "selected_forms.csv", index=False)
    if not smoke:
        intervals = []
        exact = {seed: stats for seed, stats in allstats.items()}
        for reference in ["multivers", "multivers_rationale"]:
            for row in paired_intervals(exact, "scet_rationale_extension", reference,
                                        ["evidence_precision", "annotated_per_100", "coverage", "evidence_recall", "all_facts_recall"]):
                intervals.append({"analysis": "exact_budget", **row})
        pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    audit = {"smoke": smoke, "pool": pool, "query_splits_disjoint": True,
             "test_labels_used_for_selection": False, "threshold_calibration_independent": True,
             "views": BASE_VIEWS + ["raw_multivers", "raw_multivers_rationale"],
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "elapsed_seconds": time.monotonic()}
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(data[data.analysis == "exact_budget"][["seed", "method", "evidence_precision", "coverage"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate(args.smoke)
