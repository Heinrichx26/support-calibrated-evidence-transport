"""FEVER text-availability sensitivity using frozen models and saved outcomes.

No encoder inference or model fitting is performed. Subsets depend exclusively
on original candidate identities and verified original-snapshot text.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2_admission import PROTOCOL, VIEWS, ranks, score_pipeline, summarize
from r2_statistics import query_statistics

R2 = Path(__file__).resolve().parents[1]
ROOT = R2.parent
OUT = R2 / "results/evidence_completion/fever_missingness"
EXP = R2 / "results/experiments/fever_full_k100"
SCORED = R2 / "data/scored/fever_full_k100.parquet"
METHODS = ["scet", "ce", "e5", "bm25", "nli_relevance"]
SEEDS = PROTOCOL["seeds"]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def availability(out):
    queries = json.loads((R2 / "data/processed/fever_full_queries.json").read_text(encoding="utf-8"))
    verified = set(pd.read_parquet(R2 / "data/raw/fever/verified_sentences.parquet", columns=["candidate_id"]).candidate_id)
    counts, provenance = [], []
    original = ROOT / "data/interim/evidence_retrieval/fever_full_candidates_k100.csv"
    for chunk in pd.read_csv(original, usecols=["query_id", "candidate_id", "gold_support"], dtype={"query_id": str, "candidate_id": str}, chunksize=100000):
        chunk["available"] = chunk.candidate_id.isin(verified)
        chunk["claim_verdict"] = chunk.query_id.map({q: d["claim_verdict"] for q, d in queries.items()})
        assert chunk.claim_verdict.notna().all()
        chunk["missing"] = (~chunk.available).astype(int)
        chunk["missing_gold"] = chunk.missing * chunk.gold_support
        counts.append(chunk.groupby("query_id")[["available", "missing", "gold_support", "missing_gold"]].sum())
        provenance.append(chunk.groupby(["claim_verdict", "gold_support", "available"]).size().rename("rows"))
    q = pd.concat(counts).groupby(level=0).sum().rename(columns={"gold_support": "original_gold_rows", "available": "verified_rows", "missing": "missing_rows"})
    q["original_rows"] = q.verified_rows + q.missing_rows
    q["missing_other"] = q.missing_rows - q.missing_gold
    q["claim_verdict"] = [queries[i]["claim_verdict"] for i in q.index]
    q["gold_count"] = [queries[i]["gold_count"] for i in q.index]
    q["annotated_evidence_complete"] = [queries[i]["fully_verified"] for i in q.index]
    q["all_candidates_complete"] = q.missing_rows.eq(0)
    q["has_annotated_evidence"] = q.gold_count.gt(0)
    assert len(q) == 19998 and (q.original_rows == 100).all()
    assert q.verified_rows.sum() == 1972857 and q.missing_rows.sum() == 26943
    assert q.missing_gold.sum() == 227
    assert q.annotated_evidence_complete.sum() == 19859
    assert (q.original_gold_rows == q.gold_count).all()
    assert q.loc[q.all_candidates_complete, "annotated_evidence_complete"].all()
    assert q.loc[q.annotated_evidence_complete, "missing_gold"].eq(0).all()
    assert all(q.loc[i, "verified_rows"] == queries[i]["present_candidates"] for i in q.index)
    q.to_csv(out / "query_availability.csv")
    prov = pd.concat(provenance).groupby(level=[0, 1, 2]).sum().reset_index()
    prov["share_within_verdict_membership"] = prov.rows / prov.groupby(["claim_verdict", "gold_support"]).rows.transform("sum")
    prov.to_csv(out / "candidate_provenance.csv", index=False)
    by_verdict = []
    for verdict, s in q.groupby("claim_verdict"):
        by_verdict.append({"claim_verdict": verdict, "queries": len(s), "original_rows": int(s.original_rows.sum()), "verified_rows": int(s.verified_rows.sum()), "missing_rows": int(s.missing_rows.sum()), "missing_row_fraction": s.missing_rows.sum()/s.original_rows.sum(), "missing_gold": int(s.missing_gold.sum()), "missing_other": int(s.missing_other.sum()), "queries_with_any_missing": int(s.missing_rows.gt(0).sum()), "all_candidates_complete": int(s.all_candidates_complete.sum()), "all_candidates_complete_fraction": s.all_candidates_complete.mean(), "annotated_evidence_complete": int(s.annotated_evidence_complete.sum()), "queries_with_annotated_evidence": int(s.has_annotated_evidence.sum())})
    pd.DataFrame(by_verdict).to_csv(out / "missingness_by_verdict.csv", index=False)
    q.groupby(["claim_verdict", "missing_rows"]).size().rename("queries").to_csv(out / "missing_rows_histogram.csv")
    sets = {"all": set(q.index), "gold_complete": set(q.index[q.annotated_evidence_complete]), "all_candidates_complete": set(q.index[q.all_candidates_complete]), "candidate_incomplete": set(q.index[~q.all_candidates_complete]), "candidate_incomplete_gold_complete": set(q.index[(~q.all_candidates_complete) & q.annotated_evidence_complete])}
    composition = []
    for name, ids in sets.items():
        s = q.loc[sorted(ids)]
        for verdict, v in s.groupby("claim_verdict"):
            composition.append({"subset": name, "claim_verdict": verdict, "queries": len(v), "subset_queries": len(s), "query_fraction": len(v)/len(s), "annotated_occurrences": int(v.gold_count.sum()), "original_rows": int(v.original_rows.sum()), "missing_rows": int(v.missing_rows.sum())})
    pd.DataFrame(composition).to_csv(out / "subset_composition.csv", index=False)
    return queries, q, sets


def saved_outcomes():
    return {seed: {name: pd.read_csv(EXP / f"query_seed{seed}_{name}.csv", dtype={"query_id": str}) for name in METHODS} for seed in SEEDS}


def selected_frame(ids):
    columns = ["query_id", "query_text", "candidate_id", "evidence_text", "gold_evidence", "tie_key"] + VIEWS
    return pd.read_parquet(SCORED, columns=columns, filters=[("query_id", "in", sorted(ids))]).reset_index(drop=True)


def exact_rule(frame, score, count):
    rank = ranks(frame, score)
    eligible = np.flatnonzero(rank <= 5)
    order = eligible[np.lexsort((frame.tie_key.to_numpy(str)[eligible], -score[eligible]))]
    boundary = order[count-1]
    return {"threshold": float(score[boundary]), "tie_cutoff": str(frame.iloc[boundary].tie_key)}, order


def smoke(out, queries, q, sets, saved):
    seed = 13
    model = json.loads((EXP / "model_seed13.json").read_text())
    evaluated = set(model["evaluation_query_ids"])
    smoke_ids = []
    for complete in (True, False):
        for verdict in sorted(q.claim_verdict.unique()):
            eligible = q.index[(q.all_candidates_complete == complete) & (q.claim_verdict == verdict)]
            order = sorted(set(eligible) & evaluated, key=lambda x: hashlib.sha256(("availability-smoke:"+x).encode()).hexdigest())
            smoke_ids.extend(order[:2])
    missing_gold = sorted(set(q.index[q.missing_gold.gt(0)]) & evaluated)
    smoke_ids = sorted(set(smoke_ids + missing_gold[:2]))
    frame = selected_frame(smoke_ids)
    qs = {i: queries[i] for i in smoke_ids}
    scores, stress, _ = score_pipeline(frame, model)
    checks = []
    for name in METHODS:
        actual = query_statistics(frame, scores[name], model["thresholds"][name], stress, qs).set_index("query_id")
        expected = saved[seed][name].set_index("query_id").loc[actual.index]
        for col in ["admitted", "correct", "covered", "all_facts", "stress_admitted", "stress_correct", "gold_count", "verified_gold_count"]:
            assert np.array_equal(actual[col], expected[col]), (name, col)
        rule, order = exact_rule(frame, frame["raw_ce"].to_numpy(float) if name == "ce" else scores[name], len(qs))
        stat = query_statistics(frame, frame["raw_ce"].to_numpy(float) if name == "ce" else scores[name], rule, stress, qs)
        assert stat.admitted.sum() == len(qs)
        checks.append({"method": name, "queries": len(qs), "verified_pairs": len(frame), "saved_frozen_outcomes_match": True, "fixed_budget_items": int(stat.admitted.sum())})
    summary = {s: {n: summarize(v) for n, v in ms.items()} for s, ms in saved.items()}
    delta = np.mean([v["scet"]["evidence_precision"]-v["ce"]["evidence_precision"] for v in summary.values()])
    reference = json.loads((EXP / "primary_comparison.json").read_text())
    assert abs(delta-reference["delta_precision"]) < 1e-14
    write_json(out / "smoke_checks.json", {"status": "passed", "availability_defined_queries": len(q), "all_candidates_complete_queries": len(sets["all_candidates_complete"]), "sample_ids": smoke_ids, "checks": checks, "reproduced_primary_delta": delta, "reference_primary_delta": reference["delta_precision"]})
    return delta


def prospective(out, sets, saved):
    rows, verdict_rows, denominators, stats = [], [], [], {}
    for subset, ids in sets.items():
        stats[subset] = {}
        for seed, methods in saved.items():
            stats[subset][seed] = {}
            for name, value in methods.items():
                s = value[value.query_id.isin(ids)].copy()
                stats[subset][seed][name] = s
                rows.append({"subset": subset, "seed": seed, "method": name, **summarize(s)})
                for verdict, group in s.groupby("claim_verdict"):
                    verdict_rows.append({"subset": subset, "seed": seed, "method": name, "claim_verdict": verdict, **summarize(group)})
                if name == "scet":
                    denominators.append({"subset": subset, "seed": seed, "queries": len(s), "queries_with_annotated_evidence": int(s.gold_count.gt(0).sum()), "queries_without_annotated_evidence": int(s.gold_count.eq(0).sum()), "original_annotated_occurrences": int(s.gold_count.sum()), "verified_annotated_occurrences": int(s.verified_gold_count.sum())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "prospective_per_partition.csv", index=False)
    aggregate(table, ["subset", "method"]).to_csv(out / "prospective_mean_sd.csv", index=False)
    verdict_table = pd.DataFrame(verdict_rows)
    verdict_table.to_csv(out / "prospective_verdict_per_partition.csv", index=False)
    aggregate(verdict_table, ["subset", "method", "claim_verdict"]).to_csv(out / "prospective_verdict_mean_sd.csv", index=False)
    pd.DataFrame(denominators).to_csv(out / "evaluation_denominators.csv", index=False)
    return stats


def aggregate(table, keys):
    numeric = [x for x in table.select_dtypes(include="number").columns if x not in keys + ["seed"]]
    grouped = table.groupby(keys)[numeric].agg(["mean", "std"])
    grouped.columns = ["_".join(x) for x in grouped.columns]
    return grouped.reset_index()


def paired_frozen(stats, repetitions=1000):
    rows = []
    for subset, partitions in stats.items():
        ids = sorted(set().union(*(set(ms["scet"].query_id) for ms in partitions.values())))
        index = {q: i for i, q in enumerate(ids)}
        rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
        boot = {method: [] for method in METHODS if method != "scet"}
        for start in range(0, repetitions, 32):
            weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=min(32, repetitions-start)).astype(np.float64)
            diffs = {method: [] for method in boot}
            for seed, methods in partitions.items():
                a = methods["scet"].set_index("query_id")
                w = weights[:, [index[q] for q in a.index]]
                aa = w @ a[["correct", "admitted"]].to_numpy(float)
                for name in boot:
                    b = methods[name].set_index("query_id").loc[a.index]
                    bb = w @ b[["correct", "admitted"]].to_numpy(float)
                    diffs[name].append(aa[:, 0]/np.maximum(aa[:, 1], 1)-bb[:, 0]/np.maximum(bb[:, 1], 1))
            for name in boot:
                boot[name].extend(np.mean(diffs[name], axis=0).tolist())
        for name, values in boot.items():
            point = np.mean([summarize(ms["scet"])["evidence_precision"]-summarize(ms[name])["evidence_precision"] for ms in partitions.values()])
            low, high = np.quantile(values, [.025, .975])
            rows.append({"subset": subset, "comparator": name, "delta_precision": point, "ci_low": low, "ci_high": high, "unique_query_clusters": len(ids), "repetitions": repetitions, "bootstrap_scope": "frozen prospective admission; shared query weights across overlapping partitions"})
    return pd.DataFrame(rows)


def fixed_budget(out, queries, strict_ids, saved):
    frame = selected_frame(strict_ids)
    records, verdict_rows, orders = [], [], {}
    for seed in SEEDS:
        model = json.loads((EXP / f"model_seed{seed}.json").read_text())
        ids = sorted(strict_ids & set(model["evaluation_query_ids"]))
        test = frame[frame.query_id.isin(ids)].copy().reset_index(drop=True)
        qs = {q: queries[q] for q in ids}
        scores, stress, _ = score_pipeline(test, model)
        orders[seed] = {}
        for name in METHODS:
            values = scores["scet"] if name == "scet" else test["raw_"+name].to_numpy(float)
            rule, order = exact_rule(test, values, len(ids))
            stat = query_statistics(test, values, rule, stress, qs)
            assert stat.admitted.sum() == len(ids)
            stat.to_csv(out / f"fixed_query_seed{seed}_{name}.csv", index=False)
            records.append({"subset": "all_candidates_complete", "seed": seed, "method": name, **summarize(stat)})
            for verdict, group in stat.groupby("claim_verdict"):
                verdict_rows.append({"subset": "all_candidates_complete", "seed": seed, "method": name, "claim_verdict": verdict, **summarize(group)})
            orders[seed][name] = (test.query_id.to_numpy(str)[order], test.gold_evidence.to_numpy(int)[order])
        print(f"Fixed-budget frozen score application: seed {seed}, {len(ids)} queries", flush=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "fixed_budget_per_partition.csv", index=False)
    aggregate(table, ["subset", "method"]).to_csv(out / "fixed_budget_mean_sd.csv", index=False)
    verdict_table = pd.DataFrame(verdict_rows)
    verdict_table.to_csv(out / "fixed_budget_verdict_per_partition.csv", index=False)
    aggregate(verdict_table, ["subset", "method", "claim_verdict"]).to_csv(out / "fixed_budget_verdict_mean_sd.csv", index=False)
    # Reallocate the exact budget in each bootstrap replicate, preserving query
    # multiplicity and the common query draw across overlapping partitions.
    union = sorted(set().union(*(set(d["scet"][0]) for d in orders.values())))
    lookup = {q: i for i, q in enumerate(union)}
    prepared = {}
    for seed, methods in orders.items():
        prepared[seed] = {name: (np.array([lookup[q] for q in pairs[0]]), pairs[1]) for name, pairs in methods.items()}
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    distributions = {name: [] for name in METHODS if name != "scet"}
    for _ in range(PROTOCOL["bootstrap_repetitions"]):
        w = rng.multinomial(len(union), np.full(len(union), 1/len(union)))
        differences = {name: [] for name in distributions}
        for seed, methods in prepared.items():
            budget = int(w[np.unique(methods["scet"][0])].sum())
            precision = {}
            for name, (owners, gold) in methods.items():
                multiplicity = w[owners]
                cumulative = np.cumsum(multiplicity)
                boundary = int(np.searchsorted(cumulative, budget, side="left"))
                before = int(cumulative[boundary-1]) if boundary else 0
                correct = np.dot(multiplicity[:boundary], gold[:boundary]) + (budget-before)*gold[boundary]
                precision[name] = correct/budget
            for name in distributions:
                differences[name].append(precision["scet"]-precision[name])
        for name in distributions:
            distributions[name].append(float(np.mean(differences[name])))
    rows = []
    means = table.groupby("method").evidence_precision.mean()
    for name, values in distributions.items():
        low, high = np.quantile(values, [.025, .975])
        rows.append({"subset": "all_candidates_complete", "comparator": name, "delta_precision": means["scet"]-means[name], "ci_low": low, "ci_high": high, "unique_query_clusters": len(union), "repetitions": PROTOCOL["bootstrap_repetitions"], "bootstrap_scope": "exact workload reallocated within every query-cluster resample; frozen fitted score forms"})
    pd.DataFrame(rows).to_csv(out / "fixed_budget_intervals.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    out = OUT / args.mode
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    queries, availability_table, sets = availability(out)
    saved = saved_outcomes()
    if args.mode == "smoke":
        smoke(out, queries, availability_table, sets, saved)
    else:
        check = json.loads((OUT / "smoke/smoke_checks.json").read_text(encoding="utf-8"))
        assert check["status"] == "passed"
        statistics = prospective(out, sets, saved)
        intervals = paired_frozen(statistics, PROTOCOL["bootstrap_repetitions"])
        reference = json.loads((EXP / "primary_comparison.json").read_text())
        main_comparison = intervals[(intervals.subset == "all") & (intervals.comparator == "ce")].iloc[0]
        assert np.allclose([main_comparison.delta_precision, main_comparison.ci_low, main_comparison.ci_high], [reference["delta_precision"], reference["ci_low"], reference["ci_high"]], atol=1e-12)
        intervals.to_csv(out / "prospective_intervals.csv", index=False)
        fixed_budget(out, queries, sets["all_candidates_complete"], saved)
    write_json(out / "analysis_manifest.json", {"status": "complete", "mode": args.mode, "seconds": time.monotonic()-start, "selection": "All subsets fixed by original candidate identities and snapshot text availability, before outcome comparison.", "subsets": {name: len(ids) for name, ids in sets.items()}, "gold_complete_definition": "Every original annotated sentence identity has verified text, including claims with no annotated evidence.", "all_candidates_complete_definition": "All 100 original candidate identities have verified June 2017 snapshot text; this also implies all annotated sentences have verified text in this controlled pool.", "prospective": "Original fitting, calibration, scoring form, thresholds, tie keys and held-out outcomes remain frozen. Metrics are subset-restricted only.", "fixed_budget": "Original fitting and score forms remain frozen. Exactly one admitted item per subset held-out query on average, maximum five per query; workload is allocated within each subset. Individual views use raw scores.", "recall_denominator": "Original annotated sentences per query, averaged over queries with annotated evidence; set recall requires one complete original evidence group.", "query_counts": "Four overlapping outer partitions; per-partition counts and their mean are reported separately from distinct source queries.", "bootstrap": "1000 paired query-cluster draws with common query weights across overlapping partitions. Prospective decisions frozen; fixed-budget allocation repeated in each resample.", "new_model_inference": False, "new_model_fitting": False})
    print(json.dumps({"mode": args.mode, "status": "complete", "seconds": round(time.monotonic()-start, 2), "subsets": {name: len(ids) for name, ids in sets.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
