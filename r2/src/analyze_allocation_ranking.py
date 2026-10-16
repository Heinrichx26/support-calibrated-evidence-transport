"""Frozen 2 x 2 decomposition of per-query sentence counts and sentence ranks.

Run --smoke first. No fitting, neural inference, threshold selection, or changes
to submission files occur. All outputs remain in results/allocation_ranking/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import r2_admission as admission


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/allocation_ranking"
SEEDS = [13, 17, 29, 7]
DATASETS = ["fever", "hover"]
CAP = 5
CELLS = {
    "ce_rank_ce_count": ("ce", "ce"),
    "scet_rank_ce_count": ("scet", "ce"),
    "ce_rank_scet_count": ("ce", "scet"),
    "scet_rank_scet_count": ("scet", "scet"),
}
NAMES = list(CELLS)
CONTRASTS = {
    "net_scet_minus_ce": [-1, 0, 0, 1],
    "symmetric_ranking": [-0.5, 0.5, -0.5, 0.5],
    "symmetric_count_allocation": [-0.5, -0.5, 0.5, 0.5],
    "ranking_with_ce_counts": [-1, 1, 0, 0],
    "ranking_with_scet_counts": [0, 0, -1, 1],
    "allocation_with_ce_ranks": [-1, 0, 1, 0],
    "allocation_with_scet_ranks": [0, -1, 0, 1],
    "interaction": [1, -1, -1, 1],
}
METRIC_GROUPS = [
    ("precision", "all"),
    ("evidence_recall", "all"),
    ("set_recall", "all"),
    ("complete_per100", "all"),
    ("coverage", "all"),
    ("no_evidence_admission", "all"),
    ("no_evidence_items_per100", "all"),
    ("precision", "singleton"),
    ("precision", "multiple"),
    ("set_recall", "singleton"),
    ("set_recall", "multiple"),
]
PLAN = {
    "analysis": "Frozen 2 x 2 crossing of sentence ranks and per-query admitted counts",
    "datasets": DATASETS,
    "partitions": SEEDS,
    "rank_methods": ["raw cross-encoder", "frozen selected SCET scorer"],
    "count_methods": ["saved equal-budget cross-encoder counts", "saved equal-budget SCET counts"],
    "maximum_rank": CAP,
    "full_budget": "Each cell admits exactly one sentence per held-out query on average",
    "smoke": "First 20 lexicographically sorted FEVER seed-13 evaluation IDs; inherit saved full-batch counts without reallocating within this subset",
    "ties": "Original fixed candidate tie keys",
    "bootstrap_repetitions": 1000,
    "bootstrap_seed": 2026101603,
    "resampling": "Paired query-cluster multinomial weights shared wherever a query occurs across the four overlapping partitions; frozen cell decisions",
    "intervals": "Pointwise 95% percentile intervals, conditional on frozen models, counts, ranks, and observed partitions",
    "decomposition": "Average the two conditional ranking effects and the two conditional count-allocation effects; their sum equals the net SCET-minus-cross-encoder difference",
    "interpretation": "Descriptive counterfactual decomposition; hybrid cells are diagnostic configurations, with no new model or operating policy selected",
    "new_fitting": False,
    "new_neural_inference": False,
    "new_threshold_selection": False,
    "metric_groups": METRIC_GROUPS,
}
COLUMNS = ["query_id", "candidate_id", "tie_key", "query_text", "evidence_text", "gold_evidence"] + admission.VIEWS


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paths(dataset: str, seed: int):
    return {
        "scores": ROOT / f"data/scored/{dataset}_full_k100.parquet",
        "queries": ROOT / f"data/processed/{dataset}_full_queries.json",
        "model": ROOT / f"results/experiments/{dataset}_full_k100/model_seed{seed}.json",
        "scet": ROOT / f"results/strengthening/{dataset}_full_k100/query_seed{seed}_scet.csv",
        "ce": ROOT / f"results/strengthening/{dataset}_full_k100/query_seed{seed}_ce.csv",
    }


def load_inputs(dataset: str, seed: int, smoke: bool):
    source = paths(dataset, seed)
    model = read_json(source["model"])
    ids = sorted(map(str, model["evaluation_query_ids"]))
    if smoke:
        ids = ids[:20]
    queries = read_json(source["queries"])
    saved = {}
    for method in ["ce", "scet"]:
        s = pd.read_csv(source[method], dtype={"query_id": str}).set_index("query_id")
        assert not s.index.duplicated().any()
        s = s.loc[ids]
        assert s.admitted.between(0, CAP).all()
        if not smoke:
            assert int(s.admitted.sum()) == len(ids)
        saved[method] = s
    return source, model, ids, queries, saved


def merge_top(frame: pd.DataFrame, scores: dict, top: dict):
    """Retain only five candidates per query and scorer while streaming."""
    qids = frame.query_id.to_numpy(str)
    ties = frame.tie_key.to_numpy(str)
    cids = frame.candidate_id.to_numpy(str)
    labels = frame.gold_evidence.to_numpy(int)
    for method in ["ce", "scet"]:
        score = np.asarray(scores[method], float)
        assert np.isfinite(score).all()
        order = np.lexsort((ties, -score, qids))
        ordered_q = qids[order]
        starts = np.maximum.accumulate(np.where(np.r_[True, ordered_q[1:] != ordered_q[:-1]], np.arange(len(order)), 0))
        eligible = order[np.arange(len(order)) - starts < CAP]
        for i in eligible:
            record = (-float(score[i]), str(ties[i]), str(cids[i]), int(labels[i]))
            top[method].setdefault(str(qids[i]), []).append(record)
        for qid in np.unique(qids[eligible]):
            top[method][qid] = heapq.nsmallest(CAP, top[method][qid])


def frame_top(frame: pd.DataFrame, model: dict):
    frozen, _, _ = admission.score_pipeline(frame, model)
    scores = {"scet": frozen["scet"], "ce": frame.raw_ce.to_numpy(float)}
    top = {"ce": {}, "scet": {}}
    merge_top(frame, scores, top)
    return top, scores


def streamed_top(path: Path, ids: list[str], model: dict, smoke: bool):
    top = {"ce": {}, "scet": {}}
    values = pa.array(ids, type=pa.string())
    counts = {q: 0 for q in ids}
    smoke_frames = []
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=COLUMNS, use_threads=False):
        keep = pc.is_in(batch.column(batch.schema.get_field_index("query_id")), value_set=values)
        selected = pa.Table.from_batches([batch]).filter(keep)
        if not len(selected):
            continue
        frame = selected.to_pandas().reset_index(drop=True)
        frame["query_id"] = frame.query_id.astype(str)
        frozen, _, _ = admission.score_pipeline(frame, model)
        scores = {"scet": frozen["scet"], "ce": frame.raw_ce.to_numpy(float)}
        merge_top(frame, scores, top)
        for qid, n in frame.groupby("query_id", sort=False).size().items():
            counts[qid] += int(n)
        rows += len(frame)
        if smoke:
            smoke_frames.append(frame)
        del frame, selected, frozen, scores
    assert set(top["ce"]) == set(ids) == set(top["scet"])
    assert all(n > 0 for n in counts.values())
    return top, counts, rows, pd.concat(smoke_frames, ignore_index=True) if smoke else None


def selections(top: dict, saved: dict, ids: list[str]):
    choices = {}
    for cell, (rank_method, count_method) in CELLS.items():
        choices[cell] = {}
        for qid in ids:
            k = int(saved[count_method].loc[qid, "admitted"])
            assert 0 <= k <= min(CAP, len(top[rank_method][qid]))
            choices[cell][qid] = tuple(x[2] for x in top[rank_method][qid][:k])
    return choices


def statistics(dataset, seed, ids, queries, saved, top, smoke):
    rows = []
    choices = selections(top, saved, ids)
    for cell, (rank_method, count_method) in CELLS.items():
        for qid in ids:
            q = queries[qid]
            records = top[rank_method][qid][:int(saved[count_method].loc[qid, "admitted"])]
            admitted_ids = set(choices[cell][qid])
            sets = [set(s) for s in q["gold_sets"] if s]
            smallest = min(map(len, sets)) if sets else 0
            gold = set(q["gold_ids"])
            assert bool(smallest) == bool(q["gold_count"])
            assert all(int(cid in gold) == label for _, _, cid, label in records)
            rows.append({
                "dataset": dataset, "seed": seed, "cell": cell,
                "rank_method": rank_method, "count_method": count_method,
                "query_id": qid, "claim_verdict": q["claim_verdict"],
                "gold_count": int(q["gold_count"]), "minimum_set_size": smallest,
                "group": "no_evidence" if not smallest else ("singleton" if smallest == 1 else "multiple"),
                "admitted": len(records), "correct": sum(x[3] for x in records),
                "all_facts": int(any(s <= admitted_ids for s in sets)),
                "covered": int(bool(records)),
            })
    detail = pd.DataFrame(rows)
    checks = []
    for method, cell in [("ce", "ce_rank_ce_count"), ("scet", "scet_rank_scet_count")]:
        current = detail[detail.cell == cell].set_index("query_id").loc[ids]
        for column in ["admitted", "correct", "all_facts", "covered", "gold_count"]:
            np.testing.assert_array_equal(current[column], saved[method].loc[ids, column], err_msg=f"{dataset}:{seed}:{method}:{column}")
        checks.append({"dataset": dataset, "seed": seed, "method": method,
                       "queries_checked": len(ids), "columns": ["admitted", "correct", "all_facts", "covered", "gold_count"],
                       "exact_match": True})
    budget = detail.groupby("cell").admitted.sum()
    if not smoke:
        assert budget.eq(len(ids)).all(), budget.to_dict()
    for count_method in ["ce", "scet"]:
        a = detail[detail.cell == f"ce_rank_{count_method}_count"].set_index("query_id")
        b = detail[detail.cell == f"scet_rank_{count_method}_count"].set_index("query_id").loc[a.index]
        np.testing.assert_array_equal(a.admitted, b.admitted)
        np.testing.assert_array_equal(a.covered, b.covered)
    return detail, checks, choices


def parts(s: pd.DataFrame, metric: str, group: str):
    keep = np.ones(len(s), float) if group == "all" else s.group.eq(group).to_numpy(float)
    positive = s.gold_count.to_numpy() > 0
    correct = s.correct.to_numpy(float)
    ones = np.ones(len(s), float)
    if metric == "precision":
        return correct * keep, s.admitted.to_numpy(float) * keep
    if metric == "evidence_recall":
        recall = np.divide(correct, s.gold_count.to_numpy(float), out=np.zeros(len(s)), where=positive)
        return recall * keep, positive * keep
    if metric == "set_recall":
        return s.all_facts.to_numpy(float) * keep, positive * keep
    if metric == "complete_per100":
        return 100 * s.all_facts.to_numpy(float) * keep, ones
    if metric == "coverage":
        return s.covered.to_numpy(float) * keep, keep
    if metric == "no_evidence_admission":
        return s.covered.to_numpy(float) * (~positive) * keep, (~positive) * keep
    if metric == "no_evidence_items_per100":
        return 100 * s.admitted.to_numpy(float) * (~positive) * keep, ones
    raise ValueError(metric)


def summarize(detail: pd.DataFrame):
    rows = []
    for (dataset, seed, cell), s in detail.groupby(["dataset", "seed", "cell"], sort=False):
        row = {"dataset": dataset, "seed": int(seed), "cell": cell,
               "queries": len(s), "admitted": int(s.admitted.sum()), "correct": int(s.correct.sum()),
               "complete_sets": int(s.all_facts.sum()), "count_max": int(s.admitted.max())}
        for metric, group in METRIC_GROUPS:
            n, d = parts(s, metric, group)
            key = metric if group == "all" else f"{group}_{metric}"
            row[key] = float(n.sum() / d.sum()) if d.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def invariance(frame, model, saved, ids, baseline_top):
    original, original_scores = frame_top(frame, model)
    original_selection = selections(original, saved, ids)
    assert original_selection == selections(baseline_top, saved, ids)
    flipped = frame.copy()
    flipped["gold_evidence"] = 1 - flipped.gold_evidence
    flip_top, flip_scores = frame_top(flipped, model)
    assert selections(flip_top, saved, ids) == original_selection
    for method in ["scet", "ce"]:
        np.testing.assert_allclose(flip_scores[method], original_scores[method], rtol=0, atol=0)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    reverse_top, reverse_scores = frame_top(reversed_frame, model)
    assert selections(reverse_top, saved, ids) == original_selection
    for method in ["scet", "ce"]:
        np.testing.assert_allclose(reverse_scores[method][::-1], original_scores[method], rtol=0, atol=1e-12)
    return {"streaming_matches_single_frame": True, "label_flip_selection_invariant": True,
            "order_reversal_selection_invariant": True, "frozen_scores_label_invariant": True}


def paired_bootstrap(detail, dataset, out):
    s = detail[detail.dataset == dataset]
    ids = sorted(s.query_id.unique())
    lookup = {qid: i for i, qid in enumerate(ids)}
    groups = {(seed, cell): s[(s.seed == seed) & (s.cell == cell)].copy() for seed in SEEDS for cell in NAMES}
    point_rows, interval_rows = [], []
    # Reinitialize the fixed RNG for each metric so all contrasts and metrics
    # share identical query-cluster resamples while limiting working memory.
    for metric, stratum in METRIC_GROUPS:
        matrix = np.zeros((len(ids), len(SEEDS) * len(NAMES) * 2), dtype=np.float64)
        for si, seed in enumerate(SEEDS):
            for ci, cell in enumerate(NAMES):
                q = groups[(seed, cell)]
                ix = np.array([lookup[qid] for qid in q.query_id], dtype=np.int64)
                numerator, denominator = parts(q, metric, stratum)
                column = (si * len(NAMES) + ci) * 2
                matrix[ix, column] = numerator
                matrix[ix, column+1] = denominator
        sums = matrix.sum(axis=0).reshape(len(SEEDS), len(NAMES), 2)
        if not sums[:, :, 1].any():
            point_rows.append({"dataset": dataset, "metric": metric, "group": stratum, "unavailable": True,
                               "reason": "No queries in the metric denominator"})
            continue
        points = np.divide(sums[:, :, 0], sums[:, :, 1], out=np.full(sums.shape[:2], np.nan), where=sums[:, :, 1] > 0)
        mean = np.mean(points, axis=0)
        rng = np.random.default_rng(PLAN["bootstrap_seed"])
        boot = np.empty((PLAN["bootstrap_repetitions"], len(NAMES)), dtype=float)
        for start in range(0, len(boot), 50):
            n = min(50, len(boot)-start)
            weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=n).astype(np.float64)
            totals = (weights @ matrix).reshape(n, len(SEEDS), len(NAMES), 2)
            ratios = np.divide(totals[:, :, :, 0], totals[:, :, :, 1], out=np.full(totals.shape[:3], np.nan), where=totals[:, :, :, 1] > 0)
            boot[start:start+n] = np.mean(ratios, axis=1)
        for ci, cell in enumerate(NAMES):
            low, high = np.nanquantile(boot[:, ci], [.025, .975])
            point_rows.append({"dataset": dataset, "metric": metric, "group": stratum, "cell": cell,
                               "mean": float(mean[ci]), "sample_sd": float(np.std(points[:, ci], ddof=1)),
                               "ci95_low": float(low), "ci95_high": float(high), "unavailable": False})
        decomposed = {}
        for name, coefficients in CONTRASTS.items():
            coeff = np.array(coefficients, float)
            draw = boot @ coeff
            point = float(mean @ coeff)
            low, high = np.nanquantile(draw, [.025, .975])
            interval_rows.append({"dataset": dataset, "metric": metric, "group": stratum, "contrast": name,
                                  "difference": point, "ci95_low": float(low), "ci95_high": float(high),
                                  "unique_query_clusters": len(ids), "bootstrap_repetitions": len(boot),
                                  "valid_draws": int(np.isfinite(draw).sum()),
                                  "interval_scope": "Pointwise; frozen cell decisions; shared query weights across overlapping partitions"})
            decomposed[name] = point
        assert abs(decomposed["symmetric_ranking"] + decomposed["symmetric_count_allocation"] - decomposed["net_scet_minus_ce"]) < 1e-12
        if metric in ["coverage", "no_evidence_admission", "no_evidence_items_per100"]:
            assert abs(decomposed["symmetric_ranking"]) < 1e-12
        print(dataset, metric, stratum, "bootstrap complete", flush=True)
        del matrix, weights, totals, ratios, boot
    cells, contrasts = pd.DataFrame(point_rows), pd.DataFrame(interval_rows)
    cells.to_csv(out / f"{dataset}_cell_intervals.csv", index=False)
    contrasts.to_csv(out / f"{dataset}_decomposition.csv", index=False)
    return cells, contrasts


def render_report(summary, cells, contrasts, audits, out, smoke):
    lines = ["# Frozen allocation and ranking decomposition", "", "No models, transforms, thresholds, or per-query counts were fitted or selected in this analysis. The four cells cross the saved sentence ranks and saved per-query admitted counts. The cross-encoder uses raw scores and every rank uses the original fixed tie keys.", ""]
    if smoke:
        lines += ["This smoke uses the first 20 lexicographically sorted FEVER seed-13 evaluation IDs. Counts are inherited from the complete held-out batch. The two count policies can therefore have different sentence totals within this small subset. Full-run cells use an identical total sentence budget.", ""]
    lines += ["| Dataset | Seed | Rank | Count allocation | Queries | Sentences | Correct | Complete sets | Precision | Set recall |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        rank, count = CELLS[row.cell]
        lines.append(f"| {row.dataset} | {row.seed} | {rank} | {count} | {row.queries} | {row.admitted} | {row.correct} | {row.complete_sets} | {row.precision:.5f} | {row.set_recall:.5f} |")
    lines += ["", "All original diagonals reproduce admitted counts, correct counts, complete-set outcomes, coverage, and original evidence counts exactly, for every evaluated query. Sorting and label invariance are checked in the smoke. Every full-run cell retains the original query denominators.", ""]
    if not smoke:
        lines += ["| Dataset | Metric | Group | Net SCET minus cross-encoder | Ranking contribution | Count-allocation contribution |", "|---|---|---|---:|---:|---:|"]
        for dataset in DATASETS:
            for metric, group in METRIC_GROUPS:
                a = contrasts[(contrasts.dataset == dataset) & (contrasts.metric == metric) & (contrasts.group == group)].set_index("contrast")
                if a.empty:
                    continue
                def ci(name):
                    r = a.loc[name]
                    return f"{r.difference:.5f} [{r.ci95_low:.5f}, {r.ci95_high:.5f}]"
                lines.append(f"| {dataset} | {metric} | {group} | {ci('net_scet_minus_ce')} | {ci('symmetric_ranking')} | {ci('symmetric_count_allocation')} |")
        lines += ["", "Ranking contribution averages the ranking change under each fixed count allocation. Count-allocation contribution averages the count change under each fixed ranking. They sum exactly to the original net difference. Conditional effects and their interaction are retained in the decomposition CSV files. Negative contributions and intervals spanning zero are reported without changing the scorer or analysis specification.", "", "Intervals use 1,000 paired query-cluster resamples with a common weight for each query wherever it occurs across the four overlapping partitions. Decisions remain frozen inside resamples. Intervals are pointwise and conditional on the observed partitions. The diagnostic hybrid cells do not define newly selected deployment methods.", "", "Coverage, admission on queries without annotated evidence, and the number of items assigned to those queries depend entirely on the count allocation. HoVer contains no query without annotated evidence, so its corresponding admission proportion is unavailable."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(smoke=False):
    started = time.monotonic()
    mode = "smoke" if smoke else "full"
    out = OUT / mode
    out.mkdir(parents=True, exist_ok=True)
    code_hash = sha(Path(__file__))
    if not smoke:
        gate = read_json(OUT / "smoke/audit.json")
        assert gate["passed"] and gate["code_sha256"] == code_hash
        assert gate["nontrivial_count_or_rank_contrast"]
    details, checks, audits, source_hashes = [], [], [], {}
    for dataset in ["fever"] if smoke else DATASETS:
        for seed in [13] if smoke else SEEDS:
            source, model, ids, queries, saved = load_inputs(dataset, seed, smoke)
            top, counts, rows, smoke_frame = streamed_top(source["scores"], ids, model, smoke)
            detail, diagonal_checks, selected = statistics(dataset, seed, ids, queries, saved, top, smoke)
            invariance_checks = invariance(smoke_frame, model, saved, ids, top) if smoke else {}
            different_counts = int(saved["ce"].admitted.ne(saved["scet"].admitted).sum())
            different_ranking_choices = sum(selected["ce_rank_ce_count"][q] != selected["scet_rank_ce_count"][q] for q in ids)
            different_ranking_choices_scet = sum(selected["ce_rank_scet_count"][q] != selected["scet_rank_scet_count"][q] for q in ids)
            if dataset == "hover" and seed == 17:
                assert model["selected_form"] == "ce" and model["penalty"] == 0
                assert different_counts == 0
                # Frozen empirical score transforms are monotonic and can
                # create flat tails. Check metric equivalence without assuming
                # that raw and transformed scores resolve every tie identically.
                base = detail[detail.cell == NAMES[0]].set_index("query_id").loc[ids]
                for cell in NAMES[1:]:
                    other = detail[detail.cell == cell].set_index("query_id").loc[ids]
                    np.testing.assert_array_equal(base[["admitted", "correct", "all_facts"]],
                                                  other[["admitted", "correct", "all_facts"]])
            audits.append({"dataset": dataset, "seed": seed, "queries": len(ids), "candidate_rows": rows,
                           "selected_form": model["selected_form"], "selected_penalty": model["penalty"],
                           "queries_with_different_counts": different_counts,
                           "queries_with_different_rank_choices_ce_counts": different_ranking_choices,
                           "queries_with_different_rank_choices_scet_counts": different_ranking_choices_scet,
                           "cell_admitted_totals": detail.groupby("cell").admitted.sum().to_dict(),
                           **invariance_checks})
            detail.to_csv(out / f"query_{dataset}_seed{seed}.csv", index=False)
            details.append(detail)
            checks.extend(diagonal_checks)
            for name, path in source.items():
                relative = str(path.relative_to(ROOT))
                if relative not in source_hashes:
                    source_hashes[relative] = sha(path)
            print(dataset, seed, "done", "queries", len(ids), "different counts", different_counts,
                  "elapsed", round(time.monotonic()-started, 1), flush=True)
            del queries, saved, top, selected, detail, smoke_frame, model
            gc.collect()
    detail = pd.concat(details, ignore_index=True)
    del details
    summary = summarize(detail)
    summary.to_csv(out / "by_partition.csv", index=False)
    cells_all, contrasts_all = [], []
    if not smoke:
        for dataset in DATASETS:
            cells, contrasts = paired_bootstrap(detail, dataset, out)
            cells_all.append(cells)
            contrasts_all.append(contrasts)
        cells, contrasts = pd.concat(cells_all, ignore_index=True), pd.concat(contrasts_all, ignore_index=True)
        cells.to_csv(out / "cell_intervals.csv", index=False)
        contrasts.to_csv(out / "decomposition.csv", index=False)
    else:
        cells, contrasts = pd.DataFrame(), pd.DataFrame()
    nontrivial = any(a["queries_with_different_counts"] and (a["queries_with_different_rank_choices_ce_counts"] or a["queries_with_different_rank_choices_scet_counts"]) for a in audits)
    assert nontrivial
    audit = {"passed": True, "mode": mode, "code_sha256": code_hash, "plan": PLAN,
             "source_sha256": source_hashes, "diagonal_checks": checks, "partitions": audits,
             "nontrivial_count_or_rank_contrast": nontrivial,
             "maximum_candidate_batch": 20_000,
             "maximum_bootstrap_batch": 50,
             "elapsed_seconds": time.monotonic()-started}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    render_report(summary, cells, contrasts, audits, out, smoke)
    print(json.dumps({"passed": True, "mode": mode, "elapsed_seconds": round(audit["elapsed_seconds"], 1),
                      "output": str(out), "checks": audits}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.smoke)
