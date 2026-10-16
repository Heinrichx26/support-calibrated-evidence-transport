"""Single-scorer list calibration and top-one abstention controls.

Only frozen raw scores are consumed. Run --smoke before --full. All new outputs
are confined to results/r5_simple_controls. No manuscript/core files are edited.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import r2_admission as admission

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/r5_simple_controls'
SEEDS = [13, 17, 29, 7]
PLAN = {
    'version': '20261016_v1',
    'question': 'Does frozen SCET improve on one raw scorer given simple learned list-based allocation and top-one abstention?',
    'sources': {'fever': ['ce'], 'hover': ['ce', 'e5'], 'scifact': ['ce', 'multivers']},
    'partitions': SEEDS,
    'fitting': 'Original fitting IDs only; three identifier-hashed query folds; no calibration/test labels used for training or choosing C.',
    'C': [0.1, 1.0, 10.0],
    'top1_gate_target': 'Membership label of the raw scorer first-ranked candidate, predicted using query-list score summaries.',
    'top1_C_selection': 'Minimum query-fold out-of-fold Brier score, then smaller C.',
    'list_target': 'Membership of each of the raw scorer first five candidates; query-balanced weights; no class balancing.',
    'list_C_selection': 'Lexicographic OOF precision, annotated-hit query fraction, annotated items/query at 100 items/100 fitting queries; then smaller C.',
    'query_features': ['raw_top1', 'raw_top1_minus_top2', 'raw_top1_minus_top5', 'top5_mean', 'top5_population_sd', 'pool_mean', 'pool_population_sd', 'log1p_pool_count'],
    'list_features': ['raw_candidate_score', 'raw_gap_to_top1', 'rank_indicators_1_to_5', 'query_features'],
    'ordering': 'Raw scorer candidate order with original fixed tie keys is preserved; cap-five predictions use cumulative minima with nextafter for strict prefix ordering.',
    'methods': ['raw_cap5', 'raw_top1', 'list_logit_cap5', 'list_logit_top1', 'frozen_scet'],
    'workload_targets': [25, 50, 100, 200, 300, 400, 500],
    'prospective_thresholds': 'Independent calibration order statistics, scores and existing tie keys; a top-one policy saturates at 100 items/100 queries.',
    'fixed_budget': 'Label-free batch score order admits min(number of queries, eligible candidates); diagnostic only, separately labelled.',
    'risk_targets': [0.01, 0.025, 0.05, 0.1, 0.2, 0.3],
    'risk_thresholds': '101 fitting-prediction quantiles plus +/-infinity fixed before calibration labels. Calibration chooses maximum admitted count subject to empirical query-error or no-evidence-admission target. No finite-sample risk guarantee is claimed for these controls.',
    'risk_metrics': ['query_error_rate', 'false_admission_no_evidence'],
    'reporting': 'All sources, models, Cs, thresholds and specified operating points retained. No test-side model or operating-point choice.',
    'bootstrap': '1000 paired query-cluster draws, shared across overlapping partitions, frozen decisions. Primary prospective workload 100, exact budget, and calibration risk targets 0.1/0.2; pointwise 95% intervals.',
    'smoke': 'FEVER seed13: first 600 fitting, 200 calibration and 300 test IDs; full starts only after finite predictions, disjoint splits, label/order invariance, prefix and independent calibration checks pass.',
    'threads': 2,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compact(frame, score, queries, cap=5):
    """Collect same-scorer raw list features; every input is label-independent."""
    qids = np.asarray(sorted(frame.query_id.unique()), str)
    codes = pd.Categorical(frame.query_id, categories=qids).codes
    order = np.lexsort((frame.tie_key.to_numpy(str), -np.asarray(score), codes))
    ordered_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1])+1]
    sizes = np.diff(np.r_[starts, len(order)])
    ranks = np.arange(len(order))-np.repeat(starts, sizes)+1
    indices = order[ranks <= cap]
    row_codes = codes[indices]
    row_ranks = ranks[ranks <= cap]
    raw = np.asarray(score, float)[indices]
    all_scores = np.asarray(score, float)[order]
    top_values = np.empty((len(qids), 5), float)
    for i, (start, size) in enumerate(zip(starts, sizes)):
        vals = all_scores[start:start+min(size, 5)]
        top_values[i] = np.pad(vals, (0, 5-len(vals)), mode='edge')
    count = sizes.astype(float)
    means = np.add.reduceat(all_scores, starts)/count
    variance = np.maximum(np.add.reduceat(all_scores**2, starts)/count-means**2, 0)
    qx = np.column_stack([top_values[:, 0], top_values[:, 0]-top_values[:, 1],
                          top_values[:, 0]-top_values[:, 4], top_values.mean(axis=1),
                          top_values.std(axis=1), means, np.sqrt(variance), np.log1p(count)])
    x = np.column_stack([raw, top_values[row_codes, 0]-raw,
                         np.eye(5)[row_ranks-1], qx[row_codes]])
    data = {'qids': qids, 'codes': row_codes, 'ranks': row_ranks,
            'cids': frame.candidate_id.to_numpy(str)[indices],
            'ties': frame.tie_key.to_numpy(str)[indices], 'raw': raw,
            'y': frame.gold_evidence.to_numpy(int)[indices], 'X': x,
            'QX': qx, 'queries': queries}
    assert np.isfinite(x).all()
    return data


def take(data, ids):
    ids = np.asarray(sorted(ids), str)
    keep = np.isin(data['qids'][data['codes']], ids)
    old = {q: i for i, q in enumerate(data['qids'])}
    select = [old[q] for q in ids]
    lookup = {old[q]: i for i, q in enumerate(ids)}
    result = {k: data[k][keep] for k in ['ranks', 'cids', 'ties', 'raw', 'y', 'X']}
    result.update(qids=ids, codes=np.asarray([lookup[x] for x in data['codes'][keep]], int),
                  QX=data['QX'][select], queries=data['queries'])
    return result


def limit_top1(data):
    keep = data['ranks'] == 1
    result = dict(data)
    result.update({k: data[k][keep] for k in ['codes', 'ranks', 'cids', 'ties', 'raw', 'y', 'X']})
    return result


def pipeline(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, solver='lbfgs', random_state=13))


def fit_one(data, C, kind, rows=None):
    if rows is None:
        rows = np.ones(len(data['y']), bool)
    if kind == 'top1':
        rows = rows & (data['ranks'] == 1)
        x = data['QX'][data['codes'][rows]]
    else:
        x = data['X'][rows]
    y = data['y'][rows]
    if len(np.unique(y)) == 1:
        return {'constant': float(y.mean())}
    model = pipeline(C)
    counts = np.bincount(data['codes'][rows], minlength=len(data['qids']))
    weight = 1/counts[data['codes'][rows]]
    weight /= weight.mean()
    model.fit(x, y, logisticregression__sample_weight=weight)
    return model


def prefix(data, score):
    score = np.asarray(score, float).copy()
    for rank in range(2, 6):
        ix = np.flatnonzero(data['ranks'] == rank)
        # compact rows are sorted by query and rank, so each predecessor is i-1.
        score[ix] = np.minimum(score[ix], np.nextafter(score[ix-1], -np.inf))
    return score


def predict(model, data, kind):
    if kind == 'top1':
        x = data['QX'][data['codes']]
    else:
        x = data['X']
    if isinstance(model, dict):
        p = np.full(len(x), model['constant'])
    else:
        p = model.predict_proba(x)[:, 1]
    return prefix(data, p) if kind == 'list' else p


def threshold(data, score, target):
    order = np.lexsort((data['ties'], -score))
    wanted = min(len(order), round(len(data['qids'])*target/100))
    if not wanted:
        return {'threshold': float('inf'), 'tie_cutoff': ''}
    i = order[wanted-1]
    return {'threshold': float(score[i]), 'tie_cutoff': str(data['ties'][i])}


def admitted(data, score, rule):
    return (score > rule['threshold']) | ((score == rule['threshold']) & (data['ties'] <= rule['tie_cutoff']))


def statistics(data, mask):
    n = len(data['qids'])
    codes = data['codes']
    issued = np.bincount(codes, weights=mask, minlength=n).astype(int)
    correct = np.bincount(codes, weights=mask*data['y'], minlength=n).astype(int)
    errors = np.bincount(codes, weights=mask*(1-data['y']), minlength=n) > 0
    q = data['queries']
    gold = np.asarray([q[x]['gold_count'] for x in data['qids']])
    if '_groups' not in data:
        lookup = {(int(c), x): i for i, (c, x) in enumerate(zip(codes, data['cids']))}
        groups, owners = [], []
        for c, qid in enumerate(data['qids']):
            for group in q[qid]['gold_sets']:
                if group and all((c, x) in lookup for x in group):
                    groups.append([lookup[(c, x)] for x in set(group)])
                    owners.append(c)
        width = max(map(len, groups), default=0)
        matrix = np.full((len(groups), width), len(mask), int)
        for i, group in enumerate(groups):
            matrix[i, :len(group)] = group
        data['_groups'] = matrix, np.asarray(owners, int)
    matrix, owners = data['_groups']
    complete = np.zeros(n, int)
    if len(matrix):
        # Padding is true while missing candidate groups were excluded above.
        passing = np.r_[mask, True][matrix].all(axis=1)
        complete[np.unique(owners[passing])] = 1
    smallest = [min([len(set(s)) for s in q[x]['gold_sets'] if s], default=0) for x in data['qids']]
    return pd.DataFrame({'query_id': data['qids'], 'gold_count': gold,
                         'minimum_set_size': smallest, 'admitted': issued, 'correct': correct,
                         'covered': issued > 0, 'has_error': errors,
                         'all_facts': complete})


def summary(stats):
    gold = stats.gold_count.to_numpy() > 0
    issued = stats.admitted.sum()
    return {'queries': len(stats), 'evidence_precision': stats.correct.sum()/issued if issued else np.nan,
            'volume_per_100': 100*stats.admitted.mean(), 'annotated_per_100': 100*stats.correct.mean(),
            'coverage': stats.covered.mean(), 'zero_admission_rate': 1-stats.covered.mean(),
            'query_error_rate': stats.has_error.mean(),
            'false_admission_no_evidence': stats.loc[~gold, 'covered'].mean() if (~gold).any() else np.nan,
            'evidence_recall': (stats.loc[gold, 'correct']/stats.loc[gold, 'gold_count']).mean(),
            'all_facts_recall': stats.loc[gold, 'all_facts'].mean(),
            'multiple_set_recall': stats.loc[stats.minimum_set_size > 1, 'all_facts'].mean(),
            'singleton_set_recall': stats.loc[stats.minimum_set_size == 1, 'all_facts'].mean()}


def train(data, seed, out):
    folds = {q: i % 3 for i, q in enumerate(sorted(data['qids'], key=lambda q: hashlib.sha256(f'{seed}:inner:{q}'.encode()).hexdigest()))}
    fold = np.asarray([folds[q] for q in data['qids']])[data['codes']]
    rows, selected = [], {}
    for kind in ['list', 'top1']:
        values = []
        for C in PLAN['C']:
            oof = np.full(len(data['y']), np.nan)
            for f in range(3):
                model = fit_one(data, C, kind, fold != f)
                # Prefix correction acts within complete queries, so subset whole queries.
                valid_ids = [q for q in data['qids'] if folds[q] == f]
                part = take(data, valid_ids)
                oof[fold == f] = predict(model, part, kind)
            assert np.isfinite(oof).all()
            one = data['ranks'] == 1
            brier = float(np.mean((oof[one]-data['y'][one])**2))
            mask = admitted(data, oof, threshold(data, oof, 100))
            stats = statistics(data, mask)
            quality = (float(data['y'][mask].mean()), float((stats.correct > 0).mean()), float(stats.correct.mean()))
            key = quality+(-C,) if kind == 'list' else (-brier, -C)
            values.append((key, C))
            rows.append({'kind': kind, 'C': C, 'oof_top1_brier': brier,
                         'oof_precision_at_100': quality[0], 'oof_hit_fraction_at_100': quality[1]})
        C = max(values, key=lambda x: x[0])[1]
        fitted = fit_one(data, C, kind)
        selected[kind] = fitted
        with (out/f'model_{kind}.pkl').open('wb') as handle:
            pickle.dump(fitted, handle)
        dump(out/f'choice_{kind}.json', {'selected_C': C, 'selection': PLAN[kind+'_C_selection'], 'fitting_queries': list(data['qids'])})
    pd.DataFrame(rows).to_csv(out/'fitting_candidates.csv', index=False)
    return selected


def risk_choices(fit_data, fit_score, cal_data, cal_score):
    cutoffs = np.r_[-np.inf, np.unique(np.quantile(fit_score, np.linspace(0, 1, 101))), np.inf]
    rows = []
    nq = len(cal_data['qids'])
    gold = np.asarray([cal_data['queries'][q]['gold_count'] > 0 for q in cal_data['qids']])
    for cutoff in cutoffs:
        mask = cal_score >= cutoff
        error = np.bincount(cal_data['codes'], weights=mask*(1-cal_data['y']), minlength=nq) > 0
        covered = np.bincount(cal_data['codes'], weights=mask, minlength=nq) > 0
        rows.append({'threshold': float(cutoff), 'admitted': int(mask.sum()),
                     'query_error_rate': float(error.mean()),
                     'false_admission_no_evidence': float(covered[~gold].mean()) if (~gold).any() else np.nan})
    choices = []
    for metric in PLAN['risk_metrics']:
        for target in PLAN['risk_targets']:
            valid = [x for x in rows if np.isfinite(x[metric]) and x[metric] <= target]
            if not valid:
                continue
            best = max(valid, key=lambda x: (x['admitted'], -x['threshold']))
            choices.append((f'{metric}_{target:g}', dict(best, target=target, metric=metric)))
    return rows, choices


def evaluate(method, f, c, t, fs, cs, ts, out, base):
    assert all(np.isfinite(x).all() for x in [fs, cs, ts])
    rows, saved, rules = [], {}, []
    for target in PLAN['workload_targets']:
        rule = threshold(c, cs, target)
        policies = [(f'workload_{target}', rule)]
        if target == 100:
            policies.append(('exact_budget_100', threshold(t, ts, 100)))
        for analysis, decision in policies:
            stat = statistics(t, admitted(t, ts, decision))
            assert stat.admitted.max() <= (1 if 'top1' in method else 5)
            rows.append(dict(base, method=method, analysis=analysis, **summary(stat)))
            stat.to_parquet(out/f'query_{method}_{analysis}.parquet', index=False)
            if analysis in ['workload_100', 'exact_budget_100']:
                saved[(method, analysis)] = stat
            rules.append({'method': method, 'analysis': analysis, **decision})
    grid, choices = risk_choices(f, fs, c, cs)
    pd.DataFrame(grid).to_csv(out/f'calibration_risk_grid_{method}.csv', index=False)
    for analysis, entry in choices:
        rule = {'threshold': entry['threshold'], 'tie_cutoff': 'ffffffffffffffff'}
        stat = statistics(t, admitted(t, ts, rule))
        rows.append(dict(base, method=method, analysis=analysis, **summary(stat)))
        stat.to_parquet(out/f'query_{method}_{analysis}.parquet', index=False)
        rules.append({'method': method, 'analysis': analysis, **entry})
        if entry['target'] in [0.1, 0.2]:
            saved[(method, analysis)] = stat
    pd.DataFrame(rules).to_csv(out/f'rules_{method}.csv', index=False)
    return rows, saved


def invariance(data, models):
    altered = dict(data, y=1-data['y'])
    for kind, model in models.items():
        np.testing.assert_array_equal(predict(model, data, kind), predict(model, altered, kind))
    # List feature construction and raw ordering are checked independently below.
    return True


def metric_parts(s, metric):
    gold = s.gold_count.to_numpy() > 0
    if metric == 'evidence_precision':
        return s.correct.to_numpy(float), s.admitted.to_numpy(float)
    if metric == 'all_facts_recall':
        return s.all_facts.to_numpy(float)*gold, gold.astype(float)
    if metric == 'false_admission_no_evidence':
        return s.covered.to_numpy(float)*(~gold), (~gold).astype(float)
    if metric in ['coverage', 'query_error_rate']:
        return s['covered' if metric == 'coverage' else 'has_error'].to_numpy(float), np.ones(len(s))
    raise ValueError(metric)


def bootstrap(collection, dataset, source, out):
    keys = sorted(set.intersection(*(set(x) for x in collection.values())))
    ids = sorted(set().union(*(set(next(iter(x.values())).query_id) for x in collection.values())))
    lookup = {q: i for i, q in enumerate(ids)}
    rng = np.random.default_rng(2026101617)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=1000).astype(np.float32)
    rows = []
    for method, analysis in keys:
        if method == 'frozen_scet':
            continue
        for metric in ['evidence_precision', 'all_facts_recall', 'coverage', 'query_error_rate', 'false_admission_no_evidence']:
            samples, points = [], []
            for seed, stats in collection.items():
                a = stats[('frozen_scet', analysis)].set_index('query_id')
                b = stats[(method, analysis)].set_index('query_id').reindex(a.index)
                w = weights[:, [lookup[q] for q in a.index]]
                an, ad = metric_parts(a, metric)
                bn, bd = metric_parts(b, metric)
                if not ad.sum() or not bd.sum():
                    continue
                av, bv = w@np.column_stack([an, ad]), w@np.column_stack([bn, bd])
                samples.append(av[:, 0]/np.maximum(av[:, 1], 1)-bv[:, 0]/np.maximum(bv[:, 1], 1))
                points.append(an.sum()/ad.sum()-bn.sum()/bd.sum())
            if samples:
                low, high = np.quantile(np.mean(samples, axis=0), [.025, .975])
                rows.append({'dataset': dataset, 'source': source, 'method': 'frozen_scet', 'reference': method,
                             'analysis': analysis, 'metric': metric, 'difference': np.mean(points),
                             'ci_low': low, 'ci_high': high, 'repetitions': 1000, 'query_clusters': len(ids),
                             'scope': 'pointwise; frozen decisions; shared query weights across overlapping partitions'})
    pd.DataFrame(rows).to_csv(out/'paired_intervals.csv', index=False)


def run(args):
    started = time.monotonic()
    stage = 'smoke' if args.smoke else 'full'
    destination = OUT/stage
    destination.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        gate = load_json(OUT/'smoke/audit.json')
        assert gate['passed'] and gate['code_sha256'] == sha(__file__) and gate['plan'] == PLAN
    dump(destination/'analysis_plan.json', PLAN)
    import scipy, sklearn, pyarrow, threadpoolctl
    dump(destination/'environment.json', {'python': sys.version, 'executable': sys.executable,
        'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
        'sklearn': sklearn.__version__, 'pyarrow': pyarrow.__version__, 'threadpoolctl': threadpoolctl.__version__})
    all_rows = []
    input_hashes = {}
    datasets = ['fever'] if args.smoke else args.datasets
    for ds in datasets:
        path = ROOT/f'data/scored/{ds}_full_k100.parquet'
        frame = pd.read_parquet(path)
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
        queries = load_json(ROOT/f'data/processed/{ds}_full_queries.json')
        sources = PLAN['sources'][ds]
        if ds == 'scifact':
            mv_path = ROOT/'results/strengthening/multivers/candidate_scores.parquet'
            frame = frame.merge(pd.read_parquet(mv_path), on=['query_id', 'candidate_id'], validate='one_to_one')
            input_hashes[str(mv_path.relative_to(ROOT))] = sha(mv_path)
        source_data = {source: compact(frame, frame['raw_'+source].to_numpy(float), queries) for source in sources}
        collections = {source: {} for source in sources}
        for seed in SEEDS[:1] if args.smoke else SEEDS:
            model_path = ROOT/f'results/experiments/{ds}_full_k100/model_seed{seed}.json'
            model = load_json(model_path)
            input_hashes[str(model_path.relative_to(ROOT))] = sha(model_path)
            fitids, calids, testids = [sorted(model[k]) for k in ['fitting_query_ids', 'threshold_query_ids', 'evaluation_query_ids']]
            if args.smoke:
                fitids, calids, testids = fitids[:600], calids[:200], testids[:300]
            assert not(set(fitids)&set(calids) or set(fitids)&set(testids) or set(calids)&set(testids))
            used = set(fitids+calids+testids)
            part = frame[frame.query_id.isin(used)].reset_index(drop=True)
            scores, _, _ = admission.score_pipeline(part, model)
            frozen = compact(part, scores['scet'], queries)
            ff, fc, ft = [take(frozen, x) for x in [fitids, calids, testids]]
            # Use original frozen SCET calibration threshold at the primary target.
            # Recalibrating on the full unchanged calibration IDs yields the same mask.
            if not args.smoke:
                expected = model['thresholds']['scet']
                own = threshold(fc, fc['raw'], 100)
                assert expected['threshold'] == own['threshold'] and expected['tie_cutoff'] == own['tie_cutoff']
            for source in sources:
                out = destination/f'{ds}_{source}'/f'seed{seed}'
                out.mkdir(parents=True, exist_ok=True)
                f, c, t = [take(source_data[source], x) for x in [fitids, calids, testids]]
                fitted = train(f, seed, out)
                invariance(t, fitted)
                one = [limit_top1(x) for x in [f, c, t]]
                methods = {
                    'raw_cap5': ([f, c, t], [x['raw'] for x in [f, c, t]]),
                    'raw_top1': (one, [x['raw'] for x in one]),
                    'list_logit_cap5': ([f, c, t], [predict(fitted['list'], x, 'list') for x in [f, c, t]]),
                    'list_logit_top1': (one, [predict(fitted['top1'], x, 'top1') for x in one]),
                    'frozen_scet': ([ff, fc, ft], [x['raw'] for x in [ff, fc, ft]])}
                saved = {}
                for name, (parts, values) in methods.items():
                    rows, records = evaluate(name, *parts, *values, out, {'dataset': ds, 'source': source, 'seed': seed})
                    all_rows += rows
                    saved.update(records)
                    dump(out/f'prediction_{name}.json', {'features': 'one raw scorer only' if name != 'frozen_scet' else 'original frozen SCET',
                         'fit_queries': len(fitids), 'calibration_queries': len(calids), 'test_queries': len(testids),
                         'prediction_label_invariance': True, 'calibration_labels_for_training': False,
                         'test_labels_for_selection': False})
                collections[source][seed] = saved
                pd.DataFrame([r for r in all_rows if r['dataset'] == ds and r['source'] == source and r['seed'] == seed]).to_csv(out/'summary.csv', index=False)
                dump(out/'split_ids.json', {'fitting': fitids, 'calibration': calids, 'evaluation': testids})
                print(ds, source, seed, 'finished; seconds', round(time.monotonic()-started, 1), flush=True)
            if args.smoke:
                reverse = part.iloc[::-1].reset_index(drop=True)
                rev = compact(reverse, reverse.raw_ce.to_numpy(float), queries)
                original = take(source_data['ce'], sorted(used))
                np.testing.assert_array_equal(rev['cids'], original['cids'])
                np.testing.assert_allclose(rev['X'], original['X'], atol=0, rtol=0)
                assert any(x.covered.mean() < 1 for k, x in saved.items())
                assert np.isfinite([r['evidence_precision'] for r in all_rows if r['volume_per_100'] > 0]).all()
            del part, scores, frozen
            gc.collect()
        if not args.smoke:
            for source in sources:
                bootstrap(collections[source], ds, source, destination/f'{ds}_{source}')
        del frame, source_data, collections
        gc.collect()
    pd.DataFrame(all_rows).to_csv(destination/'summary.csv', index=False)
    dat = pd.DataFrame(all_rows)
    columns = [x for x in dat.select_dtypes(include='number') if x != 'seed']
    aggregate = dat.groupby(['dataset', 'source', 'method', 'analysis'])[columns].agg(['mean', 'std'])
    aggregate.columns = ['_'.join(x) for x in aggregate.columns]
    aggregate.to_csv(destination/'mean_sd.csv')
    dump(destination/'audit.json', {'passed': True, 'stage': stage, 'code_sha256': sha(__file__), 'plan': PLAN,
         'input_sha256': input_hashes, 'split_disjointness': True, 'prediction_label_invariance': True,
         'raw_list_order_reversal_invariance': args.smoke, 'new_neural_inference': False,
         'core_models_modified': False, 'elapsed_seconds': time.monotonic()-started})
    print(aggregate.loc[aggregate.index.get_level_values('analysis').isin(['workload_100','exact_budget_100']),
          ['evidence_precision_mean', 'volume_per_100_mean', 'coverage_mean', 'all_facts_recall_mean']].to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=['fever','hover','scifact'], default=['fever'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)
