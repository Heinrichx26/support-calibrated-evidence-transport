"""Query-conditional admission risk with independent threshold calibration.

The saved fitted scorers are fixed. Twenty thresholds depend on fitting scores
only. Calibration simultaneously bounds conditional query error and coverage.
No fitted scorer, threshold grid, risk target or coverage target uses test labels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import beta
from threadpoolctl import threadpool_limits

import r2_admission as admission
from r2_statistics import query_statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/r5_risk_conflict/risk'
PLAN = {
    'version': 'r5_conditional_query_risk_20261016',
    'datasets': ['scifact', 'fever', 'hover'],
    'seeds': [13, 17, 29, 7],
    'methods': ['scet', 'ce', 'selected_single'],
    'fitting_top_one_quantiles': np.linspace(0, .95, 20).tolist(),
    'conditional_query_error_target': .20,
    'coverage_lower_target': .05,
    'family_failure_probability': .05,
    'correction_family': '2 one-sided bounds x 20 thresholds x all 3 methods, separately within each fixed partition',
    'rule_selection': 'maximum calibration coverage lower bound among jointly passing rules; ties: lower threshold',
    'assumptions': 'iid future queries from the same candidate protocol; scorer and threshold grid frozen before independent calibration',
    'empty_output_conditional_risk': 'undefined',
    'threads': 2,
}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_risk(stats):
    n = len(stats)
    covered = int(stats.covered.sum())
    errors = int(stats.has_error.sum())
    assert 0 <= errors <= covered <= n
    conditional = errors / covered if covered else np.nan
    coverage = covered / n
    unconditional = errors / n
    if covered:
        assert np.isclose(unconditional, coverage * conditional, atol=1e-14)
    return {
        'queries': n, 'covered_queries': covered, 'error_queries': errors,
        'coverage': coverage, 'unconditional_query_error': unconditional,
        'conditional_query_error': conditional,
        'items_per_100': 100 * stats.admitted.sum() / n,
        'evidence_precision': stats.correct.sum() / stats.admitted.sum() if stats.admitted.sum() else np.nan,
    }


def aggregate(table, groups):
    numeric = [x for x in table.select_dtypes(include='number').columns if x not in groups + ['seed']]
    result = table.groupby(groups)[numeric].agg(['mean', 'std']).reset_index()
    result.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in result.columns]
    return result


def saved_descriptive(out):
    rows = []
    for dataset in PLAN['datasets']:
        tag = f'{dataset}_full_k100'
        root = ROOT / 'results/experiments' / tag
        for seed in PLAN['seeds']:
            model = json.loads((root / f'model_seed{seed}.json').read_text())
            for method in ['scet', 'ce', 'selected_single', 'scet_risk_control']:
                name = model['primary_single'] if method == 'selected_single' else method
                stats = pd.read_csv(root / f'query_seed{seed}_{name}.csv', dtype={'query_id': str})
                rows.append({'dataset': dataset, 'seed': seed, 'method': method,
                             'scorer': name, **query_risk(stats)})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'saved_query_risk_by_partition.csv', index=False)
    mean = aggregate(table, ['dataset', 'method'])
    mean.to_csv(out / 'saved_query_risk_mean_sd.csv', index=False)
    return mean


def load(dataset, ids):
    columns = ['query_id', 'candidate_id', 'query_text', 'evidence_text', 'gold_evidence', 'tie_key'] + admission.VIEWS
    return pd.read_parquet(ROOT / f'data/scored/{dataset}_full_k100.parquet',
                           columns=columns, filters=[('query_id', 'in', list(ids))]).reset_index(drop=True)


def method_scores(frame, model):
    values = admission.score_pipeline(frame, model)[0]
    return {'scet': values['scet'], 'ce': frame.raw_ce.to_numpy(float),
            'selected_single': frame['raw_' + model['primary_single']].to_numpy(float)}


def top_values(frame, score):
    first = admission.ranks(frame, score) == 1
    return np.asarray(score)[first]


def rule_stats(frame, score, threshold, ids):
    rule = {'threshold': float(threshold), 'tie_cutoff': 'ffffffffffffffff', 'eligible_rank': 5}
    admitted, _ = admission.apply_threshold(frame, score, rule)
    grouped = frame[['query_id', 'gold_evidence']].copy()
    grouped['admitted'] = admitted
    grouped['error'] = admitted & frame.gold_evidence.eq(0).to_numpy()
    counts = grouped.groupby('query_id')[['admitted', 'error']].sum().reindex(ids, fill_value=0)
    return {
        'queries': len(ids), 'covered_queries': int(counts.admitted.gt(0).sum()),
        'error_queries': int(counts.error.gt(0).sum()), 'admitted': int(counts.admitted.sum()),
    }


def calibrate(fit, fit_scores, cal, cal_scores, ids):
    k = len(PLAN['fitting_top_one_quantiles'])
    level = PLAN['family_failure_probability'] / (2 * k * len(PLAN['methods']))
    tests, chosen = [], {}
    for method in PLAN['methods']:
        grid = np.quantile(top_values(fit, fit_scores[method]), PLAN['fitting_top_one_quantiles'])
        eligible = admission.ranks(cal, cal_scores[method]) <= 5
        small_cal = cal.loc[eligible].copy().reset_index(drop=True)
        small_scores = np.asarray(cal_scores[method])[eligible]
        records = []
        for index, value in enumerate(grid):
            result = rule_stats(small_cal, small_scores, value, ids)
            n, m, e = result['queries'], result['covered_queries'], result['error_queries']
            upper = float(beta.ppf(1-level, e+1, m-e)) if e < m else 1.
            lower = float(beta.ppf(level, m, n-m+1)) if m else 0.
            passes = bool(m and upper <= PLAN['conditional_query_error_target'] and lower >= PLAN['coverage_lower_target'])
            row = {'method': method, 'grid_index': index, 'fitting_quantile': PLAN['fitting_top_one_quantiles'][index],
                   'threshold': float(value), **result, 'conditional_error_upper': upper,
                   'coverage_lower': lower, 'one_sided_failure_probability': level, 'passes': passes}
            records.append(row)
            tests.append(row)
        passing = [x for x in records if x['passes']]
        best = max(passing, key=lambda r: (r['coverage_lower'], -r['threshold'])) if passing else None
        chosen[method] = best
    return tests, chosen


def run(stage, datasets, seeds):
    start = time.monotonic()
    out = OUT / stage
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'protocol.json', PLAN)
    if stage == 'descriptive':
        mean = saved_descriptive(out)
        print(mean[['dataset', 'method', 'coverage_mean', 'unconditional_query_error_mean', 'conditional_query_error_mean']].to_string(index=False), flush=True)
        return
    if stage == 'full':
        gate = json.loads((OUT / 'smoke/audit.json').read_text())
        assert gate['passed'] and gate['code_sha256'] == digest(Path(__file__))
    summaries, all_tests, selected = [], [], []
    for dataset in datasets:
        source = ROOT / f'results/experiments/{dataset}_full_k100'
        queries = json.loads((ROOT / f'data/processed/{dataset}_full_queries.json').read_text(encoding='utf-8'))
        for seed in seeds:
            model = json.loads((source / f'model_seed{seed}.json').read_text())
            fit_ids, cal_ids, test_ids = [model[x] for x in ['fitting_query_ids', 'threshold_query_ids', 'evaluation_query_ids']]
            assert not (set(fit_ids)&set(cal_ids) or set(fit_ids)&set(test_ids) or set(cal_ids)&set(test_ids))
            fit, cal = load(dataset, fit_ids), load(dataset, cal_ids)
            fs, cs = method_scores(fit, model), method_scores(cal, model)
            tests, chosen = calibrate(fit, fs, cal, cs, cal_ids)
            for x in tests:
                all_tests.append({'dataset': dataset, 'seed': seed, **x})
            # Every learned choice is serialized before any held-out outcomes are loaded.
            dump(out / f'freeze_{dataset}_seed{seed}.json', {
                'dataset': dataset, 'seed': seed, 'chosen': chosen,
                'fitting_ids': fit_ids, 'calibration_ids': cal_ids, 'test_ids': test_ids,
                'scorer_model_sha256': digest(source / f'model_seed{seed}.json'),
                'test_labels_accessed_at_freeze': False,
            })
            if stage == 'smoke':
                reversed_cal = cal.iloc[::-1].copy().reset_index(drop=True)
                reversed_cal['gold_evidence'] = 1-reversed_cal.gold_evidence
                changed = method_scores(reversed_cal, model)
                for name in cs:
                    assert np.allclose(changed[name][::-1], cs[name], atol=1e-12, rtol=0)
                # Constructed checks establish distinct unconditional/conditional events.
                fake = pd.DataFrame({'admitted':[1,1,0,0], 'correct':[0,1,0,0], 'covered':[1,1,0,0], 'has_error':[1,0,0,0]})
                check = query_risk(fake)
                assert check['conditional_query_error']==.5 and check['unconditional_query_error']==.25
            else:
                test = load(dataset, test_ids)
                ts = method_scores(test, model)
                stress = np.zeros(len(test), dtype=bool)
                for name, choice in chosen.items():
                    rule = {'threshold': choice['threshold'] if choice else float('inf'), 'tie_cutoff':'ffffffffffffffff', 'eligible_rank':5}
                    stats = query_statistics(test, ts[name], rule, stress, {q:queries[q] for q in test_ids})
                    stats.to_csv(out / f'query_{dataset}_seed{seed}_{name}.csv', index=False)
                    summaries.append({'dataset':dataset, 'seed':seed, 'method':name, 'feasible':bool(choice), **query_risk(stats)})
                    selected.append({'dataset':dataset, 'seed':seed, 'method':name, 'feasible':bool(choice),
                                     'threshold':rule['threshold'],
                                     'calibration_conditional_error_upper': choice['conditional_error_upper'] if choice else np.nan,
                                     'calibration_coverage_lower': choice['coverage_lower'] if choice else np.nan,
                                     'calibration_covered_queries': choice['covered_queries'] if choice else 0})
            print(dataset, seed, stage, 'feasible', {k:bool(v) for k,v in chosen.items()}, 'seconds', round(time.monotonic()-start,1), flush=True)
    pd.DataFrame(all_tests).to_csv(out / 'calibration_tests.csv', index=False)
    if summaries:
        table = pd.DataFrame(summaries)
        table.to_csv(out / 'evaluation_by_partition.csv', index=False)
        aggregate(table, ['dataset','method']).to_csv(out / 'evaluation_mean_sd.csv', index=False)
        pd.DataFrame(selected).to_csv(out / 'selected_rules.csv', index=False)
    dump(out / 'audit.json', {
        'passed':True, 'stage':stage, 'code_sha256':digest(Path(__file__)), 'plan':PLAN,
        'test_outcomes_used_for_model_or_threshold_selection':False,
        'smoke_accessed_test':False, 'smoke_label_and_order_invariance':stage=='smoke',
        'elapsed_seconds':time.monotonic()-start,
        'validity':'For each fixed threshold, error count conditional on the random covered-query count is binomial with selective query-error probability. Clopper-Pearson bounds and coverage lower bounds are jointly corrected across all method-threshold pairs. Empty output has undefined conditional error and fails the minimum coverage criterion.',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['descriptive','smoke','full'], default='smoke')
    parser.add_argument('--datasets', nargs='+', choices=PLAN['datasets'])
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.stage, args.datasets or (['scifact'] if args.stage=='smoke' else PLAN['datasets']),
            args.seeds or ([13] if args.stage=='smoke' else PLAN['seeds']))
