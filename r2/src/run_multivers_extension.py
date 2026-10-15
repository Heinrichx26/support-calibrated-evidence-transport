"""Frozen-score MultiVerS combinations and shared risk-control comparisons."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import beta, binom
from threadpoolctl import threadpool_limits

import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule

R2 = Path(__file__).resolve().parents[1]
SPEC = json.loads((R2/'data/multivers_extension_protocol.json').read_text())
ORIGINAL_VIEWS = list(admission.VIEWS)
OUT = R2/'results/multivers_extension'


@contextmanager
def extra_view():
    admission.VIEWS = ORIGINAL_VIEWS + ['raw_multivers']
    try:
        yield
    finally:
        admission.VIEWS = ORIGINAL_VIEWS


def shared_risk_rule(fitting, fit_score, calibration, cal_score, alpha):
    eligible = admission.ranks(fitting, fit_score) <= SPEC['eligible_rank']
    cutoffs = np.quantile(fit_score[eligible], SPEC['risk_grid_quantiles'])
    tests = []
    n = calibration.query_id.nunique()
    for cutoff in cutoffs:
        rule = {'threshold': float(cutoff), 'tie_cutoff': 'ffffffffffffffff'}
        mask, _ = admission.apply_threshold(calibration, cal_score, rule)
        errors = calibration.loc[mask & (calibration.gold_evidence == 0), 'query_id'].nunique()
        p = float(binom.cdf(errors, n, alpha))
        bound = float(beta.ppf(1-SPEC['failure_probability']/len(cutoffs), errors+1, n-errors)) if errors < n else 1.
        tests.append({'threshold': float(cutoff), 'queries': n, 'errors': errors,
                      'p_value': p, 'upper_bound': bound, 'admitted': int(mask.sum()),
                      'passes': p <= SPEC['failure_probability']/len(cutoffs)})
    passing = [r for r in tests if r['passes'] and r['admitted']]
    best = min(passing, key=lambda r: r['threshold']) if passing else None
    rule = {'threshold': best['threshold'] if best else float('inf'),
            'tie_cutoff': 'ffffffffffffffff'}
    if best:
        assert best['upper_bound'] <= alpha + 1e-10
    return rule, {'alpha': alpha, 'nonempty_feasible': bool(best),
                  'upper_bound': best['upper_bound'] if best else 0., 'tests': tests}


def metric(frame, weights, name):
    def total(column):
        return weights @ frame[column].to_numpy(float)
    n = weights.sum(axis=1)
    if name == 'evidence_precision':
        den = total('admitted')
        return np.divide(total('correct'), den, out=np.full_like(den, np.nan), where=den > 0)
    if name in ['annotated_per_100', 'volume_per_100']:
        return 100*total('correct' if name == 'annotated_per_100' else 'admitted')/n
    if name in ['coverage', 'query_error_rate']:
        return total('covered' if name == 'coverage' else 'has_error')/n
    gold = frame.gold_count.to_numpy(float)
    positive = gold > 0
    den = weights @ positive.astype(float)
    values = frame.all_facts.to_numpy(float) if name == 'all_facts_recall' else np.divide(
        frame.correct.to_numpy(float), gold, out=np.zeros_like(gold), where=positive)
    return (weights @ (values*positive))/den


def paired_intervals(stats, method, reference, metrics):
    ids = sorted(set().union(*(set(v[method].query_id) for v in stats.values())))
    lookup = {q: i for i, q in enumerate(ids)}
    rng = np.random.default_rng(SPEC['bootstrap_seed'])
    draws = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)),
                           size=SPEC['bootstrap_repetitions']).astype(float)
    rows = []
    for name in metrics:
        samples = []
        points = []
        for seed, methods in stats.items():
            a = methods[method].set_index('query_id')
            b = methods[reference].set_index('query_id').reindex(a.index)
            w = draws[:, [lookup[q] for q in a.index]]
            one = np.ones((1, len(a)))
            samples.append(metric(a, w, name)-metric(b, w, name))
            points.append(float((metric(a, one, name)-metric(b, one, name))[0]))
        lo, hi = np.quantile(np.mean(samples, axis=0), [.025, .975])
        rows.append({'method': method, 'reference': reference, 'metric': name,
                     'difference': np.mean(points), 'ci_low': lo, 'ci_high': hi,
                     'query_clusters': len(ids), 'bootstrap_repetitions': len(draws)})
    return rows


def run(smoke=False, pool=100, context_scores=None):
    start = time.monotonic()
    out = OUT/(f'k{pool}' + ('_smoke' if smoke else '') + ('_context' if context_scores else ''))
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(R2/f'data/scored/scifact_full_k{pool}.parquet')
    mv_path = Path(context_scores) if context_scores else (
        R2/'results/strengthening/multivers/candidate_scores.parquet' if pool == 100
        else OUT/'inference_full_k500/candidate_scores.parquet')
    mv = pd.read_parquet(mv_path)
    frame = frame.merge(mv, on=['query_id', 'candidate_id'], how='left', validate='one_to_one', sort=False)
    assert len(frame) == 300*pool and frame.raw_multivers.notna().all()
    assert np.isfinite(frame.raw_multivers).all()
    queries = json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    summaries, choices, riskrows = [], [], []
    collections = {}
    for seed in ([13] if smoke else SPEC['seeds']):
        old = json.loads((R2/f'results/experiments/scifact_full_k{pool}/model_seed{seed}.json').read_text())
        fitids, calids, testids = [old[k] for k in ['fitting_query_ids', 'threshold_query_ids', 'evaluation_query_ids']]
        assert not (set(fitids)&set(calids) or set(fitids)&set(testids) or set(calids)&set(testids))
        if smoke:
            testids = sorted(testids)[:12]
        fitting, calibration, test = [frame[frame.query_id.isin(ids)].copy().reset_index(drop=True)
                                      for ids in [fitids, calids, testids]]
        qtest = {q: queries[q] for q in testids}
        oldscores = [admission.score_pipeline(f, old)[0]['scet'] for f in [fitting, calibration, test]]
        _, stress, _ = admission.score_pipeline(test, old)
        with extra_view(), threadpool_limits(limits=2):
            model = admission.fit_pipeline(fitting, calibration, seed)
            extended = [admission.score_pipeline(f, model)[0]['scet'] for f in [fitting, calibration, test]]
        model.update(evaluation_query_ids=testids, extension_protocol=SPEC['version'])
        (out/f'model_seed{seed}.json').write_text(json.dumps(model, indent=2), encoding='utf-8')
        choices.append({'pool': pool, 'seed': seed, 'selected_form': model['selected_form'],
                        'penalty': model['penalty'], 'selected_single': model['primary_single'],
                        'fit_queries': len(fitids), 'calibration_queries': len(calids),
                        'evaluation_queries': len(testids)})
        scores = {'scet': oldscores, 'multivers': [f.raw_multivers.to_numpy(float) for f in [fitting, calibration, test]],
                  'scet_multivers': extended}
        for name, (fs, cs, ts) in scores.items():
            for target in ([100] if smoke else SPEC['workload_targets']):
                rule = admission.fit_threshold(calibration, cs, target)
                stat = query_statistics(test, ts, rule, stress, qtest)
                key = f'prospective_{target}'
                collections.setdefault(key, {}).setdefault(seed, {})[name] = stat
                stat.to_csv(out/f'query_seed{seed}_{name}_{key}.csv', index=False)
                summaries.append({'pool': pool, 'seed': seed, 'method': name, 'analysis': key,
                                  **admission.summarize(stat)})
            rule = budget_rule(test, ts, len(testids))
            stat = query_statistics(test, ts, rule, stress, qtest)
            assert stat.admitted.sum() == len(testids)
            collections.setdefault('exact_budget', {}).setdefault(seed, {})[name] = stat
            stat.to_csv(out/f'query_seed{seed}_{name}_exact_budget.csv', index=False)
            summaries.append({'pool': pool, 'seed': seed, 'method': name, 'analysis': 'exact_budget',
                              **admission.summarize(stat)})
            for alpha in SPEC['risk_targets']:
                rule, info = shared_risk_rule(fitting, fs, calibration, cs, alpha)
                key = f'risk_{alpha:g}'
                stat = query_statistics(test, ts, rule, stress, qtest)
                collections.setdefault(key, {}).setdefault(seed, {})[name] = stat
                stat.to_csv(out/f'query_seed{seed}_{name}_{key}.csv', index=False)
                summaries.append({'pool': pool, 'seed': seed, 'method': name, 'analysis': key,
                                  **admission.summarize(stat)})
                riskrows.append({'pool': pool, 'seed': seed, 'method': name, 'alpha': alpha,
                                 'nonempty_feasible': info['nonempty_feasible'],
                                 'simultaneous_upper_bound': info['upper_bound'],
                                 'threshold': rule['threshold'], 'calibration_queries': len(calids)})
                (out/f'risk_seed{seed}_{name}_{alpha:g}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print('pool', pool, 'seed', seed, choices[-1], 'seconds', round(time.monotonic()-start, 1), flush=True)
    data = pd.DataFrame(summaries)
    data.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(choices).to_csv(out/'selected_forms.csv', index=False)
    pd.DataFrame(riskrows).to_csv(out/'risk_rules.csv', index=False)
    numeric = [c for c in data.select_dtypes(include='number') if c not in ['pool', 'seed']]
    aggregate = data.groupby(['analysis', 'method'])[numeric].agg(['mean', 'std']).reset_index()
    aggregate.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in aggregate.columns]
    aggregate.to_csv(out/'mean_sd.csv', index=False)
    if not smoke:
        intervals = []
        for analysis, stats in collections.items():
            metrics = ['annotated_per_100', 'volume_per_100', 'coverage', 'evidence_recall', 'all_facts_recall', 'query_error_rate']
            if not analysis.startswith('risk'):
                metrics = ['evidence_precision'] + metrics
            for reference in ['multivers', 'scet']:
                for row in paired_intervals(stats, 'scet_multivers', reference, metrics):
                    intervals.append({'analysis': analysis, **row})
        pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    audit = {'smoke': smoke, 'pool': pool, 'score_pairs': len(frame),
             'finite_scores': True, 'query_splits_disjoint': True, 'exact_budget_verified': True,
             'protocol_sha256': hashlib.sha256((R2/'data/multivers_extension_protocol.json').read_bytes()).hexdigest(),
             'multivers_scores_sha256': hashlib.sha256(mv_path.read_bytes()).hexdigest(),
             'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'elapsed_seconds': time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(data[data.analysis.isin(['exact_budget', 'risk_0.2'])][
        ['seed', 'method', 'analysis', 'evidence_precision', 'annotated_per_100', 'volume_per_100', 'coverage']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--pool', type=int, default=100, choices=[100, 500])
    parser.add_argument('--context-scores')
    args = parser.parse_args()
    if not args.smoke:
        suffix = '_context' if args.context_scores else ''
        assert json.loads((OUT/f'k{args.pool}_smoke{suffix}/audit.json').read_text())['finite_scores']
    run(args.smoke, args.pool, args.context_scores)
