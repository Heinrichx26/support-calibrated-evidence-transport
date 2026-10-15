"""Frozen R2 admission evaluation. Scoring and plotting run independently."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from r2_admission import (PROTOCOL, VIEWS, split_ids, fit_pipeline, score_pipeline,
                          fit_threshold, apply_threshold, query_statistics, summarize)
from r2_statistics import query_statistics

sys.stdout.reconfigure(encoding='utf-8')
R2 = Path(__file__).resolve().parents[1]
OUT = R2 / 'results/experiments'
OUT.mkdir(parents=True, exist_ok=True)


def read_input(dataset, mode, pool):
    tag = f'{dataset}_{mode}_k{pool}'
    frame = pd.read_parquet(R2 / f'data/scored/{tag}.parquet')
    queries = json.loads((R2 / f'data/processed/{dataset}_{mode}_queries.json').read_text(encoding='utf-8'))
    queries = {q: queries[q] for q in frame.query_id.unique()}
    return tag, frame, queries


def run(dataset, mode, pool, label='gold_evidence', seed_override=None):
    tag, frame, queries = read_input(dataset, mode, pool)
    if label != 'gold_evidence':
        tag += '_directional'
    root = OUT / tag
    if seed_override is not None:
        root = root / f'partition_seed{seed_override}'
    root.mkdir(exist_ok=True, parents=True)
    primary_rows, volume_rows, verdict_rows, subset_rows, invariants, probability_rows = [], [], [], [], [], []
    seeds = [seed_override] if seed_override is not None else (PROTOCOL['seeds'] if mode == 'full' else [13])
    for seed in seeds:
        started = time.monotonic()
        calibration, evaluation = split_ids(sorted(queries), PROTOCOL['outer_calibration_fraction'], seed)
        fitting, threshold = split_ids(calibration, PROTOCOL['fitting_share_of_calibration'], seed+1000)
        assert set(fitting).isdisjoint(threshold) and set(calibration).isdisjoint(evaluation)
        fit = frame[frame.query_id.isin(fitting)].copy()
        cal = frame[frame.query_id.isin(threshold)].copy()
        test = frame[frame.query_id.isin(evaluation)].copy().reset_index(drop=True)
        qtest = {q: queries[q] for q in evaluation}
        model_path = root / f'model_seed{seed}.json'
        input_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()
        code_hash = hashlib.sha256((R2 / 'src/r2_admission.py').read_bytes()).hexdigest()
        reusable = False
        if model_path.exists():
            model = json.loads(model_path.read_text())
            reusable = model.get('input_hash') == input_hash and model.get('protocol') == PROTOCOL and model.get('code_hash') == code_hash
        if not reusable:
            model = fit_pipeline(fit, cal, seed, label=label)
            model.update(input_hash=input_hash, protocol=PROTOCOL, code_hash=code_hash, evaluation_query_ids=evaluation)
            model_path.write_text(json.dumps(model, indent=2), encoding='utf-8')
        print(tag, seed, 'selected', model['selected_form'], 'penalty', model['penalty'], 'single', model['primary_single'], flush=True)
        scores, stress, conflict = score_pipeline(test, model)
        cscores, _, _ = score_pipeline(cal, model)
        # A separate calibration-only monotone probability map is diagnostic.
        # It does not alter the selected scorer or the frozen admission rule.
        platt = LogisticRegression(C=1, max_iter=500)
        cy = cal[label].to_numpy(int)
        if len(np.unique(cy)) == 2:
            platt.fit(cscores['scet'].reshape(-1, 1), cy)
            probabilities = platt.predict_proba(scores['scet'].reshape(-1, 1))[:, 1]
            py = test[label].to_numpy(int)
            bins = np.minimum((probabilities*10).astype(int), 9)
            bincount = np.bincount(bins, minlength=10)
            bsum = np.bincount(bins, weights=probabilities, minlength=10)
            ysum = np.bincount(bins, weights=py, minlength=10)
            ece = np.sum(np.abs(bsum-ysum))/len(py)
            probability_rows.append({'seed': seed, 'brier': float(np.mean((probabilities-py)**2)), 'ece_10_equal_width': float(ece), 'pairs': len(py), 'calibration_pairs': len(cy), 'intercept': float(platt.intercept_[0]), 'coefficient': float(platt.coef_[0, 0])})
            pd.DataFrame({'bin': range(10), 'count': bincount, 'predicted_sum': bsum, 'positive_sum': ysum}).to_csv(root / f'probability_bins_seed{seed}.csv', index=False)
        # A scorer sees text and raw view outputs; swapping every outcome cannot change it.
        changed = test.copy()
        changed['gold_evidence'] = 1-changed.gold_evidence
        changed['gold_directional'] = 1-changed.gold_directional
        changed['claim_verdict'] = 'PERMUTED_OUTCOME'
        alternate, s2, c2 = score_pipeline(changed, model)
        assert np.array_equal(stress, s2) and np.array_equal(conflict, c2)
        assert all(np.array_equal(scores[k], alternate[k]) for k in scores)
        reversed_frame = test.iloc[::-1].reset_index(drop=True)
        reversed_scores, _, _ = score_pipeline(reversed_frame, model)
        for name in scores:
            a, rank = apply_threshold(test, scores[name], model['thresholds'][name])
            a2, rank2 = apply_threshold(reversed_frame, reversed_scores[name], model['thresholds'][name])
            assert np.array_equal(a, a2[::-1]) and np.array_equal(rank, rank2[::-1]), (name, 'order invariance')
        invariants.append({'seed': seed, 'disjoint_queries': True, 'outcome_invariant_scores': True,
                           'order_invariant_admission': True, 'all_finite_raw_scores': bool(np.isfinite(test[VIEWS]).all().all()),
                           'fitting_queries': len(fitting), 'threshold_queries': len(threshold), 'evaluation_queries': len(evaluation)})
        for name, score in scores.items():
            stat = query_statistics(test, score, model['thresholds'][name], stress, qtest, label)
            stat.to_csv(root / f'query_seed{seed}_{name}.csv', index=False)
            primary_rows.append({'dataset': dataset, 'pool': pool, 'seed': seed, 'method': name, **summarize(stat)})
            for verdict, part in stat.groupby('claim_verdict'):
                verdict_rows.append({'dataset': dataset, 'pool': pool, 'seed': seed, 'method': name, 'claim_verdict': verdict, **summarize(part)})
            subset_rows.append({'dataset': dataset, 'pool': pool, 'seed': seed, 'method': name, **summarize(stat[stat.fully_verified])})
            if name in {'scet_risk_control', 'scet_zero_penalty'} or name.startswith('portfolio_'):
                continue
            for target in PROTOCOL['workload_targets_per_100']:
                rule = fit_threshold(cal, cscores[name], target)
                vstat = query_statistics(test, score, rule, stress, qtest, label)
                volume_rows.append({'dataset': dataset, 'pool': pool, 'seed': seed, 'method': name, 'calibration_target': target, 'threshold': rule['threshold'], **summarize(vstat)})
        for name in ['ce', 'bm25', 'scet']:
            rule = {'threshold': float('-inf'), 'tie_cutoff': '', 'eligible_rank': 1}
            stat = query_statistics(test, scores[name], rule, stress, qtest, label)
            stat.to_csv(root / f'query_seed{seed}_{name}_top1.csv', index=False)
            primary_rows.append({'dataset': dataset, 'pool': pool, 'seed': seed, 'method': name+'_top1', **summarize(stat)})
        pd.DataFrame({'query_id': test.query_id, 'candidate_id': test.candidate_id, 'conflict': conflict, 'stress': stress}).to_parquet(root / f'shared_stress_seed{seed}.parquet', index=False)
        print(tag, seed, 'evaluation complete', round(time.monotonic()-started, 1), 'seconds', flush=True)
    for filename, rows in [('primary', primary_rows), ('volume', volume_rows), ('verdict', verdict_rows), ('fully_verified', subset_rows)]:
        data = pd.DataFrame(rows)
        data.to_csv(root / f'{filename}.csv', index=False)
        keys = ['dataset', 'pool', 'method'] + (['calibration_target'] if filename == 'volume' else []) + (['claim_verdict'] if filename == 'verdict' else [])
        numeric = [c for c in data if c not in keys+['seed']]
        summary = data.groupby(keys)[numeric].agg(['mean', 'std']).reset_index()
        summary.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in summary.columns]
        summary.to_csv(root / f'{filename}_mean_sd.csv', index=False)
    (root / 'integrity.json').write_text(json.dumps(invariants, indent=2), encoding='utf-8')
    pd.DataFrame(probability_rows).to_csv(root / 'probability_diagnostic.csv', index=False)
    if mode == 'full' and seed_override is None:
        bootstrap(root, seeds)
    return root


def merge_partitions(dataset, mode, pool, directional=False):
    tag=f'{dataset}_{mode}_k{pool}'+('_directional' if directional else '')
    root=OUT/tag
    seeds=PROTOCOL['seeds'] if mode=='full' else [13]
    parts=[root/f'partition_seed{s}' for s in seeds]
    for filename in ['primary','volume','verdict','fully_verified']:
        data=pd.concat([pd.read_csv(p/f'{filename}.csv') for p in parts],ignore_index=True)
        data.to_csv(root/f'{filename}.csv',index=False)
        keys=['dataset','pool','method']+(['calibration_target'] if filename=='volume' else [])+(['claim_verdict'] if filename=='verdict' else [])
        numeric=[c for c in data if c not in keys+['seed']]
        summary=data.groupby(keys)[numeric].agg(['mean','std']).reset_index()
        summary.columns=['_'.join(filter(None,col)) if isinstance(col,tuple) else col for col in summary.columns]
        summary.to_csv(root/f'{filename}_mean_sd.csv',index=False)
    integrity=[]
    for p in parts:
        integrity.extend(json.loads((p/'integrity.json').read_text()))
        for source in p.iterdir():
            if source.name.startswith(('query_seed','model_seed','shared_stress_seed','probability_bins_seed')):
                shutil.copy2(source,root/source.name)
    (root/'integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    pd.concat([pd.read_csv(p/'probability_diagnostic.csv') for p in parts],ignore_index=True).to_csv(root/'probability_diagnostic.csv',index=False)
    if mode=='full':bootstrap(root,seeds)
    return root


def bootstrap(root, seeds):
    """Query cluster bootstrap preserves overlap across the four partitions."""
    methods = [v[4:] for v in VIEWS] + ['conflict_pressure', 'scet_zero_penalty']
    stats = {s: pd.read_csv(root / f'query_seed{s}_scet.csv', dtype={'query_id': str}).set_index('query_id') for s in seeds}
    all_ids = sorted(set().union(*(set(x.index) for x in stats.values())))
    lookup = {q: j for j, q in enumerate(all_ids)}
    n = len(all_ids)
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    draws = rng.multinomial(n, np.full(n, 1/n), size=PROTOCOL['bootstrap_repetitions']).astype(np.float32)
    records = []
    for name in methods:
        point, differences, counts = [], [], []
        for seed in seeds:
            a = stats[seed]
            b = pd.read_csv(root / f'query_seed{seed}_{name}.csv', dtype={'query_id': str}).set_index('query_id').reindex(a.index)
            ix = [lookup[q] for q in a.index]
            w = draws[:, ix]
            avals = w @ a[['correct', 'admitted']].to_numpy(float)
            bvals = w @ b[['correct', 'admitted']].to_numpy(float)
            differences.append(avals[:, 0]/np.maximum(avals[:, 1], 1)-bvals[:, 0]/np.maximum(bvals[:, 1], 1))
            point.append(a.correct.sum()/max(a.admitted.sum(), 1)-b.correct.sum()/max(b.admitted.sum(), 1))
            counts.append(len(a))
        dist = np.mean(differences, axis=0)
        lo, hi = np.quantile(dist, [.025, .975])
        records.append({'comparator': name, 'delta_precision_mean': float(np.mean(point)), 'ci_low': lo, 'ci_high': hi,
                        'repetitions': len(draws), 'unique_query_clusters': n, 'partitions': len(seeds),
                        'method': 'paired query cluster percentile bootstrap; shared query weights across partitions'})
    pd.DataFrame(records).to_csv(root / 'paired_bootstrap.csv', index=False)
    # Comparator choice comes exclusively from fitting data in each partition.
    selected = []
    deltas = []
    for seed in seeds:
        model = json.loads((root / f'model_seed{seed}.json').read_text())
        name = model['primary_single']
        a = stats[seed]
        b = pd.read_csv(root / f'query_seed{seed}_{name}.csv', dtype={'query_id': str}).set_index('query_id').reindex(a.index)
        w = draws[:, [lookup[q] for q in a.index]]
        aa, bb = w @ a[['correct', 'admitted']].to_numpy(float), w @ b[['correct', 'admitted']].to_numpy(float)
        deltas.append(aa[:, 0]/np.maximum(aa[:, 1], 1)-bb[:, 0]/np.maximum(bb[:, 1], 1))
        selected.append({'seed': seed, 'selected_single': name, 'scet_precision': a.correct.sum()/max(a.admitted.sum(), 1),
                         'single_precision': b.correct.sum()/max(b.admitted.sum(), 1),
                         'scet_volume': 100*a.admitted.mean(), 'single_volume': 100*b.admitted.mean()})
    dist = np.mean(deltas, axis=0)
    (root / 'primary_comparison.json').write_text(json.dumps({'partitions': selected,
        'delta_precision': float(np.mean([x['scet_precision']-x['single_precision'] for x in selected])),
        'ci_low': float(np.quantile(dist, .025)), 'ci_high': float(np.quantile(dist, .975)),
        'bootstrap_repetitions': len(draws), 'bootstrap_unit': 'query cluster shared across partitions'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['scifact', 'fever', 'hover'])
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--pool', type=int, default=100)
    p.add_argument('--directional', action='store_true')
    p.add_argument('--seed',type=int)
    p.add_argument('--merge-only',action='store_true')
    args = p.parse_args()
    for d in args.datasets:
        if args.merge_only:merge_partitions(d,args.mode,args.pool,args.directional)
        else:run(d, args.mode, args.pool, 'gold_directional' if args.directional else 'gold_evidence',args.seed)
