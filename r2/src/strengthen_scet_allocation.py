"""Fixed-budget allocation sensitivity using existing frozen scores only."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import r2_admission as admission
from r2_statistics import query_statistics

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'data/scet_allocation_protocol.json'
SPEC = json.loads(PLAN.read_text(encoding='utf-8'))
OUT = ROOT/'results/scet_allocation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule(frame, scores, budget, cap):
    ranks = admission.ranks(frame, scores)
    eligible = np.flatnonzero(ranks <= cap)
    order = eligible[np.lexsort((frame.tie_key.to_numpy(str)[eligible], -scores[eligible]))]
    assert len(order) >= budget
    boundary = order[budget-1]
    return {'threshold': float(scores[boundary]), 'tie_cutoff': str(frame.iloc[boundary].tie_key),
            'eligible_rank': int(cap)}


def choose(frame, score):
    values = []
    for cap in SPEC['caps']:
        mask, _ = admission.apply_threshold(frame, score, rule(frame, score, frame.query_id.nunique(), cap))
        values.append((float(frame.gold_evidence.to_numpy(int)[mask].mean()), -cap))
    return -max(values)[1], {str(cap): value[0] for cap, value in zip(SPEC['caps'], values)}


def metric(stats, weight, name):
    weight = np.asarray(weight, float)
    def total(values):
        return weight @ np.asarray(values, float)
    def ratio(a, b):
        denominator = total(b)
        return np.divide(total(a), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
    if name == 'evidence_precision':
        return ratio(stats.correct, stats.admitted)
    if name == 'coverage':
        return ratio(stats.covered, np.ones(len(stats)))
    if name == 'query_error_rate':
        return ratio(stats.has_error, np.ones(len(stats)))
    positive = stats.gold_count.to_numpy(float) > 0
    if name == 'false_admission_no_evidence':
        return ratio(stats.covered.to_numpy(float)*(~positive), ~positive)
    if name == 'all_facts_recall':
        return ratio(stats.all_facts.to_numpy(float)*positive, positive)
    if name == 'evidence_recall':
        recall = np.divide(stats.correct, stats.gold_count, out=np.zeros(len(stats)), where=positive)
        return ratio(recall*positive, positive)
    raise ValueError(name)


def intervals(collection, dataset):
    ids = sorted(set().union(*(set(next(iter(methods.values())).query_id) for methods in collection.values())))
    lookup = {q: i for i, q in enumerate(ids)}
    rng = np.random.default_rng(SPEC['seed'])
    # Query weights are shared across overlapping partitions. Chunk to bound RAM.
    draws = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=1000).astype(np.float32)
    pairs = [('scet_cap5', 'ce_cap1'), ('scet_cap1', 'ce_cap1'),
             ('scet_selected_cap', 'ce_selected_cap'), ('scet_cap5', 'ce_cap5')]
    if dataset == 'scifact':
        pairs += [('multivers_cap5', 'multivers_cap1'), ('scet_cap5', 'multivers_cap1')]
    rows = []
    for a_name, b_name in pairs:
        differences = {name: [] for name in SPEC['metrics']}
        points = {name: [] for name in SPEC['metrics']}
        for seed, methods in collection.items():
            a = methods[a_name].set_index('query_id')
            b = methods[b_name].set_index('query_id').reindex(a.index)
            w = draws[:, [lookup[q] for q in a.index]]
            for name in SPEC['metrics']:
                differences[name].append(metric(a, w, name)-metric(b, w, name))
                points[name].append(float((metric(a, np.ones((1, len(a))), name)-metric(b, np.ones((1, len(b))), name))[0]))
        for name in SPEC['metrics']:
            delta = np.mean(differences[name], axis=0)
            low, high = np.quantile(delta, [.025, .975])
            rows.append({'dataset': dataset, 'method': a_name, 'reference': b_name, 'metric': name,
                         'difference': np.mean(points[name]), 'ci_low': low, 'ci_high': high,
                         'query_clusters': len(ids), 'bootstrap_repetitions': len(draws),
                         'interval_scope': 'pointwise_descriptive'})
    return rows


def run(smoke):
    started = time.monotonic()
    out = OUT/('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    if not smoke:
        audit = json.loads((OUT/'smoke/audit.json').read_text())
        assert audit['passed'] and audit['allocation_difference']
        assert audit['code_sha256'] == digest(Path(__file__))
        assert audit['protocol_sha256'] == digest(PLAN)
    summaries, choices, uncertainty = [], [], []
    meaningful = False
    for dataset in SPEC['datasets']:
        tag = f'{dataset}_full_k100'
        frame = pd.read_parquet(ROOT/f'data/scored/{tag}.parquet')
        if dataset == 'scifact':
            mv = pd.read_parquet(ROOT/'results/strengthening/multivers/candidate_scores.parquet')
            frame = frame.merge(mv, on=['query_id', 'candidate_id'], how='left', validate='one_to_one')
            assert np.isfinite(frame.raw_multivers).all()
        queries = json.loads((ROOT/f'data/processed/{dataset}_full_queries.json').read_text(encoding='utf-8'))
        collection = {}
        for seed in SPEC['partitions'][:1] if smoke else SPEC['partitions']:
            model = json.loads((ROOT/f'results/experiments/{tag}/model_seed{seed}.json').read_text())
            fitids, testids = model['fitting_query_ids'], model['evaluation_query_ids']
            if smoke:
                fitids, testids = sorted(fitids)[:12], sorted(testids)[:12]
            assert not set(fitids) & set(testids)
            fitting, test = [frame[frame.query_id.isin(ids)].copy().reset_index(drop=True) for ids in [fitids, testids]]
            fs, _, _ = admission.score_pipeline(fitting, model)
            ts, stress, _ = admission.score_pipeline(test, model)
            fit_scores = {'scet': fs['scet'], 'ce': fitting.raw_ce.to_numpy(float)}
            test_scores = {'scet': ts['scet'], 'ce': test.raw_ce.to_numpy(float)}
            if dataset == 'scifact':
                fit_scores['multivers'] = fitting.raw_multivers.to_numpy(float)
                test_scores['multivers'] = test.raw_multivers.to_numpy(float)
            # Freeze cap choices using fitting outcomes before test summaries are made.
            selected = {}
            for name, scores in fit_scores.items():
                chosen, values = choose(fitting, scores)
                selected[name] = chosen
                choices.append({'dataset': dataset, 'seed': seed, 'method': name, 'selected_cap': chosen,
                                'fitting_queries': len(fitids), 'fitting_precision_by_cap': json.dumps(values, sort_keys=True)})
            (out/f'choices_{dataset}_seed{seed}.json').write_text(json.dumps(choices[-len(selected):], indent=2), encoding='utf-8')
            stats = {}
            for name, score in test_scores.items():
                assert np.isfinite(score).all()
                for cap in SPEC['caps']:
                    r = rule(test, score, len(testids), cap)
                    mask, _ = admission.apply_threshold(test, score, r)
                    assert int(mask.sum()) == len(testids)
                    if smoke:
                        ix = np.arange(len(test))[::-1]
                        changed = test.iloc[ix].copy().reset_index(drop=True)
                        changed['gold_evidence'] = 1-changed.gold_evidence
                        changed_rule = rule(changed, score[ix], len(testids), cap)
                        modified, _ = admission.apply_threshold(changed, score[ix], changed_rule)
                        assert np.array_equal(mask, modified[::-1])
                    stat = query_statistics(test, score, r, stress, {q: queries[q] for q in testids})
                    assert stat.admitted.sum() == len(testids) and stat.admitted.max() <= cap
                    if cap == 1:
                        assert stat.admitted.eq(1).all()
                    method = f'{name}_cap{cap}'
                    stat.to_csv(out/f'query_{dataset}_seed{seed}_{method}.csv', index=False)
                    stats[method] = stat
                    summaries.append({'dataset': dataset, 'seed': seed, 'method': method, 'cap': cap,
                                      **admission.summarize(stat)})
                    if cap == selected[name]:
                        stats[f'{name}_selected_cap'] = stat
                        summaries.append({'dataset': dataset, 'seed': seed, 'method': f'{name}_selected_cap', 'cap': cap,
                                          **admission.summarize(stat)})
                meaningful |= stats[f'{name}_cap1'].covered.mean() > stats[f'{name}_cap5'].covered.mean()
            collection[seed] = stats
            print(dataset, seed, selected, 'elapsed', round(time.monotonic()-started, 1), flush=True)
        if not smoke:
            uncertainty.extend(intervals(collection, dataset))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'summary.csv', index=False)
    numeric = [c for c in summary.select_dtypes(include='number') if c not in ['seed']]
    mean = summary.groupby(['dataset', 'method'])[numeric].agg(['mean', 'std'])
    mean.columns = ['_'.join(c) for c in mean.columns]
    mean.to_csv(out/'mean_sd.csv')
    pd.DataFrame(choices).to_csv(out/'fitting_cap_choices.csv', index=False)
    if uncertainty:
        pd.DataFrame(uncertainty).to_csv(out/'paired_intervals.csv', index=False)
    audit = {'passed': True, 'smoke': smoke, 'code_sha256': digest(Path(__file__)), 'protocol_sha256': digest(PLAN),
             'exact_budgets_verified': True, 'cap_one_covers_all_queries': True,
             'decisions_label_and_order_invariant': smoke, 'allocation_difference': bool(meaningful),
             'external_evaluation_files_read': False, 'models_refitted': False,
             'exploratory_allocation_analysis': True, 'elapsed_seconds': time.monotonic()-started}
    (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(mean[['evidence_precision_mean', 'coverage_mean', 'all_facts_recall_mean', 'false_admission_no_evidence_mean']].to_string())
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
