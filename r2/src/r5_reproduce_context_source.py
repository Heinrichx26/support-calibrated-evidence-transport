"""Annotated-page source-component sensitivity of frozen context decisions.

Runs a 12-query-per-partition smoke before full analysis. No inference, fitting,
threshold selection or fixed-batch reallocation is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

R = Path(__file__).resolve().parents[1]
ROOT = R / 'results/r5_context_gain'
OUT = ROOT / 'source_sensitivity'
SEEDS = [13, 17, 29, 7]
PROTOCOLS = ['prospective', 'fixed_batch']
METHODS = ['all', 'capacity_ce', 'original_scet', 'ce_list_logit']
METRICS = ['precision', 'complete_set_recall', 'singleton_set_recall',
           'multiple_set_recall', 'coverage', 'no_evidence_admission']
MAP = R / 'results/evidence_completion/contribution/full/fever_source_components.csv'
PLAN = {
    'created': '2026-10-16', 'seeds': SEEDS, 'protocols': PROTOCOLS,
    'methods': METHODS, 'metrics': METRICS,
    'primary': 'All minus capacity-matched CE precision in prospective and fixed-batch protocols; two comparisons.',
    'other_comparisons': 'All minus each of capacity CE, original SCET and pure CE list-logit for all listed metrics; retain every result as exploratory.',
    'bootstrap': '1000 annotated-page component multinomial draws; one source weight shared by every query in its component and by every occurrence across partitions.',
    'bootstrap_seed': 2026101607, 'simultaneous': 'Bonferroni two-primary family: 97.5% marginal percentile intervals, nominal 95% family coverage. Other intervals are pointwise 95%.',
    'subgroups': 'Original smallest nonempty annotated evidence-set size: one for singleton; at least two for multiple; original denominators retain unavailable evidence.',
    'missing_source_map': 'Assign a unique missing:<query_id> component, and report all missing query IDs.',
    'conditioning': 'Frozen models, observed partitions and frozen admission decisions. Fixed-batch budget is not reallocated during bootstrap.',
    'scope': 'Source-dependence sensitivity; this is not a source-disjoint training/evaluation experiment.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def paths(seed, protocol):
    simple = 'workload_100' if protocol == 'prospective' else 'exact_budget_100'
    folder = 'experiments' if protocol == 'prospective' else 'strengthening'
    return {
        'all': ROOT / f'evaluation/query_seed{seed}_all_{protocol}.csv',
        'capacity_ce': ROOT / f'evaluation/query_seed{seed}_ce_{protocol}.csv',
        'original_scet': R / f'results/{folder}/fever_full_k100/query_seed{seed}_scet.csv',
        'ce_list_logit': R / f'results/r5_simple_controls/full/fever_ce/seed{seed}/query_list_logit_cap5_{simple}.parquet',
    }


def components(frame, metric):
    if metric == 'precision':
        return frame.correct.to_numpy(float), frame.admitted.to_numpy(float)
    if metric == 'coverage':
        return frame.covered.to_numpy(float), np.ones(len(frame))
    if metric == 'no_evidence_admission':
        mask = frame.gold_count.eq(0).to_numpy()
        return frame.covered.to_numpy(float) * mask, mask.astype(float)
    mask = frame.minimum_set_size.to_numpy()
    mask = mask > 0 if metric == 'complete_set_recall' else mask == 1 if metric == 'singleton_set_recall' else mask >= 2
    return frame.all_facts.to_numpy(float) * mask, mask.astype(float)


def run(smoke):
    started = time.monotonic()
    out = OUT / ('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    planpath = OUT / 'plan.json'
    serialized = json.dumps(PLAN, ensure_ascii=False, indent=2) + '\n'
    if planpath.exists():
        assert planpath.read_text(encoding='utf-8') == serialized
    else:
        planpath.write_text(serialized, encoding='utf-8')
    if not smoke:
        prior = json.loads((OUT / 'smoke/audit.json').read_text())
        assert prior['passed'] and prior['code_sha256'] == sha(__file__)
    evaluation_audit = ROOT / 'evaluation/audit.json'
    assert json.loads(evaluation_audit.read_text())['passed'], 'Formal frozen evaluation must finish first.'
    source = pd.read_csv(MAP, dtype={'query_id': str, 'gold_page_component': str})
    assert not source.query_id.duplicated().any()
    lookup = source.set_index('query_id').gold_page_component.to_dict()
    queries_path = R / 'data/processed/fever_full_queries.json'
    queries = json.loads(queries_path.read_text(encoding='utf-8'))
    minimum = {qid: min((len(set(group)) for group in q['gold_sets'] if group), default=0)
               for qid, q in queries.items()}
    collection, dependencies, descriptive = {}, [], []
    for seed in SEEDS:
        for protocol in PROTOCOLS:
            files = paths(seed, protocol)
            held = pd.read_csv(files['all'], dtype={'query_id': str})
            ids = held.query_id.tolist()[:12] if smoke else held.query_id.tolist()
            assert len(ids) == len(set(ids))
            pair = {}
            for method, path in files.items():
                data = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'query_id': str})
                data['query_id'] = data.query_id.astype(str)
                assert not data.query_id.duplicated().any()
                assert set(ids) <= set(data.query_id)
                data = data.set_index('query_id').reindex(ids).reset_index()
                expected = data.query_id.map(minimum).to_numpy(int)
                if 'minimum_set_size' in data:
                    assert np.array_equal(data.minimum_set_size.to_numpy(int), expected)
                data['minimum_set_size'] = expected
                assert np.array_equal(data.gold_count.to_numpy(int),
                    np.asarray([queries[q]['gold_count'] for q in ids]))
                pair[method] = data
                record = {'seed': seed, 'protocol': protocol, 'method': method, 'queries': len(data),
                          'singleton_queries': int((expected == 1).sum()), 'multiple_queries': int((expected >= 2).sum()),
                          'no_evidence_queries': int((expected == 0).sum()), 'admitted': int(data.admitted.sum()),
                          'annotated_admitted': int(data.correct.sum())}
                for metric in METRICS:
                    num, den = components(data, metric)
                    record[metric] = num.sum() / den.sum() if den.sum() else np.nan
                    if 'set_recall' in metric:
                        record[metric + '_complete_claims'] = int(num.sum())
                        record[metric + '_denominator'] = int(den.sum())
                descriptive.append(record)
                dependencies.append({'path': str(path.relative_to(R)), 'sha256': sha(path)})
            collection[(seed, protocol)] = pair
    ids = sorted(set().union(*(set(value['all'].query_id) for value in collection.values())))
    missing = [qid for qid in ids if qid not in lookup]
    assignments = {qid: 'source:' + lookup[qid] if qid in lookup else 'missing:' + qid for qid in ids}
    group_ids = sorted(set(assignments.values()))
    group_lookup = {group: index for index, group in enumerate(group_ids)}
    ncomponents = len(group_ids)
    weights = np.random.default_rng(PLAN['bootstrap_seed']).multinomial(
        ncomponents, np.full(ncomponents, 1 / ncomponents), size=1000).astype(np.float32)
    cache = {}
    for key, methods in collection.items():
        for method, frame in methods.items():
            codes = np.asarray([group_lookup[assignments[q]] for q in frame.query_id])
            for metric in METRICS:
                num, den = components(frame, metric)
                nums = np.bincount(codes, weights=num, minlength=ncomponents)
                dens = np.bincount(codes, weights=den, minlength=ncomponents)
                aa, bb = weights @ nums, weights @ dens
                draws = np.divide(aa, bb, out=np.full(len(weights), np.nan), where=bb > 0)
                point = nums.sum() / dens.sum() if dens.sum() else np.nan
                cache[(key, method, metric)] = (point, draws)
    intervals = []
    for protocol in PROTOCOLS:
        for reference in METHODS[1:]:
            for metric in METRICS:
                points, distributions = [], []
                for seed in SEEDS:
                    pa, da = cache[((seed, protocol), 'all', metric)]
                    pb, db = cache[((seed, protocol), reference, metric)]
                    points.append(pa-pb)
                    distributions.append(da-db)
                valid = np.all(np.isfinite(distributions), axis=0)
                distribution = np.mean(np.asarray(distributions)[:, valid], axis=0)
                lo, hi = np.quantile(distribution, [.025, .975]) if len(distribution) else (np.nan, np.nan)
                primary = reference == 'capacity_ce' and metric == 'precision'
                simultaneous = np.quantile(distribution, [.0125, .9875]) if primary and len(distribution) else [np.nan, np.nan]
                intervals.append({'method': 'all', 'reference': reference, 'protocol': protocol, 'metric': metric,
                    'difference_mean_of_four_partitions': float(np.mean(points)), 'pointwise_ci_low': lo, 'pointwise_ci_high': hi,
                    'primary': primary, 'two_primary_simultaneous_low': simultaneous[0], 'two_primary_simultaneous_high': simultaneous[1],
                    'bootstrap_draws': len(weights), 'finite_draws': len(distribution), 'source_components': ncomponents,
                    'scope': 'Conditional frozen decisions; source-component resampling with shared weights across partitions.'})
    values = pd.DataFrame(descriptive)
    values.to_csv(out / 'descriptive_by_partition.csv', index=False)
    numeric = [c for c in values.select_dtypes(include='number') if c != 'seed']
    means = values.groupby(['protocol', 'method'])[numeric].agg(['mean', 'std'])
    means.columns = ['_'.join(value) for value in means.columns]
    means.to_csv(out / 'descriptive_mean_sd.csv')
    results = pd.DataFrame(intervals)
    results.to_csv(out / 'source_paired_intervals.csv', index=False)
    assignments_frame = pd.DataFrame([{'query_id': qid, 'component': assignments[qid],
                                      'missing_from_source_map': qid in missing} for qid in ids])
    assignments_frame.to_csv(out / 'source_assignments.csv', index=False)
    group_sizes = assignments_frame.groupby('component').size()
    audit = {'passed': True, 'smoke': smoke, 'code_sha256': sha(__file__), 'plan_sha256': sha(planpath),
             'evaluation_audit_sha256': sha(evaluation_audit), 'source_map_sha256': sha(MAP),
             'queries_sha256': sha(queries_path), 'evaluated_query_union': len(ids),
             'source_components': ncomponents, 'largest_component_queries': int(group_sizes.max()),
             'source_map_missing_queries': len(missing), 'source_map_missing_query_ids': missing,
             'source_map_query_coverage': 1-len(missing)/len(ids), 'unique_missing_components_assigned': len(missing),
             'all_prespecified_comparisons_saved': len(results) == 36,
             'primary_comparisons': 2, 'bootstrap_repetitions': 1000, 'all_input_dependencies': dependencies,
             'smallest_original_evidence_set_defines_subgroups': True,
             'fixed_decision_conditioning': True, 'fixed_batch_budget_reallocated': False,
             'neural_inference_or_refitting': False, 'source_disjoint_training_test': False,
             'elapsed_seconds': time.monotonic()-started}
    dump(out / 'audit.json', audit)
    lines = ['# Annotated-page component sensitivity', '',
        'These intervals resample frozen query outcomes by shared annotated-page component. They do not evaluate source-disjoint training or refit models. The fixed-batch budget remains frozen during resampling.', '',
        f'Query union: {len(ids)}; source components: {ncomponents}; largest component: {int(group_sizes.max())}; queries missing from map: {len(missing)}.', '',
        'Two primary precision comparisons (all minus capacity-matched CE), with nominal 95% simultaneous coverage across the two protocols:', '']
    for row in results[results.primary].to_dict('records'):
        lines.append(f'- {row["protocol"]}: {row["difference_mean_of_four_partitions"]:.6f} '
            f'[{row["two_primary_simultaneous_low"]:.6f}, {row["two_primary_simultaneous_high"]:.6f}].')
    lines += ['', 'All secondary comparisons, including negative results, are retained in source_paired_intervals.csv. Singleton and multiple-sentence set-recall comparisons use the smallest original annotated set, preserving missing sentences in denominators. Secondary intervals are pointwise 95%.', '',
              'Selected descriptive outcomes and trade-offs at the same fixed batch budget:', '']
    for method in METHODS:
        row = means.loc[('fixed_batch', method)]
        lines.append(f'- {method}: precision {row["precision_mean"]:.6f}; singleton set recall {row["singleton_set_recall_mean"]:.6f}; '
            f'multiple set recall {row["multiple_set_recall_mean"]:.6f}; no-evidence admission {row["no_evidence_admission_mean"]:.6f}.')
    lines += ['', 'No-evidence admission uses queries without annotated evidence as its denominator; smaller values mean fewer such queries receive output. Higher precision and set recall describe more annotated evidence. These outcomes express distinct trade-offs.']
    (out / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['passed', 'smoke', 'evaluated_query_union', 'source_components',
          'source_map_missing_queries', 'all_prespecified_comparisons_saved', 'elapsed_seconds']}), flush=True)
    print(results[results.primary][['protocol', 'difference_mean_of_four_partitions',
          'two_primary_simultaneous_low', 'two_primary_simultaneous_high']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    with threadpool_limits(limits=2):
        run(parser.parse_args().smoke)
