"""Verify additional-validation files and reconstruct saved numerical outcomes.

This verification reads saved records, cached scores and fixed rules. It does
not train models or run pretrained neural inference. Run --smoke before --full.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/r5_strengthening/release_verification'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def equal(actual, expected, label):
    assert np.isclose(actual, expected, atol=2e-12, rtol=2e-12,
                      equal_nan=True), (label, actual, expected)


def outcome_summary(stats):
    gold = stats.gold_count.gt(0)
    issued = stats.admitted.sum()
    covered = stats.covered.astype(float)
    result = {
        'queries': len(stats),
        'evidence_precision': stats.correct.sum() / issued if issued else np.nan,
        'volume_per_100': 100 * stats.admitted.mean(),
        'annotated_per_100': 100 * stats.correct.mean(),
        'coverage': covered.mean(), 'zero_admission_rate': 1 - covered.mean(),
        'query_error_rate': stats.has_error.mean(),
        'false_admission_no_evidence': covered[~gold].mean(),
        'evidence_recall': (stats.loc[gold, 'correct'] / stats.loc[gold, 'gold_count']).mean(),
        'all_facts_recall': stats.loc[gold, 'all_facts'].mean(),
        'conditional_query_error': stats.has_error.sum() / covered.sum() if covered.sum() else np.nan,
        'covered_queries': covered.sum(), 'error_queries': stats.has_error.sum(),
        'unconditional_query_error': stats.has_error.mean(),
        'items_per_100': 100 * stats.admitted.mean(),
        'admitted_total': issued, 'correct_total': stats.correct.sum(),
    }
    if 'minimum_set_size' in stats:
        result['multiple_set_recall'] = stats.loc[stats.minimum_set_size.gt(1), 'all_facts'].mean()
        result['singleton_set_recall'] = stats.loc[stats.minimum_set_size.eq(1), 'all_facts'].mean()
    return result


def check_row(stats, row, label):
    values = outcome_summary(stats)
    checked = 0
    for key, actual in values.items():
        if key in row:
            equal(actual, row[key], label + ':' + key)
            checked += 1
    assert stats.correct.le(stats.admitted).all(), label
    assert stats.has_error.le(stats.covered).all(), label
    assert stats.all_facts.le(stats.gold_count.gt(0)).all(), label
    return checked


def verify_manifest(smoke):
    manifest = json.loads((ROOT / 'ADDITIONAL_VALIDATION_MANIFEST.json').read_text())
    rows = manifest['files']
    if smoke:
        rows = [v for v in rows if v['path'].endswith('.py')][:4]
    for row in rows:
        p = ROOT / row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'], row['path']
        assert digest(p) == row['sha256'], row['path']
    return len(rows)


def verify_simple(smoke):
    checked_rows = checked_values = 0
    tables = sorted((ROOT / 'results/r5_simple_controls/full').glob('*_*/seed*/summary.csv'))
    if smoke:
        tables = tables[:1]
    for path in tables:
        table = pd.read_csv(path)
        if smoke:
            table = table[table.analysis.eq('workload_100')].head(1)
        for row in table.to_dict('records'):
            qp = path.parent / f"query_{row['method']}_{row['analysis']}.parquet"
            stats = pd.read_parquet(qp)
            checked_values += check_row(stats, row, str(qp.relative_to(ROOT)))
            checked_rows += 1
    return {'rows': checked_rows, 'numeric_values': checked_values}


def ranked(frame, column):
    ordered = frame.sort_values(['query_id', column, 'tie_key'],
                                ascending=[True, False, True], kind='stable').copy()
    keep = ordered.groupby('query_id', sort=False).cumcount().lt(5)
    return ordered.loc[keep].reset_index(drop=True)


def independent_rule(frame, column, target):
    ordered = frame.sort_values([column, 'tie_key'], ascending=[False, True], kind='stable')
    count = min(len(ordered), round(frame.query_id.nunique() * target / 100))
    if count == 0:
        return float('inf'), ''
    row = ordered.iloc[count - 1]
    return float(row[column]), str(row.tie_key)


def cached_outcomes(frame, column, rule, ids):
    threshold, tie = float(rule['threshold']), str(rule['tie_cutoff'])
    admitted = frame[column].gt(threshold) | (frame[column].eq(threshold) & frame.tie_key.le(tie))
    tmp = frame[['query_id', 'gold_evidence']].copy()
    tmp['admitted'] = admitted.astype(int)
    tmp['correct'] = tmp.gold_evidence * tmp.admitted
    tmp['has_error'] = (1 - tmp.gold_evidence) * tmp.admitted
    values = tmp.groupby('query_id')[['admitted', 'correct', 'has_error']].sum().reindex(ids, fill_value=0)
    values['has_error'] = values.has_error.gt(0).astype(int)
    values['covered'] = values.admitted.gt(0).astype(int)
    return values


def verify_context(smoke):
    base = ROOT / 'results/r5_context_gain/controls/full'
    table = pd.read_csv(base / 'summary.csv')
    seeds = [13] if smoke else [13, 17, 29, 7]
    counts = {'rows': 0, 'numeric_values': 0, 'candidate_reconstructions': 0,
              'threshold_reconstructions': 0}
    for seed in seeds:
        folder = base / f'seed{seed}'
        evaluation = pd.read_parquet(folder / 'candidate_scores_evaluation.parquet')
        calibration = pd.read_parquet(folder / 'candidate_scores_calibration.parquet')
        evaluation['query_id'] = evaluation.query_id.astype(str)
        calibration['query_id'] = calibration.query_id.astype(str)
        for method, column in [('context_all', 'score_all'), ('context_ce', 'score_ce')]:
            ev, ca = ranked(evaluation, column), ranked(calibration, column)
            rules = pd.read_csv(folder / f'rules_{method}.csv', dtype={'tie_cutoff': str},
                                float_precision='round_trip')
            if smoke:
                rules = rules[rules.analysis.eq('workload_100')]
            for rule in rules.to_dict('records'):
                qp = folder / f"query_{method}_{rule['analysis']}.parquet"
                stats = pd.read_parquet(qp)
                stats['query_id'] = stats.query_id.astype(str)
                row = table[table.seed.eq(seed) & table.method.eq(method) & table.analysis.eq(rule['analysis'])]
                assert len(row) == 1
                counts['numeric_values'] += check_row(stats, row.iloc[0], str(qp.relative_to(ROOT)))
                counts['rows'] += 1
                if rule['analysis'].startswith('workload_'):
                    target = int(rule['analysis'].split('_')[-1])
                    t, tie = independent_rule(ca, column, target)
                    equal(t, rule['threshold'], method + ':calibration_threshold')
                    assert tie == rule['tie_cutoff']
                    counts['threshold_reconstructions'] += 1
                elif rule['analysis'] == 'exact_budget_100':
                    t, tie = independent_rule(ev, column, 100)
                    equal(t, rule['threshold'], method + ':batch_threshold')
                    assert tie == rule['tie_cutoff']
                    counts['threshold_reconstructions'] += 1
                subset = stats.head(12) if smoke else stats
                rebuilt = cached_outcomes(ev[ev.query_id.isin(subset.query_id)], column,
                                          rule, subset.query_id)
                for key in ['admitted', 'correct', 'has_error', 'covered']:
                    assert np.array_equal(rebuilt[key].to_numpy(), subset[key].to_numpy()), (seed, method, rule['analysis'], key)
                counts['candidate_reconstructions'] += len(subset)
    return counts


def verify_other_queries(smoke):
    risk = ROOT / 'results/r5_risk_conflict'
    total_rows = values = 0
    paths = [(risk / 'risk/full/evaluation_by_partition.csv', 'risk'),
             (risk / 'conflict/full/evaluation_by_partition.csv', 'conflict')]
    for path, kind in paths:
        table = pd.read_csv(path)
        if kind == 'conflict':
            table = table[table.method.ne('original')]
        if smoke:
            table = table.head(1)
        for row in table.to_dict('records'):
            suffix = '' if kind == 'risk' else '_' + row['analysis']
            qp = path.parent / f"query_{row['dataset']}_seed{row['seed']}_{row['method']}{suffix}.csv"
            if not qp.exists() and kind == 'conflict':
                qp = path.parent / f"query_{row['dataset']}_seed{row['seed']}_{row['analysis']}_{row['method']}.csv"
            stats = pd.read_csv(qp, dtype={'query_id': str})
            values += check_row(stats, row, str(qp.relative_to(ROOT)))
            total_rows += 1
    evaluation = ROOT / 'results/r5_context_gain/evaluation'
    table = pd.read_csv(evaluation / 'by_partition.csv')
    table = table[table.method.isin(['all', 'ce'])]
    if smoke:
        table = table.head(1)
    for row in table.to_dict('records'):
        qp = evaluation / f"query_seed{row['seed']}_{row['method']}_{row['protocol']}.csv"
        stats = pd.read_csv(qp, dtype={'query_id': str})
        values += check_row(stats, row, str(qp.relative_to(ROOT)))
        total_rows += 1
    return {'rows': total_rows, 'numeric_values': values}


def verify_cp(smoke):
    paths = [ROOT / 'results/r5_risk_conflict/risk/full/calibration_tests.csv']
    extra = ROOT / 'results/r5_context_gain/conditional_risk'
    if extra.exists():
        paths += sorted(extra.glob('full/calibration_tests.csv'))
    checked = 0
    for path in paths:
        rows = pd.read_csv(path)
        if smoke:
            rows = rows.head(3)
        for row in rows.to_dict('records'):
            n, m, e = int(row['queries']), int(row['covered_queries']), int(row['error_queries'])
            alpha = row['one_sided_failure_probability']
            upper = float(beta.ppf(1 - alpha, e + 1, m - e)) if e < m else 1.
            lower = float(beta.ppf(alpha, m, n - m + 1)) if m else 0.
            equal(upper, row['conditional_error_upper'], 'conditional_error_upper')
            equal(lower, row['coverage_lower'], 'coverage_lower')
            assert bool(row['passes']) == bool(m and upper <= .2 and lower >= .05)
            checked += 1
    return checked


def verify_conditional(smoke):
    base = ROOT / 'results/r5_context_gain/conditional_risk/full'
    table = pd.read_csv(base / 'evaluation_by_partition.csv')
    grids = pd.read_csv(base / 'calibration_tests.csv', float_precision='round_trip')
    rules = pd.read_csv(base / 'selected_rules.csv', float_precision='round_trip')
    count = {'evaluation_rows': 0, 'calibration_grid_rows': 0, 'numeric_values': 0}
    seeds = [13] if smoke else [13, 17, 29, 7]
    for seed in seeds:
        caches = ROOT / f'results/r5_context_gain/controls/full/seed{seed}'
        frames = {name: pd.read_parquet(caches / f'candidate_scores_{name}.parquet')
                  for name in ['fitting', 'calibration', 'evaluation']}
        for frame in frames.values():
            frame['query_id'] = frame.query_id.astype(str)
        for method in ['all', 'ce']:
            column = 'score_' + method
            compact = {name: ranked(frame, column) for name, frame in frames.items()}
            selected = rules[rules.seed.eq(seed) & rules.method.eq(method)].iloc[0]
            grid = grids[grids.seed.eq(seed) & grids.method.eq(method)]
            passing = grid[grid.passes]
            assert bool(selected.feasible) == (len(passing) > 0)
            if len(passing):
                best = passing.sort_values(['coverage_lower', 'threshold'], ascending=[False, True]).iloc[0]
                equal(selected.threshold, best.threshold, 'conditional_selected_threshold')
            qp = base / f'query_seed{seed}_{method}.parquet'
            stats = pd.read_parquet(qp)
            stats['query_id'] = stats.query_id.astype(str)
            row = table[table.seed.eq(seed) & table.method.eq(method)].iloc[0]
            count['numeric_values'] += check_row(stats, row, str(qp.relative_to(ROOT)))
            subset = stats.head(12) if smoke else stats
            rule = {'threshold': selected.threshold, 'tie_cutoff': 'ffffffffffffffff'}
            ev = compact['evaluation']
            rebuilt = cached_outcomes(ev[ev.query_id.isin(subset.query_id)], column, rule, subset.query_id)
            for key in ['admitted', 'correct', 'has_error', 'covered']:
                assert np.array_equal(rebuilt[key].to_numpy(), subset[key].to_numpy()), (seed, method, key)
            count['evaluation_rows'] += 1
            fit = compact['fitting'].groupby('query_id', sort=False).head(1)
            cal = compact['calibration']
            ids = cal.query_id.drop_duplicates()
            if smoke:
                grid = grid.head(2)
            for item in grid.to_dict('records'):
                threshold = float(np.quantile(fit[column], item['fitting_quantile']))
                equal(threshold, item['threshold'], 'conditional_fitting_quantile')
                rule = {'threshold': item['threshold'], 'tie_cutoff': 'ffffffffffffffff'}
                rebuilt = cached_outcomes(cal, column, rule, ids)
                assert rebuilt.covered.sum() == item['covered_queries']
                assert rebuilt.has_error.sum() == item['error_queries']
                assert rebuilt.admitted.sum() == item['admitted']
                count['calibration_grid_rows'] += 1
    return count


def verify_refit():
    folder = ROOT / 'results/r5_reproduction/retrained_seed13/full'
    table = pd.read_csv(folder / 'parameter_comparison.csv')
    assert table.within_1e_10.all() and table.exact_equal.all()
    assert table.max_abs_error.fillna(0).max() == 0
    frozen = json.loads((ROOT / 'results/r5_reproduction/fever_seed13_full/summary.json').read_text())
    refit = json.loads((folder / 'summary.json').read_text())
    assert frozen['passed'] and refit['passed']
    return {'parameter_comparison_rows': len(table), 'maximum_parameter_error': 0}


def verify_tables():
    import r5_additional_tables as exporter
    values = exporter.tables()
    manifest = json.loads((ROOT / 'results/r5_strengthening/additional_table_manifest.json').read_text())
    rows = {v['table']: v for v in manifest}
    for name, content in values.items():
        buffer = io.StringIO(newline='')
        csv.writer(buffer, lineterminator='\n').writerows([content['headers']] + content['rows'])
        actual = buffer.getvalue().encode('utf-8')
        path = ROOT / rows[name]['file']
        assert actual == path.read_bytes(), name
        assert hashlib.sha256(actual).hexdigest() == rows[name]['sha256'], name
    return {'tables': len(values), 'csv_bytes_identical': True}


def main(smoke):
    start = time.monotonic()
    if not smoke:
        prior = json.loads((DEST / 'smoke.json').read_text())
        assert prior['passed'] and prior['verifier_sha256'] == digest(__file__)
    result = {'passed': True, 'smoke': smoke, 'verifier_sha256': digest(__file__),
              'manifest_files': verify_manifest(smoke), 'simple_controls': verify_simple(smoke),
              'context_controls': verify_context(smoke), 'other_saved_queries': verify_other_queries(smoke),
              'cp_rows': verify_cp(smoke), 'conditional_context': verify_conditional(smoke),
              'independent_refit': verify_refit(), 'tables': verify_tables()}
    result['elapsed_seconds'] = time.monotonic() - start
    result['scope'] = 'Saved-file integrity, numerical aggregation, context admission from cached scores, independent calibration thresholds, and Clopper-Pearson bounds. No model fitting or pretrained neural inference.'
    dump(DEST / ('smoke.json' if smoke else 'full.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke', action='store_true')
    group.add_argument('--full', action='store_true')
    main(parser.parse_args().smoke)
