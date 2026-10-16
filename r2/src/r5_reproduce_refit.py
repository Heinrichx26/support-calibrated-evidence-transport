"""Refit one FEVER partition from released raw view scores, then verify outputs.

No pretrained model inference. Uses two numerical-library threads and writes only
under results/r5_reproduction/retrained_seed13. Run --smoke before --full.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '2'
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits, threadpool_info

R2 = Path(__file__).resolve().parents[1]
ROOT = R2 / 'results/r5_reproduction'
RESTORE = ROOT / 'restored_public_archive'
DEST = ROOT / 'retrained_seed13'
SEED = 13


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def nested_compare(actual, expected):
    """Compare all fitted fields, reporting exactness and practical tolerances."""
    rows = []
    def visit(a, b, path):
        if isinstance(a, dict):
            assert isinstance(b, dict) and set(a) == set(b), (path, 'field mismatch')
            for key in a:
                visit(a[key], b[key], path + '.' + str(key))
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), (path, 'length mismatch')
            if a and all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in a):
                aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                finite = np.isfinite(aa) & np.isfinite(bb)
                diff = np.abs(aa[finite] - bb[finite])
                rows.append({'field': path, 'elements': len(a), 'exact_equal': bool(np.array_equal(aa, bb)),
                             'max_abs_error': float(diff.max()) if len(diff) else 0.0,
                             'within_1e_10': bool(np.allclose(aa, bb, atol=1e-10, rtol=1e-10, equal_nan=True))})
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f'{path}[{i}]')
        elif isinstance(a, (float, int)) and not isinstance(a, bool):
            rows.append({'field': path, 'elements': 1, 'exact_equal': a == b,
                         'max_abs_error': float(abs(a-b)) if np.isfinite(a) and np.isfinite(b) else 0.0,
                         'within_1e_10': bool(np.isclose(a, b, atol=1e-10, rtol=1e-10, equal_nan=True))})
        else:
            if a != b:
                rows.append({'field': path, 'elements': 1, 'exact_equal': False,
                             'max_abs_error': None, 'within_1e_10': False,
                             'actual': str(a), 'expected': str(b)})
    visit(actual, expected, 'model')
    return rows


def main(smoke):
    started = time.monotonic()
    out = DEST / ('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    code_hash = sha(Path(__file__))
    if not smoke:
        prior = json.loads((DEST / 'smoke/summary.json').read_text(encoding='utf-8'))
        assert prior['passed'] and prior['driver_code_sha256'] == code_hash
    assert json.loads((ROOT / 'smoke/summary.json').read_text())['passed']
    sys.path.insert(0, str(RESTORE / 'src'))
    admission = importlib.import_module('r2_admission')
    stats_module = importlib.import_module('r2_statistics')
    # Independent recomputation helper uses the same restored archive and no core mutations.
    sys.path.insert(0, str(R2 / 'src'))
    helper = importlib.import_module('r5_reproduce_frozen')
    assert Path(admission.__file__).resolve().is_relative_to(RESTORE.resolve())
    original = json.loads((RESTORE / 'results/experiments/fever_full_k100/model_seed13.json').read_text())
    assert original['code_hash'] == sha(RESTORE / 'src/r2_admission.py') == sha(R2 / 'src/r2_admission.py')
    # Reconstruction verifies that model split lists equal the deterministic current split.
    queries = json.loads((RESTORE / 'data/processed/fever_full_queries.json').read_text(encoding='utf-8'))
    calibration, evaluation = admission.split_ids(sorted(queries), admission.PROTOCOL['outer_calibration_fraction'], SEED)
    fitting, threshold = admission.split_ids(calibration, admission.PROTOCOL['fitting_share_of_calibration'], SEED + 1000)
    assert sorted(fitting) == original['fitting_query_ids']
    assert sorted(threshold) == original['threshold_query_ids']
    assert evaluation == original['evaluation_query_ids']
    if smoke:
        fitting, threshold, evaluation = fitting[:60], threshold[:30], evaluation[:12]
    fit = helper.load_frame(fitting)
    cal = helper.load_frame(threshold)
    print('Starting independent fit:', len(fitting), 'queries /', len(fit), 'candidate rows;',
          len(threshold), 'calibration queries; numerical threads = 2', flush=True)
    # Instrument function calls without changing the fitting algorithm or source files.
    original_portfolio = admission.fit_portfolio
    original_quality = admission.selection_quality
    telemetry, fit_counter, quality_counter = [], 0, 0
    def portfolio(*args, **kwargs):
        nonlocal fit_counter
        tick = time.monotonic()
        value = original_portfolio(*args, **kwargs)
        fit_counter += 1
        telemetry.append({'stage': 'portfolio_fit', 'call': fit_counter,
                          'rows': len(args[0]), 'C': args[2], 'seconds': time.monotonic()-tick})
        print('Portfolio fit', fit_counter, 'complete;', round(time.monotonic()-started, 1), 'seconds', flush=True)
        return value
    def quality(*args, **kwargs):
        nonlocal quality_counter
        value = original_quality(*args, **kwargs)
        quality_counter += 1
        if quality_counter % 20 == 0:
            print('Scoring-form evaluations', quality_counter, 'complete;',
                  round(time.monotonic()-started, 1), 'seconds', flush=True)
        return value
    admission.fit_portfolio, admission.selection_quality = portfolio, quality
    try:
        with threadpool_limits(limits=2):
            thread_limits = threadpool_info()
            model = admission.fit_pipeline(fit, cal, SEED)
    finally:
        admission.fit_portfolio, admission.selection_quality = original_portfolio, original_quality
    # JSON round trip normalizes numeric dictionary keys exactly as saved model artifacts.
    model = json.loads(json.dumps(model))
    # Preserve the original model artifact convention for an empty risk rule (Infinity).
    (out / 'refitted_model.json').write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')
    dump(out / 'fit_telemetry.json', telemetry)
    del fit, cal
    if not smoke:
        expected = {key: original[key] for key in model}
        comparisons = nested_compare(model, expected)
        pd.DataFrame(comparisons).to_csv(out / 'parameter_comparison.csv', index=False)
        all_close = all(x['within_1e_10'] for x in comparisons)
        exact = all(x['exact_equal'] for x in comparisons)
        max_error = max(x['max_abs_error'] or 0.0 for x in comparisons)
        differing = [x for x in comparisons if not x['exact_equal']]
    else:
        all_close, exact, max_error, differing = None, None, None, []
    print('Refit selected', model['selected_form'], 'penalty', model['penalty'],
          'single', model['primary_single'], 'elapsed', round(time.monotonic()-started, 1), flush=True)
    invariant_frame = helper.load_frame(evaluation[:12])
    invariance = helper.invariance(admission, invariant_frame, model)
    del invariant_frame
    actual = {}
    matches = []
    score_max_difference = 0.0
    for begin in range(0, len(evaluation), 1000):
        ids = evaluation[begin:begin+1000]
        frame = helper.load_frame(ids)
        scores, stress, _ = admission.score_pipeline(frame, model)
        if not smoke:
            archived_scores, archived_stress, _ = admission.score_pipeline(frame, original)
            assert set(scores) == set(archived_scores)
            for method in scores:
                score_max_difference = max(score_max_difference,
                    float(np.max(np.abs(scores[method]-archived_scores[method]))))
        qpart = {q: queries[q] for q in ids}
        for method, values in scores.items():
            data = stats_module.query_statistics(frame, values, model['thresholds'][method], stress, qpart)
            actual.setdefault(method, []).append(data)
        stats_module._CACHE.clear()
        print('Refitted model applied to', min(begin+len(ids), len(evaluation)), '/', len(evaluation), 'queries', flush=True)
    merged = {}
    for method, pieces in actual.items():
        data = pd.concat(pieces, ignore_index=True)
        merged[method] = data
        data.to_csv(out / f'query_refitted_{method}.csv', index=False)
        if not smoke:
            saved = pd.read_csv(RESTORE / f'results/experiments/fever_full_k100/query_seed13_{method}.csv', dtype={'query_id': str})
            try:
                matches.append(helper.compare_stats(data, saved, method))
            except AssertionError as error:
                matches.append({'method': method, 'queries': len(data), 'all_columns_match': False, 'details': str(error)})
    if not smoke:
        pd.DataFrame(matches).to_csv(out / 'refitted_outcome_comparison.csv', index=False)
        if all(x['all_columns_match'] for x in matches):
            interval = helper.bootstrap(merged, out, False, admission.PROTOCOL)
        else:
            interval = {'published_interval_match': False, 'reason': 'New outcomes differ; preserved for review.'}
    else:
        interval = helper.bootstrap(merged, out, True, admission.PROTOCOL)
    result = {
        'passed': True if smoke else bool(all_close and all(x['all_columns_match'] for x in matches)),
        'mode': 'refit_smoke' if smoke else 'refit_full_fever_seed13',
        'driver_code_sha256': code_hash, 'interpreter': sys.executable,
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                    'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'numerical_threads': thread_limits, 'code_hash_matches_published_and_current': True,
        'splits_reconstructed_from_protocol': True,
        'fitting_queries': len(fitting), 'fitting_candidate_rows': sum(x['rows'] for x in telemetry[-3:]) // 3,
        'threshold_queries': len(threshold), 'evaluation_queries': len(evaluation),
        'portfolio_fit_calls': fit_counter, 'scoring_form_quality_evaluations': quality_counter,
        'selected_form': model['selected_form'], 'conflict_penalty': model['penalty'],
        'primary_single': model['primary_single'],
        'all_model_fields_within_1e_10': all_close, 'all_numeric_fields_bit_exact': exact,
        'maximum_numeric_parameter_difference': max_error,
        'nonexact_numeric_fields': differing,
        'maximum_evaluation_score_difference_from_published_model': score_max_difference if not smoke else None,
        'all_recomputed_query_metrics_match_published': all(x['all_columns_match'] for x in matches) if not smoke else None,
        'query_methods_checked': len(matches), 'invariance': invariance, 'fresh_bootstrap': interval,
        'scope': 'Independent fitting of scale, reliability, inner-fold portfolios, full-fit portfolios, scorer/penalty selection and independent calibration thresholds from published raw view scores; fresh held-out scoring/admission/query metrics/bootstrap.',
        'not_recomputed': ['Original pretrained neural forward inference', 'Other three partition fits and scoring'],
        'elapsed_seconds': time.monotonic()-started,
    }
    dump(out / 'summary.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'mode', 'selected_form', 'conflict_penalty',
        'all_model_fields_within_1e_10', 'all_numeric_fields_bit_exact', 'maximum_numeric_parameter_difference',
        'all_recomputed_query_metrics_match_published', 'elapsed_seconds']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    main(parser.parse_args().smoke)
