"""Independent archive-based frozen admission recomputation; no neural inference or refitting.

Run --smoke first, inspect its result, then --full. Source archives remain read-only.
All restored inputs and new outputs stay below results/r5_reproduction/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from zipfile import ZipFile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
import sklearn

R2 = Path(__file__).resolve().parents[1]
DEST = R2 / 'results/r5_reproduction'
RESTORE = DEST / 'restored_public_archive'
ARCHIVE = R2 / 'results/R2_public_reproducibility.zip'
ARCHIVE_SHA256 = '7e50b281ba52c6d729f499514062c991d0fe1774885f67561eedcdfff3ad1762'
ROOT = 'results/experiments/fever_full_k100'
SEEDS = [13, 17, 29, 7]
SEED = 13


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def restore():
    DEST.mkdir(parents=True, exist_ok=True)
    assert sha(ARCHIVE) == ARCHIVE_SHA256
    required = [
        'src/r2_admission.py', 'src/r2_statistics.py', 'src/run_r2_experiments.py',
        'data/r2_protocol.json', 'data/scored/fever_full_k100.parquet',
        'data/scored/fever_full_k100.json', 'data/processed/fever_full_queries.json',
        'results/reproduction_environment.json',
        ROOT + '/paired_bootstrap.csv', ROOT + '/primary_comparison.json',
    ]
    for seed in SEEDS:
        required += [f'{ROOT}/model_seed{seed}.json', f'{ROOT}/query_seed{seed}_scet.csv',
                     f'{ROOT}/query_seed{seed}_ce.csv']
    with ZipFile(ARCHIVE) as archive:
        manifest = {v['path']: v for v in json.loads(archive.read('results/public_manifest.json'))}
        model = json.loads(archive.read(f'{ROOT}/model_seed{SEED}.json'))
        required += [i.filename for i in archive.infolist()
                     if i.filename.startswith(f'{ROOT}/query_seed{SEED}_')]
        verified = []
        for name in sorted(set(required)):
            path = RESTORE / name
            expected = manifest[name]['sha256']
            if not path.exists() or sha(path) != expected:
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target, 2 ** 20)
            assert sha(path) == expected, name
            verified.append({'path': name, 'sha256': expected, 'bytes': path.stat().st_size})
    dump(DEST / 'restored_input_hashes.json', {
        'archive': ARCHIVE.name, 'archive_sha256': ARCHIVE_SHA256,
        'source': 'Local file whose digest identifies the published release asset; no network download in this run.',
        'members': verified,
    })
    return model


def load_frame(ids):
    return pd.read_parquet(RESTORE / 'data/scored/fever_full_k100.parquet',
                           filters=[('query_id', 'in', ids)]).reset_index(drop=True)


def compare_stats(actual, saved, name):
    actual = actual.set_index('query_id').sort_index()
    saved = saved.set_index('query_id').reindex(actual.index)
    assert actual.columns.tolist() == saved.columns.tolist(), name
    mismatch = []
    max_float_error = 0.0
    for col in actual.columns:
        a, b = actual[col], saved[col]
        if pd.api.types.is_numeric_dtype(a):
            delta = np.abs(a.to_numpy(float) - b.to_numpy(float))
            if delta.size:
                max_float_error = max(max_float_error, float(np.nanmax(delta)))
            if not np.allclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True):
                mismatch.append(col)
        elif not (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all():
            mismatch.append(col)
    assert not mismatch, (name, mismatch)
    return {'method': name, 'queries': len(actual), 'all_columns_match': True,
            'maximum_absolute_numeric_difference': max_float_error}


def invariance(admission, frame, model):
    scores, stress, conflict = admission.score_pipeline(frame, model)
    changed = frame.copy()
    changed['gold_evidence'] = 1 - changed.gold_evidence
    changed['gold_directional'] = 1 - changed.gold_directional
    changed['claim_verdict'] = 'PERMUTED_OUTCOME'
    alt, alt_stress, alt_conflict = admission.score_pipeline(changed, model)
    assert all(np.array_equal(v, alt[k]) for k, v in scores.items())
    assert np.array_equal(stress, alt_stress) and np.array_equal(conflict, alt_conflict)
    shuffled = frame.sample(frac=1, random_state=20261016).copy()
    original_index = shuffled.index.to_numpy()
    shuffled = shuffled.reset_index(drop=True)
    changed_scores, changed_stress, changed_conflict = admission.score_pipeline(shuffled, model)
    assert np.array_equal(changed_conflict, conflict[original_index])
    for name, values in scores.items():
        assert np.array_equal(changed_scores[name], values[original_index])
        a, rank = admission.apply_threshold(frame, values, model['thresholds'][name])
        b, rank2 = admission.apply_threshold(shuffled, changed_scores[name], model['thresholds'][name])
        assert np.array_equal(b, a[original_index]) and np.array_equal(rank2, rank[original_index])
    return {'queries': int(frame.query_id.nunique()), 'candidate_rows': len(frame),
            'methods': len(scores), 'labels_and_verdict_score_invariant': True,
            'labels_and_verdict_conflict_invariant': True, 'shuffled_scores_ranks_admission_invariant': True}


def bootstrap(actual, out, smoke, protocol):
    """Fresh arithmetic from query outcomes; recalculated seed13 replaces its saved outcome."""
    stats = {}
    use_seeds = [SEED] if smoke else SEEDS
    for seed in use_seeds:
        pair = {}
        for name in ['scet', 'ce']:
            pair[name] = (actual[name].copy() if seed == SEED else
                pd.read_csv(RESTORE / f'{ROOT}/query_seed{seed}_{name}.csv', dtype={'query_id': str}))
            pair[name] = pair[name].set_index('query_id')
        pair['ce'] = pair['ce'].reindex(pair['scet'].index)
        stats[seed] = pair
    ids = sorted(set().union(*(set(v['scet'].index) for v in stats.values())))
    positions = {q: j for j, q in enumerate(ids)}
    repetitions = 100 if smoke else protocol['bootstrap_repetitions']
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    draws = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=repetitions).astype(np.float32)
    all_differences, point = [], []
    for seed, pair in stats.items():
        a, b = pair['scet'], pair['ce']
        w = draws[:, [positions[q] for q in a.index]]
        aa = w @ a[['correct', 'admitted']].to_numpy(float)
        bb = w @ b[['correct', 'admitted']].to_numpy(float)
        all_differences.append(aa[:, 0] / np.maximum(aa[:, 1], 1) - bb[:, 0] / np.maximum(bb[:, 1], 1))
        point.append(a.correct.sum() / a.admitted.sum() - b.correct.sum() / b.admitted.sum())
    dist = np.mean(all_differences, axis=0)
    lo, hi = np.quantile(dist, [.025, .975])
    report = {'comparator': 'ce', 'delta_precision_mean': float(np.mean(point)),
              'ci_low': float(lo), 'ci_high': float(hi), 'repetitions': repetitions,
              'unique_query_clusters': len(ids), 'partitions': len(use_seeds),
              'recalculated_partition': SEED,
              'remaining_partitions': 'None' if smoke else 'Saved published query outcomes; not independently rescored in this run.',
              'conditional_on_frozen_models_and_observed_partitions': True}
    if not smoke:
        saved = pd.read_csv(RESTORE / ROOT / 'paired_bootstrap.csv')
        saved = saved[saved.comparator == 'ce'].iloc[0]
        for col in ['delta_precision_mean', 'ci_low', 'ci_high']:
            assert np.isclose(report[col], saved[col], atol=1e-12, rtol=1e-12), (col, report[col], saved[col])
        report['published_four_partition_bootstrap_matches'] = True
        # Also perform a completely recomputed, single-partition interval.
        a, b = stats[SEED]['scet'], stats[SEED]['ce']
        rng = np.random.default_rng(protocol['bootstrap_seed'])
        d = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=repetitions).astype(np.float32)
        aa, bb = d @ a[['correct', 'admitted']].to_numpy(float), d @ b[['correct', 'admitted']].to_numpy(float)
        delta = aa[:, 0] / np.maximum(aa[:, 1], 1) - bb[:, 0] / np.maximum(bb[:, 1], 1)
        report['independently_recomputed_seed13_interval'] = {
            'point': float(point[0]), 'ci_low': float(np.quantile(delta, .025)),
            'ci_high': float(np.quantile(delta, .975)), 'queries': len(a), 'repetitions': repetitions}
    dump(out / 'fresh_bootstrap.json', report)
    return report


def main(smoke):
    started = time.monotonic()
    out = DEST / ('smoke' if smoke else 'fever_seed13_full')
    out.mkdir(parents=True, exist_ok=True)
    code_hash = sha(Path(__file__))
    if not smoke:
        prior = json.loads((DEST / 'smoke/summary.json').read_text(encoding='utf-8'))
        assert prior['passed'] and prior['driver_code_sha256'] == code_hash, 'Run current smoke first.'
    model = restore()
    sys.path.insert(0, str(RESTORE / 'src'))
    admission = importlib.import_module('r2_admission')
    statistics = importlib.import_module('r2_statistics')
    assert Path(admission.__file__).resolve().is_relative_to(RESTORE.resolve())
    assert model['code_hash'] == sha(RESTORE / 'src/r2_admission.py') == sha(R2 / 'src/r2_admission.py')
    queries = json.loads((RESTORE / 'data/processed/fever_full_queries.json').read_text(encoding='utf-8'))
    fit_ids, cal_ids, test_ids = [model[v] for v in ['fitting_query_ids', 'threshold_query_ids', 'evaluation_query_ids']]
    assert set(fit_ids).isdisjoint(cal_ids) and set(test_ids).isdisjoint(fit_ids + cal_ids)
    # Recompute every frozen threshold from the complete independent calibration split.
    print('Recomputing calibration thresholds', len(cal_ids), 'queries', flush=True)
    cal = load_frame(cal_ids)
    calibration_scores, _, _ = admission.score_pipeline(cal, model)
    thresholds = []
    for name, values in calibration_scores.items():
        if name == 'scet_risk_control':
            continue
        rule = admission.fit_threshold(cal, values, model['target'])
        assert rule == model['thresholds'][name], (name, rule, model['thresholds'][name])
        thresholds.append({'method': name, **rule, 'matches_published_frozen_threshold': True})
    pd.DataFrame(thresholds).to_csv(out / 'recalculated_thresholds.csv', index=False)
    del cal, calibration_scores
    wanted = test_ids[:12] if smoke else test_ids
    parts = {}
    invariant_record = None
    candidates_checked = 0
    for start in range(0, len(wanted), 1000):
        ids = wanted[start:start + 1000]
        frame = load_frame(ids)
        scores, stress, conflict = admission.score_pipeline(frame, model)
        qpart = {q: queries[q] for q in ids}
        if invariant_record is None:
            check_ids = ids[:12]
            check = frame[frame.query_id.isin(check_ids)].copy().reset_index(drop=True)
            invariant_record = invariance(admission, check, model)
        for name, values in scores.items():
            s = statistics.query_statistics(frame, values, model['thresholds'][name], stress, qpart)
            parts.setdefault(name, []).append(s)
        for name in ['ce', 'bm25', 'scet']:
            rule = {'threshold': float('-inf'), 'tie_cutoff': '', 'eligible_rank': 1}
            s = statistics.query_statistics(frame, scores[name], rule, stress, qpart)
            parts.setdefault(name + '_top1', []).append(s)
        candidates_checked += len(frame)
        print('Recomputed', min(start + len(ids), len(wanted)), '/', len(wanted),
              'queries;', candidates_checked, 'candidate rows;', round(time.monotonic() - started, 1), 'seconds', flush=True)
        statistics._CACHE.clear()
        del frame, scores, stress, conflict
    actual, comparisons = {}, []
    for name, segments in parts.items():
        data = pd.concat(segments, ignore_index=True)
        actual[name] = data
        saved = pd.read_csv(RESTORE / f'{ROOT}/query_seed{SEED}_{name}.csv', dtype={'query_id': str})
        comparisons.append(compare_stats(data, saved, name))
        data.to_csv(out / f'query_recomputed_{name}.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'saved_outcome_comparison.csv', index=False)
    interval = bootstrap(actual, out, smoke, admission.PROTOCOL)
    # Stream the entire input using the published pandas content hash rule.
    input_content_digest = hashlib.sha256()
    for batch in pq.ParquetFile(RESTORE / 'data/scored/fever_full_k100.parquet').iter_batches(batch_size=25000):
        frame = batch.to_pandas()
        input_content_digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    content_hash = input_content_digest.hexdigest()
    assert content_hash == model['input_hash'], (content_hash, model['input_hash'])
    result = {
        'passed': True, 'mode': 'smoke' if smoke else 'full_fever_seed13',
        'driver_code_sha256': code_hash, 'interpreter': sys.executable,
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                    'scipy': scipy.__version__, 'pyarrow': pyarrow.__version__, 'sklearn': sklearn.__version__},
        'archive_sha256': ARCHIVE_SHA256, 'imported_frozen_scoring_program': str(Path(admission.__file__).relative_to(DEST)),
        'current_program_hash_equals_archive_and_model': True,
        'scored_input_file_sha256': sha(RESTORE / 'data/scored/fever_full_k100.parquet'),
        'input_content_hash_matches_fitted_model': True, 'input_content_sha256': content_hash,
        'fitting_queries': len(fit_ids), 'threshold_queries': len(cal_ids),
        'independent_query_splits': True, 'thresholds_recomputed_and_matched': len(thresholds),
        'evaluation_queries_recomputed': len(wanted), 'candidate_rows_recomputed': candidates_checked,
        'methods_query_metrics_recomputed_and_matched': len(actual), 'invariance': invariant_record,
        'bootstrap': interval,
        'scope': 'Published raw neural/lexical view scores -> frozen transformations, conflict and portfolio scoring -> independent calibration thresholds -> admission and ranks -> all query metrics -> fresh bootstrap.',
        'not_recomputed': ['Raw neural model forward inference', 'Fitting transformations/portfolio parameters',
                           'Other three partitions scoring and query outcomes'],
        'elapsed_seconds': time.monotonic() - started,
    }
    dump(out / 'summary.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'mode', 'evaluation_queries_recomputed',
          'candidate_rows_recomputed', 'thresholds_recomputed_and_matched',
          'methods_query_metrics_recomputed_and_matched', 'elapsed_seconds']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    main(parser.parse_args().smoke)
