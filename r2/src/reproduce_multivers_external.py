"""Replay frozen external predictions, selections, thresholds and intervals.

This verification writes to a separate results directory and preserves the
historical study specifications, freezes, outcomes and one-shot test marker.
No neural checkpoint is required. Restore provider data with
fetch_multivers_external.py before query-level evaluation or fitting.
"""
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
import evaluate_multivers_external as original
import run_multivers_adaptive_calibration as adaptive
import run_multivers_admission_calibration as calibrated
from infer_multivers_local_context import screens
from evaluate_multivers_confirmation import reselected_precision

R2 = Path(__file__).resolve().parents[1]
ROOT = R2 / 'results/multivers_extension/external_evaluation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_and_freezes():
    frame = pd.read_parquet(R2 / 'data/multivers_external/probabilities.parquet')
    assert not frame.duplicated(['query_id', 'candidate_id']).any()
    freezes = {d: json.loads((ROOT / d / 'freeze.json').read_text())
               for d in ['healthver', 'covidfact']}
    for dataset, freeze in freezes.items():
        assert freeze['code_sha256'] == sha(Path(original.__file__))
        assert freeze['test_outcomes_accessed'] is False
        for info in freeze['models'].values():
            assert sha(ROOT / dataset / info['path']) == info['sha256']
    return frame, freezes


def smoke(out):
    frame, freezes = frame_and_freezes()
    for dataset, freeze in freezes.items():
        part = frame[(frame.dataset == dataset) & (frame.split == 'test')].copy().reset_index(drop=True)
        part = part[screens(part)].reset_index(drop=True)
        ids = sorted(part.query_id.unique())[:12]
        part = part[part.query_id.isin(ids)].reset_index(drop=True)
        for label, info in freeze['models'].items():
            model = joblib.load(ROOT / dataset / info['path'])
            score = calibrated.score(model, part)
            changed = part.copy()
            changed['gold_evidence'] = np.arange(len(changed)) % 2
            assert np.allclose(score, calibrated.score(model, changed.iloc[::-1])[::-1], rtol=0, atol=1e-12)
            saved = pd.read_parquet(ROOT / dataset / f'scores_{label}.parquet')
            check = part[['query_id', 'candidate_id']].merge(saved, validate='one_to_one', how='left')
            assert np.allclose(score, check.score, rtol=0, atol=1e-12), (dataset, label)
            rule = budget_rule(part, score, len(ids))
            mask, _ = admission.apply_threshold(part, score, rule)
            assert mask.sum() == len(ids)
    record = {'passed': True, 'scores_match_frozen_outputs': True,
              'outcome_and_row_order_independent': True, 'exact_budget_verified': True,
              'source_sha256': sha(Path(__file__))}
    (out / 'smoke.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)


def replay(out, repetitions=2000):
    frame, freezes = frame_and_freezes()
    summaries, intervals = [], []
    for dataset, freeze in freezes.items():
        queries = original.label_data(dataset, 'test')
        part = original.with_labels(frame, queries)
        part = part[screens(part)].reset_index(drop=True)
        families = sorted({q['source_family'] for q in queries.values()})
        lookup = {f: i for i, f in enumerate(families)}
        rng = np.random.default_rng(2026101584)
        draws = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=repetitions)
        boot, point = {}, {}
        for label, info in freeze['models'].items():
            score = calibrated.score(joblib.load(ROOT / dataset / info['path']), part)
            saved = pd.read_parquet(ROOT / dataset / f'scores_{label}.parquet')
            check = part[['query_id', 'candidate_id']].merge(saved, validate='one_to_one', how='left')
            assert np.allclose(score, check.score, rtol=0, atol=1e-12), (dataset, label)
            for analysis, rule in [('exact_budget', budget_rule(part, score, len(queries))),
                                   ('prospective', info['prospective_rule'])]:
                stats = query_statistics(part, score, rule, np.zeros(len(part), bool), queries)
                saved_stats = pd.read_csv(ROOT / dataset / f'query_{label}_{analysis}.csv')
                pd.testing.assert_frame_equal(stats, saved_stats, check_dtype=False, atol=1e-12, rtol=0)
                destination = out / dataset
                destination.mkdir(exist_ok=True)
                stats.to_csv(destination / f'query_{label}_{analysis}.csv', index=False)
                summaries.append({'dataset': dataset, 'method': label, 'selected_form': info['name'],
                                  'analysis': analysis, 'source_groups': len(families), **admission.summarize(stats)})
                if analysis == 'exact_budget':
                    assert stats.admitted.sum() == len(queries)
                    point[label] = stats.correct.sum() / stats.admitted.sum()
                    boot[label] = reselected_precision(part, score, queries, draws, lookup)
        for reference in ['best_single', 'best_blend', 'full_raw', 'full_product', 'single_raw',
                          'matched_full', 'without_context', 'without_focus']:
            delta = boot['selected'] - boot[reference]
            lo, hi = np.quantile(delta, [.025, .975])
            slo, shi = np.quantile(delta, [.00625, .99375])
            intervals.append({'dataset': dataset, 'reference': reference,
                              'delta_precision': point['selected'] - point[reference],
                              'ci95_low': lo, 'ci95_high': hi,
                              'simultaneous_low': slo, 'simultaneous_high': shi,
                              'primary': reference in ['best_single', 'best_blend'], 'source_groups': len(families)})
    summary, interval = pd.DataFrame(summaries), pd.DataFrame(intervals)
    pd.testing.assert_frame_equal(summary, pd.read_csv(ROOT / 'summary.csv'), check_dtype=False, atol=1e-12, rtol=0)
    if repetitions == 2000:
        pd.testing.assert_frame_equal(interval, pd.read_csv(ROOT / 'paired_intervals.csv'), check_dtype=False, atol=1e-12, rtol=0)
    summary.to_csv(out / 'summary.csv', index=False)
    interval.to_csv(out / 'paired_intervals.csv', index=False)
    (out / 'replay.json').write_text(json.dumps({'passed': True, 'models_retrained': False,
        'all_36_query_outputs_match': True, 'bootstrap_repetitions': repetitions,
        'intervals_match': repetitions == 2000, 'historical_records_preserved': True}, indent=2), encoding='utf-8')
    print('Frozen external outcomes and intervals reproduced.', flush=True)


def fitting_data(frame, dataset, freeze):
    fitting_queries = original.label_data(dataset, 'train')
    calibration_queries = original.label_data(dataset, 'dev') if dataset == 'healthver' else fitting_queries
    train = {q: fitting_queries[q] for q in freeze['fitting_ids']}
    calibration = {q: calibration_queries[q] for q in freeze['calibration_ids']}
    train_all = original.with_labels(frame, train)
    cal_all = original.with_labels(frame, calibration)
    return train_all, train_all[screens(train_all)].reset_index(drop=True), cal_all[screens(cal_all)].reset_index(drop=True)


def refit(out, selection=False):
    frame, freezes = frame_and_freezes()
    for dataset, freeze in freezes.items():
        all_train, train, cal = fitting_data(frame, dataset, freeze)
        destination = out / dataset
        destination.mkdir(exist_ok=True)
        if selection:
            folds, rows = adaptive.folds(train), []
            for k in range(4):
                a, b = train[folds != k].reset_index(drop=True), train[folds == k].reset_index(drop=True)
                assert not set(a.family) & set(b.family)
                for method in calibrated.candidates():
                    fitted = calibrated.fit(a, method)
                    quality = adaptive.quality(b, calibrated.score(fitted, b))[0]
                    rows.append({'fold': k, 'method': method, **quality})
                print(dataset, 'fitting fold', k, 'reproduced', flush=True)
            cv = pd.DataFrame(rows)
            pd.testing.assert_frame_equal(cv, pd.read_csv(ROOT / dataset / 'train_cv.csv'), check_dtype=False, atol=1e-12, rtol=0)
            cv.to_csv(destination / 'train_cv.csv', index=False)
        for label, info in freeze['models'].items():
            fitted = calibrated.fit(all_train if label == 'without_focus' else train, info['name'])
            frozen = joblib.load(ROOT / dataset / info['path'])
            for part in [train, cal]:
                assert np.allclose(calibrated.score(fitted, part), calibrated.score(frozen, part), rtol=0, atol=1e-12)
            rule = admission.fit_threshold(cal, calibrated.score(fitted, cal), 100)
            assert rule == info['prospective_rule'], (dataset, label)
            joblib.dump(fitted, destination / f'{label}.joblib')
        print(dataset, 'frozen fits and thresholds reproduced', flush=True)
    (out / 'refit.json').write_text(json.dumps({'passed': True, 'selection_recomputed': selection,
        'thresholds_match': True, 'test_outcomes_used_for_fitting': False}, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--refit', action='store_true', help='Reproduce all nine frozen calibrators and thresholds per dataset')
    parser.add_argument('--selection', action='store_true', help='Also reproduce every training-fold candidate comparison')
    parser.add_argument('--output', type=Path, default=R2 / 'results/external_reproduction')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        if args.smoke:
            smoke(args.output)
        else:
            gate = json.loads((args.output / 'smoke.json').read_text())
            assert gate['passed'] and gate['source_sha256'] == sha(Path(__file__))
            if args.refit or args.selection:
                refit(args.output, selection=args.selection)
            else:
                replay(args.output)


if __name__ == '__main__':
    main()
