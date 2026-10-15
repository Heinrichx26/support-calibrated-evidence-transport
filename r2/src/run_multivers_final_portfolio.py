"""Final development family: nonlinear multi-scorer admission calibration."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_multivers_dual_calibration as dual
import run_multivers_context_calibration as context
import r2_admission as admission
from strengthen_r2_analysis import budget_rule

R2 = Path(__file__).resolve().parents[1]
SPEC_PATH = R2/'data/multivers_portfolio_protocol.json'
SPEC = json.loads(SPEC_PATH.read_text())
OUT = R2/'results/multivers_extension/final_portfolio'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def features(frame, group):
    arrays = [context.features(frame, 'scores'), frame[SPEC['raw_features']].to_numpy(float)]
    if group == 'portfolio_context':
        arrays.append(frame[context.SPEC['context_features']].to_numpy(float))
    result = np.column_stack(arrays)
    assert np.isfinite(result).all()
    return result


def fit_model(train, name):
    if not name.startswith('portfolio'):
        return {'kind': 'control', 'model': dual.fit_model(train, name), 'name': name}
    group, variant = name.split(':')
    x, y = features(train, group), train.gold_evidence.to_numpy(int)
    counts = train.groupby('query_id').size()
    w = train.query_id.map(1/counts).to_numpy(float)
    for label in [0, 1]:
        w[y == label] *= .5/w[y == label].sum()
    w *= len(w)/w.sum()
    classifier = HistGradientBoostingClassifier(**SPEC['classifier'], **SPEC['portfolio_variants'][int(variant)], random_state=SPEC['seed'])
    classifier.fit(x, y, sample_weight=w)
    return {'kind': group, 'classifier': classifier, 'name': name}


def score_model(model, frame):
    if model['kind'] == 'control':
        return dual.score_model(model['model'], frame)
    return model['classifier'].predict_proba(features(frame, model['kind']))[:, 1]


def run(smoke):
    start = time.monotonic()
    source_sha = digest(Path(__file__))
    if not smoke:
        a = json.loads((OUT/'smoke/audit.json').read_text())
        assert a['passed'] and a['source_sha256'] == source_sha and a['protocol_sha256'] == digest(SPEC_PATH)
    out = OUT/('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    frame, families = dual.prepare(), context.source_families()
    oldfreeze = json.loads((dual.OUT/'full/confirmation_freeze.json').read_text())
    fitids, calids = oldfreeze['fitting_query_ids'], oldfreeze['calibration_query_ids']
    fit, cal = frame[frame.query_id.isin(fitids)].reset_index(drop=True), frame[frame.query_id.isin(calids)].reset_index(drop=True)
    groups = sorted({families[q] for q in fitids}, key=lambda g: hashlib.sha256(f"{SPEC['seed']}:context:{g}".encode()).hexdigest())
    mapping = {g: i % 4 for i, g in enumerate(groups)}
    folds = fit.query_id.map(lambda q: mapping[families[q]]).to_numpy()
    names = SPEC['controls']+[f'{group}:{v}' for group in SPEC['feature_groups'] for v in range(len(SPEC['portfolio_variants']))]
    rows = []
    for f in range(2 if smoke else 4):
        train, valid = fit.loc[folds != f].reset_index(drop=True), fit.loc[folds == f].reset_index(drop=True)
        for name in names:
            model = fit_model(train, name)
            s = score_model(model, valid)
            assert np.isfinite(s).all()
            if smoke:
                changed = valid.copy()
                changed['gold_evidence'] = 1-changed.gold_evidence
                changed['claim_verdict'] = 'unused'
                assert np.allclose(score_model(model, changed.iloc[::-1])[::-1], s, atol=1e-12, rtol=0)
            admitted, _ = admission.apply_threshold(valid, s, budget_rule(valid, s, valid.query_id.nunique()))
            assert admitted.sum() == valid.query_id.nunique()
            rows.append({'fold': f, 'method': name, 'precision': float(valid.gold_evidence.to_numpy(int)[admitted].mean())})
        print('fold', f, 'elapsed', round(time.monotonic()-start, 1), flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(out/'inner_cv.csv', index=False)
    cv.groupby('method').precision.agg(['mean', 'std']).sort_values('mean', ascending=False).to_csv(out/'cv_mean_sd.csv')
    mean = cv.groupby('method').precision.mean().to_dict()
    key = lambda n: (round(mean[n], 12), -names.index(n))
    chosen, control = max(names, key=key), max(SPEC['controls'], key=key)
    freeze = {'chosen': chosen, 'best_control': control, 'fitting_query_ids': fitids, 'calibration_query_ids': calids,
              'source_sha256': source_sha, 'protocol_sha256': digest(SPEC_PATH),
              'dependency_sources_sha256': {Path(m.__file__).name: digest(Path(m.__file__)) for m in [dual, context]},
              'confirmation_inputs_sha256': digest(R2/'data/multivers_confirmation/candidates.parquet'),
              'confirmation_performance_accessed': False, 'smoke': smoke, 'frozen_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'models': {}}
    if not smoke:
        comparisons = [('rationale', 'rationale'), ('product', 'product'), ('full_calibration', 'full:histogram:10'),
                       ('dual_blend', 'blend_rationale:0.75'), ('best_control', control), ('selected', chosen)]
        for label, name in comparisons:
            model = fit_model(fit, name)
            path = out/f'confirmation_model_{label}.joblib'
            joblib.dump(model, path)
            freeze['models'][label] = {'name': name, 'path_name': path.name, 'sha256': digest(path),
                                      'prospective_rule': admission.fit_threshold(cal, score_model(model, cal), 100)}
    (out/'confirmation_freeze.json').write_text(json.dumps(freeze, indent=2))
    (out/'audit.json').write_text(json.dumps({'passed': True, 'smoke': smoke, 'source_sha256': source_sha,
        'protocol_sha256': digest(SPEC_PATH), 'outcome_row_order_invariance_smoke': smoke, 'elapsed_seconds': time.monotonic()-start}, indent=2))
    print('chosen', chosen, 'control', control, flush=True)
    print(cv.groupby('method').precision.mean().sort_values(ascending=False).to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
