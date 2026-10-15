"""Cross-fitted calibration of full-abstract and isolated-sentence predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import r2_admission as admission
from strengthen_r2_analysis import budget_rule
import run_multivers_context_calibration as context
from run_multivers_probability_fusion import prepare_frame

R2 = Path(__file__).resolve().parents[1]
SPEC_PATH = R2/'data/multivers_dual_protocol.json'
SPEC = json.loads(SPEC_PATH.read_text())
OUT = R2/'results/multivers_extension/dual_calibration'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    frame = context.add_context(prepare_frame(), R2/'results/strengthening/multivers/document_scores.jsonl')
    rows = []
    for line in (R2/'results/multivers_extension/inference_single/scores.jsonl').read_text().splitlines():
        r = json.loads(line)
        p = r['label_probabilities']
        rows.append({'query_id': r['query_id'], 'candidate_id': r['candidate_id'],
                     'single_rationale': r['rationale_probability'], 'single_product': r['raw_multivers'],
                     'single_evidence': 1-p[1], 'single_support': p[2], 'single_refute': p[0]})
    return frame.merge(pd.DataFrame(rows), on=['query_id', 'candidate_id'], validate='one_to_one', sort=False)


def feature_array(frame, group):
    full = context.features(frame, 'scores')
    def lp(x):
        return logit(np.clip(np.asarray(x, float), 1e-6, 1-1e-6))
    evidence = np.maximum(frame.single_evidence.to_numpy(float), 1e-6)
    single = np.column_stack([lp(frame.single_rationale), lp(evidence),
                              np.abs(frame.single_support-frame.single_refute)/evidence])
    if group == 'full':
        return full
    if group == 'isolated':
        return single
    fe = np.maximum(frame.mv_document_evidence.to_numpy(float), 1e-6)
    direction_agreement = (frame.mv_support*frame.single_support+frame.mv_refute*frame.single_refute).to_numpy(float)/(fe*evidence)
    extra = np.column_stack([direction_agreement, full[:, 0]-single[:, 0], full[:, 1]-single[:, 1]])
    arrays = [full, single, extra]
    if group == 'dual_context':
        arrays.append(frame[context.SPEC['context_features']].to_numpy(float))
    result = np.column_stack(arrays)
    assert np.isfinite(result).all()
    return result


def names():
    result = ['rationale', 'product', 'single_rationale', 'single_product']
    for group in SPEC['groups']:
        result.extend(f'{group}:logistic:{c:g}' for c in SPEC['logistic_C'])
        result.extend(f'{group}:histogram:{l:g}' for l in SPEC['histogram_l2'])
    for w in SPEC['simple_blend_weights_full']:
        result.extend([f'blend_rationale:{w:g}', f'geometric_sufficiency:{w:g}'])
    return result


def fit_model(train, name):
    if name.count(':') < 2:
        return {'name': name}
    group, kind, value = name.split(':')
    x, y = feature_array(train, group), train.gold_evidence.to_numpy(int)
    counts = train.groupby('query_id').size()
    w = train.query_id.map(1/counts).to_numpy(float)
    for label in [0, 1]:
        w[y == label] *= .5/w[y == label].sum()
    w *= len(w)/w.sum()
    scale = StandardScaler().fit(x) if kind == 'logistic' else None
    if kind == 'logistic':
        classifier = LogisticRegression(C=float(value), max_iter=1000, random_state=SPEC['seed'])
        classifier.fit(scale.transform(x), y, sample_weight=w)
    else:
        classifier = HistGradientBoostingClassifier(**SPEC['histogram'], l2_regularization=float(value), random_state=SPEC['seed'])
        classifier.fit(x, y, sample_weight=w)
    return {'name': name, 'group': group, 'classifier': classifier, 'scale': scale}


def score_model(model, frame):
    name = model['name']
    if name == 'rationale':
        return frame.raw_multivers_rationale.to_numpy(float)
    if name == 'product':
        return frame.raw_multivers.to_numpy(float)
    if name in ['single_rationale', 'single_product']:
        return frame[name].to_numpy(float)
    if name.startswith('blend_rationale:'):
        w = float(name.split(':')[1])
        return w*frame.raw_multivers_rationale.to_numpy(float)+(1-w)*frame.single_rationale.to_numpy(float)
    if name.startswith('geometric_sufficiency:'):
        w = float(name.split(':')[1])
        return np.clip(frame.raw_multivers_rationale.to_numpy(float), 1e-8, 1)**w*np.clip(frame.single_evidence.to_numpy(float), 1e-8, 1)**(1-w)
    x = feature_array(frame, model['group'])
    if model['scale'] is not None:
        x = model['scale'].transform(x)
    return model['classifier'].predict_proba(x)[:, 1]


def run(smoke):
    start = time.monotonic()
    code_sha = digest(Path(__file__))
    if not smoke:
        a = json.loads((OUT/'smoke/audit.json').read_text())
        assert a['passed'] and a['protocol_sha256'] == digest(SPEC_PATH) and a['code_sha256'] == code_sha
    out = OUT/('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    frame, families = prepare(), context.source_families()
    calids = [q for q in families if int(hashlib.sha256(f"{SPEC['seed']}:calibration:{families[q]}".encode()).hexdigest(), 16) % 5 == 0]
    fitids = sorted(set(families)-set(calids))
    fit, cal = frame[frame.query_id.isin(fitids)].reset_index(drop=True), frame[frame.query_id.isin(calids)].reset_index(drop=True)
    groups = sorted({families[q] for q in fitids}, key=lambda g: hashlib.sha256(f"{SPEC['seed']}:context:{g}".encode()).hexdigest())
    mapping = {g: i % SPEC['inner_folds'] for i, g in enumerate(groups)}
    folds = fit.query_id.map(lambda q: mapping[families[q]]).to_numpy()
    rows = []
    for f in range(2 if smoke else SPEC['inner_folds']):
        train, valid = fit.loc[folds != f].reset_index(drop=True), fit.loc[folds == f].reset_index(drop=True)
        for name in names():
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
    means = cv.groupby('method').precision.mean().to_dict()
    order = names()
    controls = [n for n in order if n.count(':') == 0 or n.startswith('full:') or n.startswith('isolated:')]
    key = lambda n: (round(means[n], 12), -order.index(n))
    chosen, control = max(order, key=key), max(controls, key=key)
    cv.groupby('method').precision.agg(['mean', 'std']).sort_values('mean', ascending=False).to_csv(out/'cv_mean_sd.csv')
    freeze = {'chosen': chosen, 'single_context_control': control, 'fitting_query_ids': fitids, 'calibration_query_ids': calids,
              'protocol_sha256': digest(SPEC_PATH), 'source_sha256': code_sha,
              'context_source_sha256': digest(Path(context.__file__)),
              'confirmation_inputs_sha256': digest(R2/'data/multivers_confirmation/candidates.parquet'),
              'confirmation_performance_accessed': False, 'smoke': smoke,
              'frozen_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'models': {}}
    if not smoke:
        for label, name in [('rationale', 'rationale'), ('product', 'product'), ('single_rationale', 'single_rationale'),
                            ('single_product', 'single_product'), ('single_context_control', control), ('selected', chosen)]:
            model = fit_model(fit, name)
            path = out/f'confirmation_model_{label}.joblib'
            joblib.dump(model, path)
            freeze['models'][label] = {'name': name, 'path_name': path.name, 'sha256': digest(path),
                                      'prospective_rule': admission.fit_threshold(cal, score_model(model, cal), 100)}
    (out/'confirmation_freeze.json').write_text(json.dumps(freeze, indent=2))
    (out/'audit.json').write_text(json.dumps({'passed': True, 'smoke': smoke, 'protocol_sha256': digest(SPEC_PATH),
        'code_sha256': code_sha, 'row_order_and_outcome_invariance': smoke, 'exact_counts': True, 'elapsed_seconds': time.monotonic()-start}, indent=2))
    print('chosen', chosen, 'single context control', control, flush=True)
    print(cv.groupby('method').precision.mean().sort_values(ascending=False).head(12).to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
