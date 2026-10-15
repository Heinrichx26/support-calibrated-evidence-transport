"""Context-dependent evidence calibration with a separately tuned score control."""
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
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
from run_multivers_probability_fusion import prepare_frame
from run_multivers_extension import paired_intervals

R2 = Path(__file__).resolve().parents[1]
SPEC_PATH = R2/'data/multivers_context_protocol.json'
SPEC = json.loads(SPEC_PATH.read_text())
OUT = R2/'results/multivers_extension/context_calibration'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_families():
    path = R2.parent/'data/raw_open/evidence_retrieval/scifact/claims_dev.jsonl'
    claims = [json.loads(l) for l in path.read_text().splitlines()]
    parent = {str(c['id']): str(c['id']) for c in claims}
    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q
    owners = {}
    for c in claims:
        q = str(c['id'])
        for d in sorted(set(map(str, c['cited_doc_ids'])) | set(c['evidence'])):
            if d in owners:
                parent[find(q)] = find(owners[d])
            else:
                owners[d] = q
    return {q: find(q) for q in parent}


def add_context(frame, document_path):
    frame = frame.copy().reset_index(drop=True)
    doc = {}
    for line in document_path.read_text().splitlines():
        r = json.loads(line)
        p = np.asarray(r['rationale_probabilities'], float)
        v = np.clip(p, 1e-6, 1-1e-6)
        entropy = float(np.mean(-v*np.log(v)-(1-v)*np.log1p(-v)))
        for sid, value in enumerate(p):
            other = np.delete(p, sid)
            doc[(r['query_id'], r['doc_id'], str(sid))] = [np.log1p(len(p)), sid/max(len(p)-1, 1),
                np.log1p(max(0, p.sum()-value)), float(other.max()) if len(other) else 0., entropy]
    metadata = np.array([doc[(q, d, s)] for q, d, s in zip(frame.query_id, frame.doc_id, frame.sentence_id)])
    for j, name in enumerate(SPEC['context_features'][:5]):
        frame[name] = metadata[:, j]
    p = frame.raw_multivers_rationale.to_numpy(float)
    frame['log_query_candidate_rank'] = np.log(admission.ranks(frame, p))
    ordered = frame.sort_values(['query_id', 'doc_id', 'raw_multivers_rationale', 'tie_key'], ascending=[True, True, False, True])
    dr = np.empty(len(frame), dtype=int)
    dr[ordered.index] = (ordered.groupby(['query_id', 'doc_id']).cumcount()+1).to_numpy()
    frame['log_document_candidate_rank'] = np.log(dr)
    for _, ix in frame.groupby('query_id', sort=False).indices.items():
        values = p[ix]
        top = np.sort(values)[::-1]
        for key, value in [('query_max_rationale', top[0]), ('query_second_rationale', top[min(1, len(top)-1)]),
                           ('log_query_rationale_mass', np.log1p(values.sum())),
                           ('log_query_high_rationale_count', np.log1p((values >= .5).sum()))]:
            frame.loc[ix, key] = value
    return frame


def features(frame, group):
    p = np.clip(frame.raw_multivers_rationale.to_numpy(float), 1e-6, 1-1e-6)
    e = np.clip(frame.mv_document_evidence.to_numpy(float), 1e-6, 1-1e-6)
    direction = np.abs(frame.mv_support.to_numpy(float)-frame.mv_refute.to_numpy(float))/e
    cols = [logit(p), logit(e), direction]
    if group == 'context':
        cols.extend(frame[k].to_numpy(float) for k in SPEC['context_features'])
    result = np.column_stack(cols)
    assert np.isfinite(result).all()
    return result


def candidates():
    names = ['rationale', 'product']
    for group in SPEC['groups']:
        names.extend(f'{group}:logistic:{c:g}' for c in SPEC['logistic_C'])
        names.extend(f'{group}:histogram:{l:g}' for l in SPEC['histogram_l2'])
    return names


def fit_model(train, name):
    if name in ['rationale', 'product']:
        return {'name': name}
    group, kind, parameter = name.split(':')
    x = features(train, group)
    y = train.gold_evidence.to_numpy(int)
    counts = train.groupby('query_id').size()
    w = train.query_id.map(1/counts).to_numpy(float)
    for value in [0, 1]:
        w[y == value] *= .5/w[y == value].sum()
    w *= len(w)/w.sum()
    if kind == 'logistic':
        scale = StandardScaler().fit(x)
        classifier = LogisticRegression(C=float(parameter), solver='lbfgs', max_iter=1000, random_state=SPEC['seed'])
        classifier.fit(scale.transform(x), y, sample_weight=w)
    else:
        scale = None
        classifier = HistGradientBoostingClassifier(**SPEC['histogram'], l2_regularization=float(parameter), random_state=SPEC['seed'])
        classifier.fit(x, y, sample_weight=w)
    return {'name': name, 'group': group, 'classifier': classifier, 'scale': scale}


def score_model(model, frame):
    if model['name'] == 'rationale':
        return frame.raw_multivers_rationale.to_numpy(float)
    if model['name'] == 'product':
        return frame.raw_multivers.to_numpy(float)
    x = features(frame, model['group'])
    if model['scale'] is not None:
        x = model['scale'].transform(x)
    return model['classifier'].predict_proba(x)[:, 1]


def select(fitting, families, seed):
    groups = sorted(set(families[q] for q in fitting.query_id), key=lambda g: hashlib.sha256(f'{seed}:context:{g}'.encode()).hexdigest())
    mapping = {g: i % SPEC['inner_folds'] for i, g in enumerate(groups)}
    fold = fitting.query_id.map(lambda q: mapping[families[q]]).to_numpy()
    rows = []
    names = candidates()
    for f in range(SPEC['inner_folds']):
        train = fitting.loc[fold != f].reset_index(drop=True)
        valid = fitting.loc[fold == f].reset_index(drop=True)
        assert not {families[q] for q in train.query_id}&{families[q] for q in valid.query_id}
        for name in names:
            model = fit_model(train, name)
            score = score_model(model, valid)
            mask, _ = admission.apply_threshold(valid, score, budget_rule(valid, score, valid.query_id.nunique()))
            rows.append({'seed': seed, 'fold': f, 'method': name, 'precision': float(valid.gold_evidence.to_numpy(int)[mask].mean())})
    cv = pd.DataFrame(rows)
    means = cv.groupby('method').precision.mean().to_dict()
    control_names = [n for n in names if not n.startswith('context:')]
    key = lambda n: (round(means[n], 12), -names.index(n))
    return max(names, key=key), max(control_names, key=key), cv


def run(smoke):
    start = time.monotonic()
    source_sha = digest(Path(__file__))
    if not smoke:
        audit = json.loads((OUT/'smoke/audit.json').read_text())
        assert audit['passed'] and audit['protocol_sha256'] == digest(SPEC_PATH) and audit['code_sha256'] == source_sha
    out = OUT/('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    frame = add_context(prepare_frame(), R2/'results/strengthening/multivers/document_scores.jsonl')
    families = source_families()
    query_info = json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    results, selections, cvs, collection = [], [], [], {}
    for seed in [13] if smoke else SPEC['original_seeds']:
        original = json.loads((R2/f'results/experiments/scifact_full_k100/model_seed{seed}.json').read_text())
        fitids, calids, testids = [original[k] for k in ['fitting_query_ids', 'threshold_query_ids', 'evaluation_query_ids']]
        if smoke:
            testids = sorted(testids)[:12]
        fit = frame[frame.query_id.isin(fitids)].reset_index(drop=True)
        cal = frame[frame.query_id.isin(calids)].reset_index(drop=True)
        test = frame[frame.query_id.isin(testids)].reset_index(drop=True)
        chosen, control, cv = select(fit, families, seed)
        cvs.append(cv)
        selections.append({'seed': seed, 'chosen': chosen, 'score_control': control})
        (out/f'choices_seed{seed}.json').write_text(json.dumps(selections[-1], indent=2))
        for label, name in [('rationale', 'rationale'), ('product', 'product'), ('score_control', control), ('selected', chosen)]:
            model = fit_model(fit, name)
            s, cs = score_model(model, test), score_model(model, cal)
            changed = test.copy()
            changed['gold_evidence'] = 1-changed.gold_evidence
            changed['claim_verdict'] = 'unused'
            assert np.allclose(score_model(model, changed.iloc[::-1])[::-1], s, rtol=0, atol=1e-12)
            for analysis, rule in [('exact_budget', budget_rule(test, s, len(testids))), ('prospective_100', admission.fit_threshold(cal, cs, 100))]:
                stat = query_statistics(test, s, rule, np.zeros(len(test), bool), {q: query_info[q] for q in testids})
                if analysis == 'exact_budget':
                    assert stat.admitted.sum() == len(testids)
                stat.to_csv(out/f'query_seed{seed}_{label}_{analysis}.csv', index=False)
                results.append({'seed': seed, 'method': label, 'chosen': name, 'analysis': analysis, **admission.summarize(stat)})
                collection.setdefault(analysis, {}).setdefault(seed, {})[label] = stat
            joblib.dump(model, out/f'model_seed{seed}_{label}.joblib')
        print('seed', seed, 'chosen', chosen, 'control', control, 'elapsed', round(time.monotonic()-start, 1), flush=True)
    data = pd.DataFrame(results)
    data.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(selections).to_csv(out/'selected.csv', index=False)
    pd.concat(cvs).to_csv(out/'inner_cv.csv', index=False)
    data.groupby(['analysis', 'method'])[['evidence_precision', 'coverage', 'evidence_recall', 'all_facts_recall', 'volume_per_100']].agg(['mean', 'std']).to_csv(out/'mean_sd.csv')
    if not smoke:
        rows = []
        for analysis, stats in collection.items():
            for reference in ['product', 'rationale', 'score_control']:
                for row in paired_intervals(stats, 'selected', reference, ['evidence_precision', 'coverage', 'evidence_recall', 'all_facts_recall', 'volume_per_100']):
                    rows.append({'analysis': analysis, **row})
        pd.DataFrame(rows).to_csv(out/'paired_intervals.csv', index=False)
        freeze_confirmation(frame, families, out)
    audit = {'passed': True, 'smoke': smoke, 'protocol_sha256': digest(SPEC_PATH), 'code_sha256': source_sha,
             'outcome_and_row_order_invariance': True, 'source_grouped_inner_selection': True,
             'original_evaluation_exploratory': True, 'elapsed_seconds': time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    print(data[data.analysis == 'exact_budget'][['seed', 'method', 'chosen', 'evidence_precision']].to_string(index=False), flush=True)


def freeze_confirmation(frame, families, out):
    calids = [q for q in families if int(hashlib.sha256(f"{SPEC['seed']}:calibration:{families[q]}".encode()).hexdigest(), 16) % 5 == 0]
    fitids = sorted(set(families)-set(calids))
    fit = frame[frame.query_id.isin(fitids)].reset_index(drop=True)
    cal = frame[frame.query_id.isin(calids)].reset_index(drop=True)
    chosen, control, cv = select(fit, families, SPEC['seed'])
    cv.to_csv(out/'pooled_inner_cv.csv', index=False)
    freeze = {'chosen': chosen, 'score_control': control, 'fitting_query_ids': fitids, 'calibration_query_ids': calids,
              'protocol_sha256': digest(SPEC_PATH), 'source_sha256': digest(Path(__file__)),
              'development_data_sha256': digest(R2/'data/scored/scifact_full_k100.parquet'),
              'confirmation_inputs_sha256': digest(R2/'data/multivers_confirmation/candidates.parquet'),
              'confirmation_performance_accessed': False, 'frozen_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'models': {}}
    for label, name in [('rationale', 'rationale'), ('product', 'product'), ('score_control', control), ('selected', chosen)]:
        model = fit_model(fit, name)
        rule = admission.fit_threshold(cal, score_model(model, cal), 100)
        path = out/f'confirmation_model_{label}.joblib'
        joblib.dump(model, path)
        freeze['models'][label] = {'name': name, 'path_name': path.name, 'sha256': digest(path), 'prospective_rule': rule}
    (out/'confirmation_freeze.json').write_text(json.dumps(freeze, indent=2))
    print('FROZEN confirmation:', chosen, 'control', control, 'fit', len(fitids), 'calibration', len(calids), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
