"""Fitting-only test of prefix-completion utility at an unchanged sentence budget.

All artifacts stay in results/evidence_set_confirmation. No held-out queries
are loaded by the smoke/development stages. Existing manuscript files are
never edited. Run --smoke before --development.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import r5_context_gain as dev
import r5_simple_controls as simple

R = dev.R
O = R/'results/evidence_set_confirmation'
PLAN = {
    'created': '2026-10-16',
    'question': 'Can fitting-only prediction of complete evidence prefixes improve multi-sentence retention at the same sentence budget?',
    'dataset': 'fever', 'seeds': [13, 17, 29, 7],
    'candidate_screen': dev.PLAN['screen'],
    'base_models': 'Same existing list-feature model, depth three, refitted only on each inner training fold; all-view and same-capacity CE-feature control.',
    'base_features': dev.PLAN['features'],
    'query_model': '100-iteration histogram gradient boosting, depth two, seven leaves, min leaf 30, learning rate .05, L2 10; no class balancing.',
    'query_features': 'Top-five base probabilities followed by the corresponding base-model features, padded with zero, plus screened candidate count; no labels in features.',
    'target': 'Smallest base-score prefix of size one to five containing any complete annotated evidence set; zero if no such prefix exists.',
    'utility': 'Sum of calibrated expected correct sentence probabilities plus beta times probability that the selected prefix is complete. A least concave majorant makes marginal utilities nonincreasing; sentence ranking stays fixed.',
    'betas': [0.0, 0.25, 0.5, 1.0, 2.0],
    'budget': 'Exactly 100 sentences per 100 queries, with at most five per query. Fixed hash ties and strictly decreasing within-query utilities preserve prefixes.',
    'folds': 'Three deterministic fitting-only annotated-page-component folds from the existing protocol.',
    'choice': 'Among all-view beta candidates with mean OOF precision at least the CE beta-zero mean minus .01, maximize mean complete-set recall, then multi-sentence recall, then precision, then smaller beta. All controls and failures retained.',
    'smoke': 'Seed13 inner fold zero only. Technical checks and at least .005 higher multi-sentence recall than beta zero while precision is within .01 of CE beta zero are required before development.',
    'evaluation': 'No calibration or evaluation query labels are read at this stage. A later evaluation needs a saved selected-model freeze and its own explicit prospective calibration protocol.',
    'threads': 1,
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True)+'\n', encoding='utf-8')


def prepared(frame, x, model, queries):
    score = dev.predict(model, x)
    data = simple.compact(frame, score, queries)
    transformed = dev.matrix(x, model['group'], model['scale'])
    lookup = {(q, c): i for i, (q, c) in enumerate(zip(frame.query_id.astype(str), frame.candidate_id.astype(str)))}
    index = np.asarray([lookup[(q, c)] for q, c in zip(data['qids'][data['codes']], data['cids'])])
    width = transformed.shape[1]
    features = np.zeros((len(data['qids']), 5+5*width+1), dtype=np.float32)
    features[data['codes'], data['ranks']-1] = data['raw']
    for rank in range(1, 6):
        keep = data['ranks']==rank
        features[data['codes'][keep], 5+(rank-1)*width:5+rank*width] = transformed[index[keep]]
    features[:, -1] = frame.groupby('query_id').size().reindex(data['qids']).to_numpy()
    assert np.isfinite(features).all()
    labels = np.zeros(len(data['qids']), int)
    for qi, qid in enumerate(data['qids']):
        ranks = {c: int(r) for c, r in zip(data['cids'][data['codes']==qi], data['ranks'][data['codes']==qi])}
        possible = [max(ranks[c] for c in set(group)) for group in queries[qid]['gold_sets']
                    if group and all(c in ranks for c in set(group))]
        labels[qi] = min(possible, default=0)
    return data, features, labels


def fit_utility(data, features, target):
    model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=100, max_depth=2,
        max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=10, max_bins=63,
        early_stopping=False, random_state=20261016)
    model.fit(features, target)
    calibrator = LogisticRegression(C=1, max_iter=300, solver='lbfgs')
    counts = np.bincount(data['codes'], minlength=len(data['qids']))
    weight = 1/counts[data['codes']]
    p = np.clip(data['raw'], 1e-6, 1-1e-6)
    calibrator.fit(np.log(p/(1-p))[:,None], data['y'], sample_weight=weight/weight.mean())
    return {'model': model, 'calibrator': calibrator}


def scores(model, data, features, beta):
    prob = np.zeros((len(data['qids']), 6))
    prob[:, model['model'].classes_.astype(int)] = model['model'].predict_proba(features)
    p = np.clip(data['raw'], 1e-6, 1-1e-6)
    correct = model['calibrator'].predict_proba(np.log(p/(1-p))[:,None])[:,1]
    marginal = np.zeros((len(data['qids']), 5))
    marginal[data['codes'], data['ranks']-1] = correct
    marginal += beta*prob[:,1:]
    # Pool adjacent increasing slopes to obtain the least concave majorant.
    out = np.empty_like(marginal)
    for qi, row in enumerate(marginal):
        blocks = []
        for value in row:
            blocks.append([float(value), 1])
            while len(blocks)>1 and blocks[-2][0]/blocks[-2][1] < blocks[-1][0]/blocks[-1][1]:
                last = blocks.pop(); blocks[-1][0] += last[0]; blocks[-1][1] += last[1]
        out[qi] = np.concatenate([np.repeat(total/size, size) for total, size in blocks])
    result = out[data['codes'], data['ranks']-1]
    result = simple.prefix(data, result)
    assert np.isfinite(result).all()
    for rank in range(2,6):
        ii = np.flatnonzero(data['ranks']==rank)
        assert np.all(result[ii]<result[ii-1])
    return result


def run(smoke):
    start = time.monotonic()
    O.mkdir(parents=True, exist_ok=True)
    protocol = O/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text(encoding='utf-8'))==PLAN
    else:
        dump(protocol, PLAN)
    target = O/('smoke' if smoke else 'development')
    target.mkdir(exist_ok=True)
    assert not (target/'audit.json').exists(), 'Results already exist; preserve recorded attempts.'
    if not smoke:
        gate = json.loads((O/'smoke/audit.json').read_text())
        assert gate['technical_passed'] and gate['direction_passed']
        assert gate['code_sha256']==sha(__file__) and gate['protocol_sha256']==sha(protocol)
    dump(target/'before_fitting.json', {'code_sha256': sha(__file__), 'protocol_sha256': sha(protocol),
         'calibration_and_evaluation_outcomes_read': False})
    queries = json.loads((R/'data/processed/fever_full_queries.json').read_text(encoding='utf-8'))
    rows = []
    for seed in PLAN['seeds'][:1] if smoke else PLAN['seeds']:
        frame, base = dev.load(seed)
        folds, groups = dev.component_folds(base['fitting_query_ids'])
        x, keep = dev.list_inputs(frame)
        frame, x = frame[keep].reset_index(drop=True), x[keep]
        foldcodes = frame.query_id.map(folds).to_numpy()
        for fold in [0] if smoke else range(3):
            train, valid = foldcodes!=fold, foldcodes==fold
            assert not set(frame.loc[train,'query_id'].map(groups))&set(frame.loc[valid,'query_id'].map(groups))
            tf, vf = frame.loc[train].reset_index(drop=True), frame.loc[valid].reset_index(drop=True)
            for group in ['ce','all']:
                bm = dev.fit(x[train], tf, group, 3)
                tr, trx, ty = prepared(tf, x[train], bm, queries)
                va, vax, vy = prepared(vf, x[valid], bm, queries)
                utility = fit_utility(tr, trx, ty)
                for beta in PLAN['betas']:
                    predicted = scores(utility, va, vax, beta)
                    rule = simple.threshold(va, predicted, 100)
                    mask = simple.admitted(va, predicted, rule)
                    stat = simple.statistics(va, mask)
                    assert stat.admitted.sum()==len(va['qids'])
                    assert stat.admitted.max()<=5
                    summary = simple.summary(stat)
                    rows.append({'seed': seed,'fold':fold,'group':group,'beta':beta,**summary})
                    stat.to_parquet(target/f'query_seed{seed}_fold{fold}_{group}_beta{beta:g}.parquet', index=False)
                    print(seed,fold,group,beta,round(summary['evidence_precision'],5),round(summary['all_facts_recall'],5),round(summary['multiple_set_recall'],5),flush=True)
                with (target/f'model_seed{seed}_fold{fold}_{group}.pkl').open('wb') as h:
                    pickle.dump({'base':bm,'utility':utility}, h)
    results = pd.DataFrame(rows)
    results.to_csv(target/'cv.csv', index=False)
    means = results.groupby(['group','beta']).mean(numeric_only=True)
    means.to_csv(target/'mean.csv')
    threshold = float(means.loc[('ce',0.0),'evidence_precision']-.01)
    pool = means.loc['all'].reset_index()
    feasible = pool[pool.evidence_precision>=threshold]
    selected = feasible.sort_values(['all_facts_recall','multiple_set_recall','evidence_precision','beta'],ascending=[False,False,False,True],kind='stable').iloc[0] if len(feasible) else None
    multi_gain = float(selected.multiple_set_recall-means.loc[('all',0.0),'multiple_set_recall']) if selected is not None else float('nan')
    audit = {'technical_passed':True,'direction_passed':bool(selected is not None and multi_gain>=.005),
        'selected_beta':float(selected.beta) if selected is not None else None,
        'selected_mean':selected.to_dict() if selected is not None else None,
        'ce_beta_zero_precision_minus_01':threshold,'multiple_set_gain_over_all_beta_zero':multi_gain,
        'code_sha256':sha(__file__),'protocol_sha256':sha(protocol),
        'calibration_and_evaluation_outcomes_read':False,'all_attempts_retained':True,
        'elapsed_seconds':time.monotonic()-start}
    dump(target/'audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--development',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.smoke)
