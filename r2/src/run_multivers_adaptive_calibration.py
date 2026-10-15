"""Exploratory source-grouped dual-context calibration over both candidate pools."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import r2_admission as admission
from strengthen_r2_analysis import budget_rule
import run_multivers_dual_calibration as olddual
from evaluate_multivers_confirmation import prepare_confirmation_frame

R2 = Path(__file__).resolve().parents[1]
DATA = R2/'data/multivers_adaptive'
OUT = R2/'results/multivers_extension/adaptive_calibration'
PROTOCOL = R2/'data/multivers_adaptive_protocol.json'
SPEC = json.loads(PROTOCOL.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA/'development.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    old = olddual.prepare()
    old['pool'] = 'controlled_pool'
    old['cohort'] = 'original_dev'
    old['family'] = old.query_id.map(lambda q: 'D:'+olddual.context.source_families()[q])
    recent = prepare_confirmation_frame()
    queries = json.loads((R2/'data/multivers_confirmation/queries.json').read_text(encoding='utf-8'))
    recent = olddual.context.add_context(recent, R2/'results/multivers_extension/confirmation_inference/document_scores.jsonl')
    recent['family'] = recent.query_id.map(lambda q: 'T:'+queries[q]['source_family'])
    recent['cohort'] = 'previous_confirmation'
    parts = [old]
    for pool in ['controlled_pool', 'natural_pool']:
        p = recent[recent[pool]].copy().reset_index(drop=True)
        p['pool'] = pool
        parts.append(p)
    frame = pd.concat(parts, ignore_index=True)
    # Derive candidate-pool features separately: no other pool enters a score.
    frame['sample_id'] = frame.query_id+':'+frame.pool
    for pool, ix in frame.groupby('pool', sort=False).indices.items():
        part = frame.iloc[ix].copy().reset_index(drop=True)
        for col, prefix in [('raw_multivers_rationale', 'full'), ('single_rationale', 'single')]:
            score = part[col].to_numpy(float)
            frame.loc[ix, prefix+'_log_rank'] = np.log(admission.ranks(part, score))
            maximum = part.groupby('query_id')[col].transform('max').to_numpy(float)
            frame.loc[ix, prefix+'_gap_to_max'] = maximum-score
    assert frame.groupby(['query_id','pool']).size().eq(100).all()
    assert not frame.duplicated(['query_id','candidate_id','pool']).any()
    assert frame.groupby('query_id').family.nunique().eq(1).all()
    frame.to_parquet(cache, index=False)
    (DATA/'development_audit.json').write_text(json.dumps({
        'claims': frame.query_id.nunique(), 'source_groups': frame.family.nunique(),
        'rows': len(frame), 'all_outcomes_previously_evaluated': True,
        'cache_sha256': digest(cache), 'code_sha256': digest(Path(__file__))}, indent=2))
    return frame


def arrays(frame, group):
    def lp(x):
        return logit(np.clip(np.asarray(x, float), 1e-5, 1-1e-5))
    full = lp(frame.raw_multivers_rationale)
    single = lp(frame.single_rationale)
    evidence = frame.mv_document_evidence.to_numpy(float)
    direction = np.abs(frame.mv_support-frame.mv_refute).to_numpy(float)/np.maximum(evidence,1e-6)
    full_features = [full, lp(evidence), direction]
    if group == 'full':
        cols = full_features
        monotone = [1,1,0]
    elif group == 'single':
        cols = [single]
        monotone = [1]
    elif group == 'dual':
        cols = full_features+[single]
        monotone = [1,1,0,1]
    elif group == 'context':
        names = olddual.context.SPEC['context_features'][:5]+[
            'full_log_rank','single_log_rank','full_gap_to_max','single_gap_to_max']
        cols = full_features+[single]+[frame[n].to_numpy(float) for n in names]
        monotone = [1,1,0,1]+[0]*len(names)
    elif group == 'full_context':
        names = olddual.context.SPEC['context_features'][:5]+['full_log_rank','full_gap_to_max']
        cols = full_features+[frame[n].to_numpy(float) for n in names]
        monotone = [1,1,0]+[0]*len(names)
    else:
        raise ValueError(group)
    x = np.column_stack(cols)
    assert np.isfinite(x).all()
    return x, monotone


def candidates():
    raw = ['full_raw','full_product','single_raw']
    blends = [f'mean:{w:g}:1' for w in [.25,.5,.75]]
    powers = [f'mean:{w:g}:{p:g}' for w in [.25,.5,.75] for p in [-1,0,2]]
    logistic = [f'logistic:{group}:{r:g}' for group in ['full','dual'] for r in [.01,.1,1]]
    boosted = [f'monotone:{group}:{depth}:{l2:g}' for group in ['full','single','dual','full_context','context']
               for depth,l2 in [(2,1),(3,10)]]
    return raw+blends+powers+logistic+boosted


def weights(frame):
    n = frame.groupby('sample_id').query_id.transform('size').to_numpy(float)
    pools = frame.groupby('query_id').pool.transform('nunique').to_numpy(float)
    w = 1/n/pools
    return w/w.mean()


def fit(train, name):
    result = {'name': name}
    if name.startswith('logistic:'):
        _, group, reg = name.split(':')
        x, constraints = arrays(train, group)
        # Positive slopes preserve confidence ordering in each model context.
        xx = np.column_stack([np.ones(len(x)), x])
        y, w = train.gold_evidence.to_numpy(float), weights(train)
        reg = float(reg)
        def objective(b):
            z = xx@b
            loss = np.sum(w*(np.logaddexp(0,z)-y*z))/w.sum()+.5*reg*np.dot(b[1:],b[1:])
            grad = xx.T@(w*(expit(z)-y))/w.sum()+np.r_[0,reg*b[1:]]
            return loss, grad
        bounds = [(None,None)]+[(0,None) if c else (None,None) for c in constraints]
        solved = minimize(objective, np.zeros(xx.shape[1]), jac=True, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter':250,'ftol':1e-10})
        assert solved.success, solved.message
        result.update(group=group, coefficients=solved.x)
    elif name.startswith('monotone:'):
        _, group, depth, l2 = name.split(':')
        x, constraints = arrays(train, group)
        clf = HistGradientBoostingClassifier(learning_rate=.05, max_iter=120, max_depth=int(depth),
            max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=float(l2),
            monotonic_cst=constraints, early_stopping=False, random_state=SPEC['seed'])
        clf.fit(x,train.gold_evidence.to_numpy(int),sample_weight=weights(train))
        result.update(group=group, classifier=clf)
    return result


def score(model, frame):
    name = model['name']
    f, s = frame.raw_multivers_rationale.to_numpy(float), frame.single_rationale.to_numpy(float)
    if name == 'full_raw':
        return f
    if name == 'single_raw':
        return s
    if name == 'full_product':
        return f*frame.mv_document_evidence.to_numpy(float)
    if name.startswith('mean:'):
        _, w, power = name.split(':')
        w, power = float(w), float(power)
        f, s = np.clip(f,1e-7,1),np.clip(s,1e-7,1)
        if power == 0:
            return np.exp(w*np.log(f)+(1-w)*np.log(s))
        return (w*f**power+(1-w)*s**power)**(1/power)
    x, _ = arrays(frame, model['group'])
    if 'coefficients' in model:
        return expit(model['coefficients'][0]+x@model['coefficients'][1:])
    return model['classifier'].predict_proba(x)[:,1]


def quality(frame, values):
    rows = []
    for pool, ix in frame.groupby('pool',sort=False).indices.items():
        part, s = frame.iloc[ix].reset_index(drop=True), values[ix]
        n = part.query_id.nunique()
        mask, _ = admission.apply_threshold(part,s,budget_rule(part,s,n))
        assert mask.sum() == n
        rows.append({'pool':pool,'precision':float(part.gold_evidence.to_numpy(int)[mask].mean()),
                     'claims':n})
    return rows


def folds(frame, count=4, salt='outer'):
    groups = sorted(frame.family.unique(),key=lambda f:hashlib.sha256(f"{SPEC['seed']}:{salt}:{f}".encode()).hexdigest())
    mapping = {g:i%count for i,g in enumerate(groups)}
    return frame.family.map(mapping).to_numpy(int)


def run(smoke):
    start = time.monotonic()
    out = OUT/('smoke' if smoke else 'development')
    out.mkdir(parents=True,exist_ok=True)
    if not smoke:
        audit = json.loads((OUT/'smoke/audit.json').read_text())
        assert audit['passed'] and audit['source_sha256'] == digest(Path(__file__))
        assert audit['protocol_sha256'] == digest(PROTOCOL)
        assert audit['useful_dual_signal']
    frame = prepare()
    fold = folds(frame)
    methods = candidates()
    rows, predictions = [], []
    for k in range(2 if smoke else SPEC['folds']):
        train, valid = frame[fold!=k].reset_index(drop=True),frame[fold==k].reset_index(drop=True)
        assert not set(train.family)&set(valid.family)
        for name in methods:
            model = fit(train,name)
            s = score(model,valid)
            assert np.isfinite(s).all()
            if smoke:
                changed = valid.copy()
                changed['gold_evidence'] = 1-changed.gold_evidence
                assert np.allclose(score(model,changed.iloc[::-1])[::-1],s,atol=1e-12,rtol=0)
            for q in quality(valid,s):
                rows.append({'fold':k,'method':name,**q})
            if not smoke:
                predictions.append(valid[['query_id','candidate_id','pool','family','gold_evidence','tie_key']].assign(
                    fold=k,method=name,score=s))
        print('fold',k,'completed; elapsed',round(time.monotonic()-start,1),'s',flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(out/'cv.csv',index=False)
    means = cv.groupby(['method','pool']).precision.mean().unstack()
    means['objective'] = means.mean(axis=1)
    means.sort_values('objective',ascending=False).to_csv(out/'mean.csv')
    key = lambda name:(round(means.loc[name,'objective'],12),-methods.index(name))
    single = [n for n in methods if n in ['full_raw','full_product','single_raw'] or ':full:' in n or ':single:' in n or ':full_context:' in n]
    arithmetic = [n for n in methods if n.startswith('mean:') and n.endswith(':1')]
    learned_dual = [n for n in methods if ':dual:' in n or ':context:' in n]
    chosen, control, blend = max(learned_dual,key=key),max(single,key=key),max(arithmetic,key=key)
    useful = means.loc[chosen,'objective'] > means.loc['full_raw','objective']
    audit = {'passed':True,'smoke':smoke,'source_sha256':digest(Path(__file__)),
             'protocol_sha256':digest(PROTOCOL),'useful_dual_signal':bool(useful),
             'source_group_separation':True,'outcome_row_invariance':smoke,
             'all_evaluated_data_are_exploratory':True,
             'chosen':chosen,'single_control':control,'arithmetic_control':blend,
             'elapsed_seconds':time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    if not smoke:
        pd.concat(predictions,ignore_index=True).to_parquet(out/'oof_scores.parquet',index=False)
    print(json.dumps(audit),flush=True)
    print(means.sort_values('objective',ascending=False).head(16).to_string(),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
