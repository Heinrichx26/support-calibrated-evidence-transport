"""Acceptance calibration fitted to the candidates eligible for evidence admission."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import joblib

import r2_admission as admission
from strengthen_r2_analysis import budget_rule
import run_multivers_adaptive_calibration as adaptive
from infer_multivers_local_context import screens

R2=Path(__file__).resolve().parents[1]
OUT=R2/'results/multivers_extension/admission_calibration'
PROTOCOL=R2/'data/multivers_admission_calibration_protocol.json'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidates():
    raw=['full_raw','full_product','single_raw','mean:0.25:1','mean:0.5:1','mean:0.75:1']
    logistic=[f'logistic:{g}:{r:g}' for g in ['full','dual'] for r in [1e-5,1e-4,.001]]
    boosted=[f'boost:{g}:{depth}:{balanced}:{mono}' for g in ['full','dual','full_context','context']
             for depth in [2,3] for balanced in [0,1] for mono in [0,1]]
    return raw+logistic+boosted


def fit(frame,name):
    result={'name':name}
    if name.startswith('boost:'):
        _,group,depth,balanced,mono=name.split(':')
        x,constraints=adaptive.arrays(frame,group)
        # Only the two rationale scores have an imposed monotone direction.
        if group in ['full','full_context']:
            constraints=[1]+[0]*(x.shape[1]-1)
        else:
            constraints=[1,0,0,1]+[0]*(x.shape[1]-4)
        y=frame.gold_evidence.to_numpy(int)
        w=adaptive.weights(frame)
        if int(balanced):
            for c in [0,1]:
                w[y==c]*=.5/w[y==c].sum()
            w/=w.mean()
        classifier=HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,
            max_depth=int(depth),max_leaf_nodes=7,min_samples_leaf=30,
            l2_regularization=5,monotonic_cst=constraints if int(mono) else None,
            early_stopping=False,random_state=2026101583)
        classifier.fit(x,y,sample_weight=w)
        result.update(group=group,classifier=classifier)
        return result
    return adaptive.fit(frame,name)


def score(model,frame):
    return adaptive.score(model,frame)


def run(smoke):
    start=time.monotonic()
    out=OUT/('smoke' if smoke else 'development')
    out.mkdir(parents=True,exist_ok=True)
    if not smoke:
        gate=json.loads((OUT/'smoke/audit.json').read_text())
        assert gate['passed'] and gate['useful_signal_over_fusion']
        assert gate['source_sha256']==digest(Path(__file__)) and gate['protocol_sha256']==digest(PROTOCOL)
    complete=adaptive.prepare()
    keep=screens(complete)
    frame=complete[keep].reset_index(drop=True)
    folds=adaptive.folds(frame)
    names=candidates()
    rows,predictions=[],[]
    for k in range(2 if smoke else 4):
        train,valid=frame[folds!=k].reset_index(drop=True),frame[folds==k].reset_index(drop=True)
        assert not set(train.family)&set(valid.family)
        for name in names:
            model=fit(train,name)
            s=score(model,valid)
            assert np.isfinite(s).all()
            if smoke:
                changed=valid.copy()
                changed['gold_evidence']=1-changed.gold_evidence
                assert np.allclose(score(model,changed.iloc[::-1])[::-1],s,atol=1e-12,rtol=0)
            for q in adaptive.quality(valid,s):
                rows.append({'fold':k,'method':name,**q})
            if not smoke:
                predictions.append(valid[['query_id','candidate_id','pool','family','gold_evidence','tie_key']].assign(
                    fold=k,method=name,score=s))
        print('fold',k,'completed; elapsed',round(time.monotonic()-start,1),'s',flush=True)
    cv=pd.DataFrame(rows)
    cv.to_csv(out/'cv.csv',index=False)
    mean=cv.groupby(['method','pool']).precision.mean().unstack()
    mean['objective']=mean.mean(axis=1)
    mean.sort_values('objective',ascending=False).to_csv(out/'mean.csv')
    key=lambda name:(round(mean.loc[name,'objective'],12),-names.index(name))
    controls=[n for n in names if n in ['full_raw','full_product','single_raw'] or ':full:' in n or ':full_context:' in n]
    blends=[n for n in names if n.startswith('mean:')]
    proposed=[n for n in names if ':dual:' in n or ':context:' in n]
    chosen,control,blend=max(proposed,key=key),max(controls,key=key),max(blends,key=key)
    useful=mean.loc[chosen,'objective']>mean.loc[blend,'objective']
    audit={'passed':True,'smoke':smoke,'useful_signal_over_fusion':bool(useful),
        'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),
        'chosen':chosen,'single_control':control,'arithmetic_control':blend,
        'candidate_rows':len(frame),'claims':frame.query_id.nunique(),
        'source_groups_disjoint':True,'all_results_exploratory':True,
        'elapsed_seconds':time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    if not smoke:
        pd.concat(predictions,ignore_index=True).to_parquet(out/'oof_scores.parquet',index=False)
    print(json.dumps(audit),flush=True)
    print(mean.sort_values('objective',ascending=False).head(16).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
