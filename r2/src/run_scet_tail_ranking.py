"""Bounded, training-only investigation of losses aligned with admission ranking."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'data/scet_tail_ranking_protocol.json'
SPEC=json.loads(PLAN.read_text())
OUT=ROOT/'results/scet_tail_ranking'
CONTEXT=json.loads((ROOT/'data/multivers_context_protocol.json').read_text())['context_features'][:5]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def load_data(name):
    if name=='scifact_development':
        frame=pd.read_parquet(ROOT/'data/multivers_adaptive/development.parquet')
    else:
        dataset=name.removesuffix('_training')
        freeze=json.loads((ROOT/f'results/multivers_extension/external_evaluation/{dataset}/freeze.json').read_text())
        frame=pd.read_parquet(ROOT/'data/multivers_external/probabilities.parquet')
        frame=frame[frame.query_id.isin(freeze['fitting_ids'])].copy().reset_index(drop=True)
        # Read training annotations only. Existing tests are never opened.
        claims=[json.loads(line) for line in (ROOT/f'data/multivers_external/data/{dataset}/claims_train.jsonl').read_text().splitlines()]
        import evaluate_multivers_external as external
        groups=external.source_groups(claims)
        gold={f"{dataset}:train:{c['id']}":{f'{d}::{s}' for d,sets in c['evidence'].items() for g in sets for s in g['sentences']} for c in claims}
        frame['family']=frame.query_id.map(lambda q:dataset+':'+groups[q.rsplit(':',1)[1]])
        frame['gold_evidence']=[int(cid in gold[q]) for q,cid in zip(frame.query_id,frame.candidate_id)]
    assert frame.groupby('query_id').family.nunique().eq(1).all()
    # The common top-five union uses predictor values and fixed tie keys.
    keep=np.zeros(len(frame),bool)
    for name in ['full_raw','full_product','single_raw','mean:0.25','mean:0.5','mean:0.75']:
        scores=raw_score(frame,name)
        for _,ix in frame.groupby('pool',sort=False).indices.items():
            part=frame.iloc[ix].reset_index(drop=True)
            keep[ix[ranks(part,scores[ix])<=5]]=True
    return frame[keep].reset_index(drop=True)

def ranks(frame,scores):
    order=np.lexsort((frame.tie_key.to_numpy(str),-scores,frame.query_id.to_numpy(str)))
    q=frame.query_id.to_numpy(str)[order]
    starts=np.r_[0,np.flatnonzero(q[1:]!=q[:-1])+1]
    values=np.arange(len(order))-np.repeat(starts,np.diff(np.r_[starts,len(order)]))+1
    result=np.empty(len(frame),int); result[order]=values
    return result

def raw_score(frame,name):
    full=frame.raw_multivers_rationale.to_numpy(float)
    single=frame.single_rationale.to_numpy(float)
    if name=='full_raw': return full
    if name=='full_product': return full*frame.mv_document_evidence.to_numpy(float)
    if name=='single_raw': return single
    w=float(name.split(':')[1]); return w*full+(1-w)*single

def arrays(frame,group):
    lp=lambda x:logit(np.clip(np.asarray(x,float),1e-5,1-1e-5))
    evidence=frame.mv_document_evidence.to_numpy(float)
    full=[lp(frame.raw_multivers_rationale),lp(evidence),np.abs(frame.mv_support-frame.mv_refute).to_numpy(float)/np.maximum(evidence,1e-6)]
    extras=CONTEXT+['full_log_rank','full_gap_to_max']
    if group=='dual':
        full.append(lp(frame.single_rationale))
        extras=CONTEXT+['full_log_rank','single_log_rank','full_gap_to_max','single_gap_to_max']
    x=np.column_stack(full+[frame[c].to_numpy(float) for c in extras])
    assert np.isfinite(x).all()
    return x

def weights(frame):
    w=1/frame.groupby('sample_id').query_id.transform('size').to_numpy(float)
    w/=frame.groupby('query_id').pool.transform('nunique').to_numpy(float)
    return w/w.mean()

def tail_derivatives(y,weights,scores,beta,return_loss=False,fixed=None):
    pos=np.flatnonzero(y==1)
    neg=np.flatnonzero(y==0)
    if fixed is None:
        count=max(1,int(np.ceil(beta*len(neg))))
        neg=neg[np.argsort(-scores[neg],kind='stable')[:count]]
    else:neg=fixed
    wp=weights[pos]/weights[pos].sum(); wn=weights[neg]/weights[neg].sum()
    grad=np.zeros(len(y)); hess=np.zeros(len(y)); loss=0.
    # Chunking bounds memory independently of corpus size.
    for start in range(0,len(pos),256):
        p=pos[start:start+256]
        difference=scores[neg][None,:]-scores[p][:,None]
        prob=expit(difference)
        pair_weights=wp[start:start+256,None]*wn[None,:]
        g=pair_weights*prob; h=g*(1-prob)
        grad[p]-=g.sum(axis=1); grad[neg]+=g.sum(axis=0)
        hess[p]+=h.sum(axis=1); hess[neg]+=h.sum(axis=0)
        if return_loss:loss+=np.sum(pair_weights*np.logaddexp(0,difference))
    scale=len(y)
    if return_loss:return loss,grad/scale*scale,neg
    return grad*scale,np.maximum(hess*scale,1e-8)

def names():
    return ['full_raw','full_product','single_raw','mean:0.25','mean:0.5','mean:0.75']+[
        f'hist:{group}:{depth}' for group in ['full','dual'] for depth in [2,3]]+[
        f'lgb:{group}:{loss}' for group in ['full','dual'] for loss in ['binary','tail05','tail20']]

def fit(frame,name):
    result={'name':name}
    if name.startswith(('full_','single_','mean:')):return result
    kind,group,option=name.split(':'); x=arrays(frame,group)
    y=frame.gold_evidence.to_numpy(int); w=weights(frame)
    if kind=='hist':
        model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_depth=int(option),max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=SPEC['seed'])
        model.fit(x,y,sample_weight=w)
    else:
        objective='binary'
        if option!='binary':
            beta={'tail05':.05,'tail20':.20}[option]
            objective=lambda pred,ds:tail_derivatives(y,w,pred,beta)
        model=lgb.train({'objective':objective,'learning_rate':.05,'max_depth':3,'num_leaves':7,
            'min_data_in_leaf':30,'lambda_l2':5,'min_sum_hessian_in_leaf':1e-3,
            'verbosity':-1,'num_threads':2,'deterministic':True,'force_col_wise':True,
            'seed':SPEC['seed']},lgb.Dataset(x,label=y,weight=w,free_raw_data=False),num_boost_round=120)
    result.update(group=group,model=model)
    return result

def predict(model,frame):
    if 'model' not in model:return raw_score(frame,model['name'])
    x=arrays(frame,model['group'])
    if model['name'].startswith('hist:'):return model['model'].predict_proba(x)[:,1]
    return model['model'].predict(x)

def metrics(frame,scores):
    rows=[]
    for pool,ix in frame.groupby('pool',sort=False).indices.items():
        p=frame.iloc[ix].reset_index(drop=True); s=scores[ix]; n=p.query_id.nunique()
        eligible=np.flatnonzero(ranks(p,s)<=5)
        order=eligible[np.lexsort((p.tie_key.to_numpy(str)[eligible],-s[eligible]))][:n]
        assert len(order)==n
        rows.append({'pool':pool,'queries':n,'admitted':len(order),
            'correct':int(p.gold_evidence.to_numpy(int)[order].sum()),
            'precision':float(p.gold_evidence.to_numpy(int)[order].mean())})
    return rows

def folds(frame,salt='outer',count=4):
    groups=sorted(frame.family.unique(),key=lambda q:hashlib.sha256(f"{SPEC['seed']}:{salt}:{q}".encode()).hexdigest())
    return frame.family.map({q:i%count for i,q in enumerate(groups)}).to_numpy(int)

def derivative_check():
    rng=np.random.default_rng(7);y=np.r_[np.ones(6),np.zeros(18)];w=rng.uniform(.1,2,len(y));s=rng.normal(size=len(y));d=rng.normal(size=len(y))
    loss,g,neg=tail_derivatives(y,w,s,.2,True)
    eps=1e-6
    numerical=(tail_derivatives(y,w,s+eps*d,.2,True,neg)[0]-tail_derivatives(y,w,s-eps*d,.2,True,neg)[0])/(2*eps)
    assert np.isclose(numerical,g@d,atol=1e-7)
    grad,hess=tail_derivatives(y,w,s,.2)
    assert abs(grad.sum())<1e-10 and (hess>0).all()

def run(smoke):
    start=time.monotonic();out=OUT/('smoke' if smoke else 'development');out.mkdir(parents=True,exist_ok=True)
    if not smoke:
        gate=json.loads((OUT/'smoke/audit.json').read_text())
        assert gate['passed'] and gate['useful_direction']
        assert gate['code_sha256']==digest(Path(__file__)) and gate['protocol_sha256']==digest(PLAN)
    derivative_check(); rows=[];predictions=[]
    for dataset in SPEC['datasets'][:1] if smoke else SPEC['datasets']:
        frame=load_data(dataset);fo=folds(frame)
        for k in range(1 if smoke else 4):
            a,b=frame[fo!=k].reset_index(drop=True),frame[fo==k].reset_index(drop=True)
            assert not set(a.family)&set(b.family)
            for name in names():
                model=fit(a,name); scores=predict(model,b)
                assert np.isfinite(scores).all()
                if smoke:
                    modified=b.iloc[::-1].copy(); modified['gold_evidence']=1-modified.gold_evidence
                    assert np.allclose(predict(model,modified)[::-1],scores,rtol=0,atol=1e-10)
                rows.extend({'dataset':dataset,'fold':k,'method':name,**m} for m in metrics(b,scores))
                if not smoke:
                    predictions.append(b[['query_id','candidate_id','pool','family','gold_evidence','tie_key']].assign(dataset=dataset,fold=k,method=name,score=scores))
                print(dataset,k,name,'elapsed',round(time.monotonic()-start,1),flush=True)
    cv=pd.DataFrame(rows);cv.to_csv(out/'cv.csv',index=False)
    means=cv.groupby(['dataset','method','pool']).precision.mean().groupby(['dataset','method']).mean().rename('precision').reset_index()
    means.to_csv(out/'mean.csv',index=False)
    probe=means[means.dataset=='scifact_development'].set_index('method').precision
    best_new=probe[[n for n in names() if n.startswith('lgb:dual:')]].max()
    best_old=probe[[n for n in names() if n.startswith('hist:dual:')]].max()
    best_raw=probe[[n for n in names() if n in ['full_raw','full_product','single_raw']]].max()
    audit={'passed':True,'smoke':smoke,'code_sha256':digest(Path(__file__)),'protocol_sha256':digest(PLAN),
        'useful_direction':bool(best_new>best_raw and best_new>=best_old-.005),
        'smoke_best_new':float(best_new),'smoke_best_existing_scet':float(best_old),'smoke_best_raw':float(best_raw),
        'previous_external_test_labels_read':False,'all_new_results_development_only':True,
        'derivatives_numerically_verified':True,'elapsed_seconds':time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    if predictions:pd.concat(predictions,ignore_index=True).to_parquet(out/'oof_scores.parquet',index=False)
    print(means.sort_values(['dataset','precision'],ascending=[True,False]).to_string(index=False),flush=True)
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    with threadpool_limits(limits=2):run(parser.parse_args().smoke)
