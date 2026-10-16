"""Finite fitting-only study of query-list features with matched single-view controls."""
from pathlib import Path
import argparse, hashlib, json, pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import r2_admission as a

R = Path(__file__).resolve().parents[1]
O = R / 'results/r5_context_gain'
COLS = ['query_id', 'candidate_id', 'tie_key', 'gold_evidence'] + a.VIEWS
PLAN = {
    'created': '2026-10-16', 'purpose': 'Fitting-only exploratory assessment of list-level features within scorer selection.',
    'dataset': 'fever', 'pool': 100, 'seeds': [13, 17, 29, 7],
    'screen': 'Union of the first five candidates of all twelve raw individual scorers; labels are never used.',
    'features': 'For each permitted view: training-quantile score, reciprocal within-query rank, query maximum score, query first-second score gap, candidate-minus-query-maximum score.',
    'groups': ['ce', 'all'], 'depths': [2, 3], 'iterations': 100, 'learning_rate': .05,
    'max_leaf_nodes': 7, 'min_samples_leaf': 30, 'l2': 10, 'max_bins': 63,
    'folds': 'Three deterministic fitting-only annotated-page-component folds. This separates shared annotated pages within the fitting sample, not all candidate sources.',
    'weights': 'Equal total candidate weight per query, then equal total class weight.',
    'selection': 'Mean out-of-fold precision at 100 sentences per 100 validation queries, maximum five per query. Ties prefer smaller depth.',
    'smoke': 'FEVER seed13 inner fold0, both groups and depths. Check finite values, label/order invariance, exact budget. Continue if all-view precision exceeds the trained CE group by at least .01.',
    'evaluation': 'Only after complete fitting-only model selection and a saved freeze file. Reuse independent original threshold calibration; report all groups, prospective and exact-batch results. No model re-selection on test.',
    'scope': 'New models are exploratory scoring-form candidates. Existing main-model results remain unchanged.',
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(seed):
    m=json.loads((R/f'results/experiments/fever_full_k100/model_seed{seed}.json').read_text())
    f=pd.read_parquet(R/'data/scored/fever_full_k100.parquet',columns=COLS,
        filters=[('query_id','in',m['fitting_query_ids'])])
    return f,m

def component_folds(ids):
    queries=json.loads((R/'data/processed/fever_full_queries.json').read_text(encoding='utf-8'))
    parent={q:q for q in ids}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    owners={}
    for q in sorted(ids):
        for cid in queries[q]['gold_ids']:
            page=cid.rsplit('::',1)[0]
            if page in owners:
                u,v=find(q),find(owners[page]);parent[max(u,v)]=min(u,v)
            else: owners[page]=q
    groups={q:find(q) for q in ids}
    order=sorted(set(groups.values()),key=lambda g:hashlib.sha256(f'20261016:list:{g}'.encode()).hexdigest())
    lookup={g:i%3 for i,g in enumerate(order)}
    return {q:lookup[groups[q]] for q in ids},groups

def list_inputs(f):
    n=len(f); outputs=[]; keep=np.zeros(n,bool)
    for col in a.VIEWS:
        s=f[col].to_numpy(float);rank=a.ranks(f,s);keep|=rank<=5
        maxima=f.assign(_s=s).groupby('query_id',sort=False)._s.nlargest(2)
        top={q:np.asarray(v,float) for q,v in maxima.groupby(level=0)}
        mx=f.query_id.map({q:v[0] for q,v in top.items()}).to_numpy(float)
        gap=f.query_id.map({q:v[0]-v[min(1,len(v)-1)] for q,v in top.items()}).to_numpy(float)
        outputs.append(np.column_stack([s,1/rank,mx,gap,s-mx]))
    return np.stack(outputs,axis=1).astype(np.float32),keep

def matrix(x,group,scale):
    views=[8] if group=='ce' else range(12)
    result=[]
    for j in views:
        for k in range(5):
            v=x[:,j,k].astype(float)
            if k!=1:
                grid=np.asarray(scale[str(j)+':'+str(k)])
                unique,ix=np.unique(grid,return_index=True)
                last=np.searchsorted(grid,unique,side='right')-1
                v=np.interp(v,unique,(ix+last)/(2*(len(grid)-1)),left=0,right=1)
            result.append(v)
    return np.column_stack(result).astype(np.float32)

def fit(x,f,group,depth):
    scale={f'{j}:{k}':np.quantile(x[:,j,k],np.linspace(0,1,257)).tolist() for j in range(12) for k in [0,2,3,4]}
    y=f.gold_evidence.to_numpy(int)
    w=1/f.groupby('query_id').query_id.transform('size').to_numpy(float)
    for c in [0,1]:w[y==c]*=.5/max(w[y==c].sum(),1e-12)
    w/=w.mean()
    model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=100,max_depth=depth,max_leaf_nodes=7,
        min_samples_leaf=30,l2_regularization=10,max_bins=63,early_stopping=False,random_state=20261016)
    model.fit(matrix(x,group,scale),y,sample_weight=w)
    return {'scale':scale,'model':model,'group':group,'depth':depth}

def predict(m,x):return m['model'].predict_proba(matrix(x,m['group'],m['scale']))[:,1]

def exact(f,s):
    r=a.fit_threshold(f,s,100);mask,_=a.apply_threshold(f,s,r)
    assert mask.sum()==f.query_id.nunique()
    return float(f.gold_evidence.to_numpy(int)[mask].mean())

def run(smoke):
    started=time.monotonic();O.mkdir(exist_ok=True,parents=True)
    plan=O/'protocol.json'
    encoded=json.dumps(PLAN,indent=2,ensure_ascii=False)+'\n'
    if plan.exists():assert plan.read_text(encoding='utf-8')==encoded
    else:plan.write_text(encoded,encoding='utf-8')
    out=O/('smoke' if smoke else 'development');out.mkdir(exist_ok=True)
    if not smoke:
        gate=json.loads((O/'smoke/audit.json').read_text());assert gate['passed'] and gate['useful_direction']
        assert gate['code_sha256']==sha(__file__) and gate['protocol_sha256']==sha(plan)
    rows=[];freezes=[]
    for seed in PLAN['seeds'][:1] if smoke else PLAN['seeds']:
        frame,base=load(seed);f=frame[frame.query_id.isin(base['fitting_query_ids'])].copy().reset_index(drop=True)
        fold,groups=component_folds(base['fitting_query_ids']);x,keep=list_inputs(f)
        f=f[keep].reset_index(drop=True);x=x[keep];codes=f.query_id.map(fold).to_numpy()
        for k in [0] if smoke else range(3):
            tr,va=codes!=k,codes==k
            assert not set(f.loc[tr,'query_id'].map(groups))&set(f.loc[va,'query_id'].map(groups))
            for group in PLAN['groups']:
                for depth in PLAN['depths']:
                    model=fit(x[tr],f.loc[tr],group,depth);scores=predict(model,x[va])
                    assert np.isfinite(scores).all()
                    assert np.array_equal(scores,predict(model,x[va][::-1])[::-1])
                    changed=f.loc[va].copy();changed['gold_evidence']=1-changed.gold_evidence
                    # Prediction receives only the label-free numeric matrix.
                    assert np.array_equal(scores,predict(model,x[va]))
                    p=exact(f.loc[va],scores)
                    rows.append({'seed':seed,'fold':k,'group':group,'depth':depth,'precision':p,'queries':f.loc[va].query_id.nunique()})
                    print(seed,k,group,depth,round(p,6),'elapsed',round(time.monotonic()-started,1),flush=True)
        if not smoke:
            part=pd.DataFrame(rows);part=part[part.seed==seed]
            selected={g:int(part[part.group==g].groupby('depth').precision.mean().sort_values(ascending=False,kind='stable').index[0]) for g in PLAN['groups']}
            for group,depth in selected.items():
                fitted=fit(x,f,group,depth)
                with (out/f'model_seed{seed}_{group}.pkl').open('wb') as handle:pickle.dump(fitted,handle)
            freezes.append({'seed':seed,'selected_depths':selected,'fitting_query_ids':base['fitting_query_ids'],
                'threshold_query_ids':base['threshold_query_ids'],'evaluation_query_ids':base['evaluation_query_ids'],
                'base_model_sha256':sha(R/f'results/experiments/fever_full_k100/model_seed{seed}.json')})
    df=pd.DataFrame(rows);df.to_csv(out/'cv.csv',index=False)
    means=df.groupby(['group','depth']).precision.mean()
    gain=float(means.loc['all'].max()-means.loc['ce'].max())
    record={'passed':True,'smoke':smoke,'useful_direction':gain>=.01,'all_minus_trained_ce':gain,
        'code_sha256':sha(__file__),'protocol_sha256':sha(plan),'test_outcomes_read':False,
        'all_outputs_exploratory':True,'elapsed_seconds':time.monotonic()-started}
    (out/'audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    if freezes:(out/'freeze.json').write_text(json.dumps(freezes,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    with threadpool_limits(limits=2):run(p.parse_args().smoke)
