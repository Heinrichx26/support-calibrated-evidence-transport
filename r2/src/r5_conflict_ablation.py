"""Refitted ablations of explicit conflict inputs, with the core method frozen.

The original fitting transforms, reliability estimates, folds, C grid and
selection objective are reused. Every ablation refits all logistic portfolios
and reselects its scoring form using fitting queries only. Independent original
calibration queries set its workload threshold. Test outcomes are evaluated once.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression

import r2_admission as admission
from r2_statistics import query_statistics
from r5_risk_query import aggregate, query_risk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/r5_risk_conflict/conflict'
PLAN = {
    'version':'r5_refitted_explicit_conflict_ablation_20261016',
    'datasets':['scifact','fever','hover'], 'seeds':[13,17,29,7],
    'variants':['without_explicit_conflict_indicator','without_explicit_conflict_and_summaries'],
    'common_removal':'delete kappa from portfolio inputs; remove kappa from RAG utility; disable all external conflict penalties',
    'summary_removal':'second variant additionally deletes inter-view SD and semantic-anchor gap engineered portfolio summaries',
    'remaining_inputs':'12 raw views and remaining generic summaries; raw NLI probabilities remain. These can retain implicit conflict information.',
    'selection':'same original 3 query folds, query/class weights, C in [0.1,1,10], selection objective and tie rules',
    'calibration':'same independent threshold-calibration queries; target100 items per100 queries; rank<=5',
    'evaluation':['prospective','exact_budget'],
    'shared_stress':'original frozen common stress subset, unchanged and excluded from the ablated predictor',
    'threads':2,
}


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(dataset,ids):
    columns=['query_id','candidate_id','gold_evidence','tie_key']+admission.VIEWS
    return pd.read_parquet(ROOT/f'data/scored/{dataset}_full_k100.parquet',
                           columns=columns,filters=[('query_id','in',list(ids))]).reset_index(drop=True)


def design(x,variant):
    # Twelve views, mean, max, SD, semantic-anchor product, gap, kappa.
    full=admission.features(x,np.zeros(len(x)))
    remove=[17] if variant=='without_explicit_conflict_indicator' else [14,16,17]
    z=np.delete(full,remove,axis=1)
    assert z.shape[1]==(17 if len(remove)==1 else 15)
    return z


def family(frame,x,z,fitted,reliability):
    # Zero signal removes the fixed internal RAG penalty as well as any other
    # explicit kappa dependency. The reduced z enters every refitted portfolio.
    return admission.family(frame,x,np.zeros(len(frame)),z,fitted,reliability)


def fit_portfolio(frame,z,c):
    # Identical estimator and weights to the core. Copy the mapped weights for
    # pandas 3, whose to_numpy view can be read-only. The core remains unchanged.
    y=frame.gold_evidence.to_numpy(int)
    if len(np.unique(y))<2:
        return {'constant':float(y.mean())}
    counts=frame.groupby('query_id').size()
    weights=frame.query_id.map(1/counts).to_numpy(float,copy=True)
    weights*=len(weights)/weights.sum()
    clf=LogisticRegression(C=c,class_weight='balanced',solver='lbfgs',max_iter=500,random_state=13)
    clf.fit(z,y,sample_weight=weights)
    return {'weights':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),'C':c}


def fit(fitting,calibration,original,variant):
    seed=original['seed']
    x=admission.transform(fitting,original['scale'])
    z=design(x,variant)
    ids=sorted(fitting.query_id.unique(),key=lambda q:hashlib.sha256(f'{seed}:inner:{q}'.encode()).hexdigest())
    fold_map={q:i%admission.PROTOCOL['inner_folds'] for i,q in enumerate(ids)}
    folds=fitting.query_id.map(fold_map).to_numpy()
    oof={c:np.zeros(len(fitting)) for c in admission.PROTOCOL['portfolio_regularization_C']}
    for fold in range(admission.PROTOCOL['inner_folds']):
        a,b=folds!=fold,folds==fold
        for c in oof:
            clf=fit_portfolio(fitting.loc[a],z[a],c)
            oof[c][b]=admission.portfolio(z[b],clf)
    fitted={c:fit_portfolio(fitting,z,c) for c in oof}
    base=family(fitting,x,z,fitted,original['reliability'])
    for c,values in oof.items():
        base['portfolio_'+format(float(c),'g')]=values
    qualities=[(admission.selection_quality(fitting,values,100,'gold_evidence'),name) for name,values in base.items()]
    best=max(qualities,key=lambda row:(row[0],row[1]))
    model={
        'seed':seed,'variant':variant,'scale':original['scale'],
        'reliability':original['reliability'],'portfolios':fitted,
        'selected_form':best[1],'selection_quality':best[0],
        'fitting_queries':len(ids),'calibration_queries':int(calibration.query_id.nunique()),
        'feature_count':z.shape[1], 'external_penalty':0,
        'fitting_query_ids':original['fitting_query_ids'],
        'threshold_query_ids':original['threshold_query_ids'],
        'candidate_qualities':[{ 'form':name,'quality':quality} for quality,name in qualities],
    }
    cs=score(calibration,model)
    model['threshold']=admission.fit_threshold(calibration,cs,100)
    return model


def score(frame,model):
    x=admission.transform(frame,model['scale'])
    fitted={float(c):v for c,v in model['portfolios'].items()}
    values=family(frame,x,design(x,model['variant']),fitted,model['reliability'])
    return np.round(values[model['selected_form']],12)


def exact_rule(frame,values):
    rank=admission.ranks(frame,values)
    indices=np.flatnonzero(rank<=5)
    order=indices[np.lexsort((frame.tie_key.to_numpy(str)[indices],-values[indices]))]
    boundary=order[frame.query_id.nunique()-1]
    return {'threshold':float(values[boundary]),'tie_cutoff':str(frame.iloc[boundary].tie_key),'eligible_rank':5}


def summarize(stats):
    values=admission.summarize(stats)
    values.update(query_risk(stats))
    return values


def bootstrap(collection,dataset):
    methods=PLAN['variants']
    metrics=['evidence_precision','evidence_recall','all_facts_recall','stress_precision','coverage',
             'unconditional_query_error','conditional_query_error','volume_per_100']
    ids=sorted(set().union(*(set(seed['original'].query_id) for seed in collection.values())))
    lookup={q:i for i,q in enumerate(ids)}
    rows=[]
    def metric(t,w,name):
        den=w.sum(axis=-1)
        total=lambda c:w@t[c].to_numpy(float)
        ratio=lambda a,b:np.divide(a,b,out=np.full_like(np.asarray(a,dtype=float),np.nan),where=b>0)
        if name=='evidence_precision':return ratio(total('correct'),total('admitted'))
        if name=='stress_precision':return ratio(total('stress_correct'),total('stress_admitted'))
        if name=='coverage':return total('covered')/den
        if name=='unconditional_query_error':return total('has_error')/den
        if name=='conditional_query_error':return ratio(total('has_error'),total('covered'))
        if name=='volume_per_100':return 100*total('admitted')/den
        positive=t.gold_count.to_numpy(float)>0
        if name=='evidence_recall':
            values=np.divide(t.correct,t.gold_count,out=np.zeros(len(t)),where=positive)
        else:values=t.all_facts.to_numpy(float)
        return ratio(w@(values*positive),w@positive.astype(float))
    # Process a few draws at a time to bound memory on FEVER.
    rng=np.random.default_rng(20261016)
    distributions={(method,name):[] for method in methods for name in metrics}
    points={(method,name):[] for method in methods for name in metrics}
    for seed,records in collection.items():
        b=records['original'].set_index('query_id')
        for method in methods:
            a=records[method].set_index('query_id').reindex(b.index)
            for name in metrics:
                points[method,name].append(float(metric(a,np.ones(len(a)),name)-metric(b,np.ones(len(b)),name)))
    for start in range(0,1000,25):
        draws=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=min(25,1000-start)).astype(float)
        local={key:[] for key in distributions}
        for seed,records in collection.items():
            b=records['original'].set_index('query_id')
            w=draws[:,[lookup[q] for q in b.index]]
            for method in methods:
                a=records[method].set_index('query_id').reindex(b.index)
                for name in metrics:
                    local[method,name].append(metric(a,w,name)-metric(b,w,name))
        for key,values in local.items():
            distributions[key].extend(np.nanmean(values,axis=0).tolist())
    for (method,name),values in distributions.items():
        lo,hi=np.nanquantile(values,[.025,.975])
        rows.append({'dataset':dataset,'method':method,'reference':'original','metric':name,
                     'difference':float(np.mean(points[method,name])),'ci_low':lo,'ci_high':hi,
                     'query_clusters':len(ids),'repetitions':1000,
                     'scope':'pointwise paired query-cluster intervals, shared weights across partitions, frozen fitted models and decisions'})
    return rows


def run(stage,datasets,seeds):
    start=time.monotonic()
    out=OUT/stage
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'protocol.json',PLAN)
    if stage=='full':
        gate=json.loads((OUT/'smoke/audit.json').read_text())
        assert gate['passed'] and gate['code_sha256']==digest(Path(__file__))
    summaries,choices,intervals=[],[],[]
    for dataset in datasets:
        root=ROOT/f'results/experiments/{dataset}_full_k100'
        queries=json.loads((ROOT/f'data/processed/{dataset}_full_queries.json').read_text(encoding='utf-8'))
        collections={name:{} for name in ['prospective','exact_budget']}
        for seed in seeds:
            path=root/f'model_seed{seed}.json'
            original=json.loads(path.read_text())
            fitids,calids,testids=[original[k] for k in ['fitting_query_ids','threshold_query_ids','evaluation_query_ids']]
            assert not(set(fitids)&set(calids) or set(fitids)&set(testids) or set(calids)&set(testids))
            fitting,calibration=load(dataset,fitids),load(dataset,calids)
            models={}
            for variant in PLAN['variants']:
                fitted=fit(fitting,calibration,original,variant)
                fitted['original_model_sha256']=digest(path)
                dump(out/f'model_{dataset}_seed{seed}_{variant}.json',fitted)
                models[variant]=fitted
                choices.append({'dataset':dataset,'seed':seed,'method':variant,
                                'selected_form':fitted['selected_form'],'feature_count':fitted['feature_count'],
                                'original_selected_form':original['selected_form'], 'original_penalty':original['penalty']})
                if stage=='smoke':
                    a=score(calibration,fitted)
                    changed=calibration.iloc[::-1].copy().reset_index(drop=True)
                    changed['gold_evidence']=1-changed.gold_evidence
                    assert np.allclose(score(changed,fitted)[::-1],a,atol=1e-12,rtol=0)
                    assert not {'evidence_text','query_text'}.intersection(calibration.columns)
                    assert 'penalty' not in fitted and fitted['external_penalty']==0
            if stage=='full':
                # Test candidates are read only after every variant is frozen.
                test=load(dataset,testids)
                shared=pd.read_parquet(root/f'shared_stress_seed{seed}.parquet')
                if np.array_equal(test[['query_id','candidate_id']].to_numpy(),shared[['query_id','candidate_id']].to_numpy()):
                    stress=shared.stress.to_numpy(bool)
                else:
                    align=test[['query_id','candidate_id']].merge(shared,on=['query_id','candidate_id'],how='left',validate='one_to_one',sort=False)
                    assert align.stress.notna().all()
                    stress=align.stress.to_numpy(bool)
                for analysis in collections:
                    file=(root/f'query_seed{seed}_scet.csv') if analysis=='prospective' else (ROOT/f'results/strengthening/{dataset}_full_k100/query_seed{seed}_scet.csv')
                    baseline=pd.read_csv(file,dtype={'query_id':str})
                    assert set(baseline.query_id)==set(testids)
                    collections[analysis][seed]={'original':baseline}
                    summaries.append({'dataset':dataset,'seed':seed,'analysis':analysis,'method':'original',**summarize(baseline)})
                for variant,fitted in models.items():
                    values=score(test,fitted)
                    for analysis in collections:
                        rule=fitted['threshold'] if analysis=='prospective' else exact_rule(test,values)
                        stats=query_statistics(test,values,rule,stress,{q:queries[q] for q in testids})
                        if analysis=='exact_budget':assert stats.admitted.sum()==len(testids)
                        stats.to_csv(out/f'query_{dataset}_seed{seed}_{variant}_{analysis}.csv',index=False)
                        collections[analysis][seed][variant]=stats
                        summaries.append({'dataset':dataset,'seed':seed,'analysis':analysis,'method':variant,**summarize(stats)})
            print(dataset,seed,stage,{name:model['selected_form'] for name,model in models.items()},'seconds',round(time.monotonic()-start,1),flush=True)
        if stage=='full':
            for analysis,collection in collections.items():
                intervals.extend({'analysis':analysis,**row} for row in bootstrap(collection,dataset))
            # Persist per dataset, allowing partial completed work to be reviewed.
            table=pd.DataFrame(summaries)
            table.to_csv(out/'evaluation_by_partition.csv',index=False)
            aggregate(table,['dataset','analysis','method']).to_csv(out/'evaluation_mean_sd.csv',index=False)
            pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'fitting_choices.csv',index=False)
    dump(out/'audit.json',{'passed':True,'stage':stage,'code_sha256':digest(Path(__file__)),
                         'plan':PLAN,'test_outcomes_used_for_selection':False,'core_method_changed':False,
                         'smoke_read_test':False,'smoke_label_and_order_invariance':stage=='smoke',
                         'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['smoke','full'],default='smoke')
    parser.add_argument('--datasets',nargs='+',choices=PLAN['datasets'])
    parser.add_argument('--seeds',nargs='+',type=int)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.stage,args.datasets or (['scifact'] if args.stage=='smoke' else PLAN['datasets']),
            args.seeds or ([13] if args.stage=='smoke' else PLAN['seeds']))
