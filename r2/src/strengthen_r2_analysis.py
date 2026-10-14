"""Additional workload, probability-reference, and evidence-retention analyses."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.special import expit
from r2_admission import VIEWS, PROTOCOL, ranks, score_pipeline, summarize
from r2_statistics import query_statistics

R2=Path(__file__).resolve().parents[1]
OUT=R2/'results/strengthening'
OUT.mkdir(exist_ok=True)


def budget_rule(frame, score, count):
    rank=ranks(frame,score)
    eligible=np.flatnonzero(rank<=5)
    order=eligible[np.lexsort((frame.tie_key.to_numpy(str)[eligible],-score[eligible]))]
    count=min(count,len(order))
    if not count:return {'threshold':float('inf'),'tie_cutoff':''}
    j=order[count-1]
    return {'threshold':float(score[j]),'tie_cutoff':str(frame.iloc[j].tie_key)}


def paired_interval(stats, comparator):
    allids=sorted(set().union(*(set(s['scet'].query_id) for s in stats.values())))
    lookup={q:i for i,q in enumerate(allids)}
    rng=np.random.default_rng(PROTOCOL['bootstrap_seed'])
    draws=rng.multinomial(len(allids),np.full(len(allids),1/len(allids)),
                          size=PROTOCOL['bootstrap_repetitions']).astype(np.float32)
    differences=[];points=[]
    for seed,methods in stats.items():
        a=methods['scet'].set_index('query_id')
        b=methods[comparator].set_index('query_id').reindex(a.index)
        weights=draws[:,[lookup[q] for q in a.index]]
        aa=weights@a[['correct','admitted']].to_numpy(float)
        bb=weights@b[['correct','admitted']].to_numpy(float)
        differences.append(aa[:,0]/np.maximum(aa[:,1],1)-bb[:,0]/np.maximum(bb[:,1],1))
        points.append(a.correct.sum()/a.admitted.sum()-b.correct.sum()/b.admitted.sum())
    lo,hi=np.quantile(np.mean(differences,axis=0),[.025,.975])
    return {'comparator':comparator,'delta_precision':np.mean(points),'ci_low':lo,'ci_high':hi,
            'bootstrap_repetitions':len(draws),'unique_query_clusters':len(allids)}


def run(ds,pool,smoke):
    start=time.monotonic();tag=f'{ds}_full_k{pool}'
    root=R2/'results/experiments'/tag
    out=OUT/(tag+('_smoke' if smoke else ''));out.mkdir(exist_ok=True)
    frame=pd.read_parquet(R2/f'data/scored/{tag}.parquet')
    allq=json.loads((R2/f'data/processed/{ds}_full_queries.json').read_text(encoding='utf-8'))
    results=[];decomposition=[];probability=[];risk=[];allstats={}
    for seed in ([13] if smoke else PROTOCOL['seeds']):
        model=json.loads((root/f'model_seed{seed}.json').read_text())
        ids=model['evaluation_query_ids']
        if smoke:ids=sorted(ids)[:12]
        test=frame[frame.query_id.isin(ids)].copy().reset_index(drop=True)
        queries={q:allq[q] for q in ids}
        scores,stress,_=score_pipeline(test,model)
        methods={v[4:]:test[v].to_numpy(float) for v in VIEWS}
        methods.update(scet=scores['scet'],conflict_pressure=scores['conflict_pressure'])
        selected=model['primary_single'];methods['selected_single']=methods[selected]
        wanted=len(ids) # predeclared fixed batch budget of 100 items / 100 queries
        stats={}
        for name,score in methods.items():
            rule=budget_rule(test,score,wanted)
            s=query_statistics(test,score,rule,stress,queries)
            assert s.admitted.sum()==wanted,(ds,seed,name)
            stats[name]=s
            s.to_csv(out/f'query_seed{seed}_{name}.csv',index=False)
            results.append({'dataset':ds,'pool':pool,'seed':seed,'method':name,
                            'selected_single':selected,**summarize(s)})
        allstats[seed]=stats
        for name in ['scet','selected_single','ce']:
            score=methods[name]
            available=query_statistics(test,score,{'threshold':float('-inf'),'tie_cutoff':'','eligible_rank':pool},stress,queries)
            eligible=query_statistics(test,score,{'threshold':float('-inf'),'tie_cutoff':'','eligible_rank':5},stress,queries)
            source=stats[name]
            positive=source.gold_count>0
            capacity=np.mean([any(len(set(g))<=5 for g in queries[q]['gold_sets'] if g)
                              for q in source.loc[positive,'query_id']])
            decomposition.append({'dataset':ds,'pool':pool,'seed':seed,'method':name,
                'capacity_ceiling':capacity,'pool_set_recall':available.loc[positive,'all_facts'].mean(),
                'eligible_set_recall':eligible.loc[positive,'all_facts'].mean(),
                'admitted_set_recall':source.loc[positive,'all_facts'].mean()})
        if pool==100:
            qcal=set(model['threshold_query_ids'])
            cal=frame[frame.query_id.isin(qcal)]
            prevalence=float(cal.gold_evidence.mean());y=test.gold_evidence.to_numpy(int)
            existing=pd.read_csv(root/'probability_diagnostic.csv')
            mapping=existing[existing.seed==seed].iloc[0]
            probs=expit(mapping.intercept+mapping.coefficient*scores['scet'])
            brier=np.mean((probs-y)**2);reference=np.mean((prevalence-y)**2)
            assert smoke or np.isclose(brier,mapping.brier,atol=1e-10)
            probability.append({'dataset':ds,'seed':seed,'calibration_prevalence':prevalence,
                'test_prevalence':y.mean(),'mapped_brier':brier,'prior_brier':reference,
                'brier_skill':1-brier/reference,'mapped_ece':np.abs(np.bincount(np.minimum((probs*10).astype(int),9),weights=probs-y,minlength=10)).sum()/len(y),
                'prior_ece':abs(prevalence-y.mean()),'test_pairs':len(y)})
            config=model['risk_control'];entry=next((v for v in config['tests'] if v['threshold']==config['threshold']),None)
            if entry:
                k=entry['error_queries'];n=entry['queries']
                bound=float(beta.ppf(1-config['delta']/len(config['tests']),k+1,n-k)) if k<n else 1.
                assert bound<=config['alpha']+1e-9
                risk.append({'dataset':ds,'seed':seed,'calibration_queries':n,'error_queries':k,'nonempty_feasible':True,'simultaneous_upper_bound':bound,'p_value':entry['p_value']})
            else:
                risk.append({'dataset':ds,'seed':seed,'calibration_queries':len(qcal),'error_queries':0,'nonempty_feasible':False,'simultaneous_upper_bound':0.,'p_value':None})
        print(tag,seed,'additional analyses',round(time.monotonic()-start,1),'s',flush=True)
    pd.DataFrame(results).to_csv(out/'exact_budget.csv',index=False)
    data=pd.DataFrame(results)
    numeric=[x for x in data.select_dtypes(include=np.number).columns if x not in ['pool','seed']]
    summary=data.groupby(['dataset','pool','method'])[numeric].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(out/'exact_budget_mean_sd.csv',index=False)
    pd.DataFrame(decomposition).to_csv(out/'retention_decomposition.csv',index=False)
    if probability:pd.DataFrame(probability).to_csv(out/'probability_reference.csv',index=False)
    if risk:pd.DataFrame(risk).to_csv(out/'risk_bound.csv',index=False)
    if not smoke:
        pd.DataFrame([paired_interval(allstats,n) for n in methods if n!='scet']).to_csv(out/'exact_budget_bootstrap.csv',index=False)
    record={'dataset':ds,'pool':pool,'smoke':smoke,'all_counts_equal':True,
        'budget':100,'eligible_rank':5,'single_scores':'raw, without within-query normalization',
        'selection':'Frozen fitting-selected scorer and SCET form; test batch budget set before outcomes.',
        'interpretation':'Fixed-batch ordering analysis; primary prospective rules retain independent calibration thresholds.',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'analysis_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    p.add_argument('--datasets',nargs='+',default=['scifact','fever','hover']);p.add_argument('--pools',nargs='+',type=int,default=[100,500])
    a=p.parse_args()
    if not a.smoke:
        for ds in a.datasets:
            for pool in a.pools:assert (OUT/f'{ds}_full_k{pool}_smoke/analysis_audit.json').exists(),(ds,pool,'smoke required')
    for ds in a.datasets:
        for pool in a.pools:run(ds,pool,a.smoke)
