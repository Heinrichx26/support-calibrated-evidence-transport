"""One frozen evaluation of finite list-feature candidates and matched controls."""
from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import r2_admission as a
from r2_statistics import query_statistics
import r5_context_gain as dev

R,O=dev.R,dev.O

def metric(s,w,name):
    if name=='precision':p,q=s.correct,s.admitted
    elif name=='complete_set_recall':p,q=s.all_facts,s.gold_count.gt(0)
    elif name=='coverage':p,q=s.covered,np.ones(len(s))
    elif name=='no_evidence_admission':
        q=s.gold_count.eq(0);p=s.covered*q
    else:raise ValueError(name)
    aa=w@np.asarray(p,float);bb=w@np.asarray(q,float)
    return np.divide(aa,bb,out=np.full_like(aa,np.nan),where=bb>0)

def run():
    start=time.monotonic();out=O/'evaluation';out.mkdir(exist_ok=True)
    audit=json.loads((O/'development/audit.json').read_text())
    assert audit['passed'] and audit['code_sha256']==dev.sha(dev.__file__)
    freezes=json.loads((O/'development/freeze.json').read_text())
    models={p.name:dev.sha(p) for p in (O/'development').glob('model_*.pkl')}
    identity={'development_audit_sha256':dev.sha(O/'development/audit.json'),
        'freeze_sha256':dev.sha(O/'development/freeze.json'),'models':models,
        'evaluation_code_sha256':dev.sha(__file__),'training_code_sha256':dev.sha(dev.__file__),
        'created_before_test_outcomes':True,'scope':'Exploratory evaluation of a finite predeclared scoring family on previously studied official development queries.'}
    marker=out/'before_evaluation.json'
    assert not marker.exists(),'One evaluation only; inspect saved outcomes instead of rerunning.'
    marker.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    summaries=[];collection={};invariance=[];calibrations=[]
    queries=json.loads((R/'data/processed/fever_full_queries.json').read_text(encoding='utf-8'))
    allcols=dev.COLS+['query_text','evidence_text','claim_verdict','gold_directional']
    for freeze in freezes:
        seed=freeze['seed'];basepath=R/f'results/experiments/fever_full_k100/model_seed{seed}.json'
        assert dev.sha(basepath)==freeze['base_model_sha256']
        base=json.loads(basepath.read_text())
        calids=freeze['threshold_query_ids'];testids=freeze['evaluation_query_ids']
        assert not set(freeze['fitting_query_ids'])&set(calids)
        assert not (set(freeze['fitting_query_ids'])|set(calids))&set(testids)
        cal=pd.read_parquet(R/'data/scored/fever_full_k100.parquet',columns=dev.COLS,filters=[('query_id','in',calids)])
        test=pd.read_parquet(R/'data/scored/fever_full_k100.parquet',columns=dev.COLS,filters=[('query_id','in',testids)])
        cx,ck=dev.list_inputs(cal);tx,tk=dev.list_inputs(test)
        cal=cal[ck].reset_index(drop=True);cx=cx[ck]
        test=test[tk].reset_index(drop=True);tx=tx[tk]
        qtest={q:queries[q] for q in testids};methods={}
        for group in dev.PLAN['groups']:
            with (O/f'development/model_seed{seed}_{group}.pkl').open('rb') as h:model=pickle.load(h)
            cs=dev.predict(model,cx);ts=dev.predict(model,tx)
            assert np.isfinite(cs).all() and np.isfinite(ts).all()
            assert np.array_equal(ts,dev.predict(model,tx[::-1])[::-1])
            for protocol,rule in [('prospective',a.fit_threshold(cal,cs,100)),('fixed_batch',a.fit_threshold(test,ts,100))]:
                stat=query_statistics(test,ts,rule,np.zeros(len(test),bool),qtest)
                name=f'{group}_{protocol}';methods[name]=stat
                stat.to_csv(out/f'query_seed{seed}_{name}.csv',index=False)
                summaries.append({'seed':seed,'method':group,'protocol':protocol,**a.summarize(stat)})
                calibrations.append({'seed':seed,'method':group,'protocol':protocol,**rule})
                if protocol=='fixed_batch':assert stat.admitted.sum()==len(testids)
            invariance.append({'seed':seed,'method':group,'score_order_invariant':True,'features_exclude_labels':True})
        # Original frozen methods use their complete original candidate pool.
        for protocol,folder in [('prospective','experiments'),('fixed_batch','strengthening')]:
            for name in ['scet','ce','selected_single']:
                source_name=base['primary_single'] if name=='selected_single' and protocol=='prospective' else name
                path=R/f'results/{folder}/fever_full_k100/query_seed{seed}_{source_name}.csv'
                stat=pd.read_csv(path,dtype={'query_id':str}).set_index('query_id').reindex(testids).reset_index()
                assert stat.admitted.notna().all()
                methods[f'original_{name}_{protocol}']=stat
                summaries.append({'seed':seed,'method':'original_'+name,'protocol':protocol,**a.summarize(stat)})
        collection[seed]=methods
        print('FEVER',seed,'evaluated',round(time.monotonic()-start,1),'seconds',flush=True)
    rows=pd.DataFrame(summaries);rows.to_csv(out/'by_partition.csv',index=False)
    numeric=[c for c in rows.select_dtypes(include='number') if c!='seed']
    means=rows.groupby(['method','protocol'])[numeric].agg(['mean','std'])
    means.columns=['_'.join(c) for c in means.columns];means.to_csv(out/'mean_sd.csv')
    pd.DataFrame(calibrations).to_csv(out/'thresholds.csv',index=False)
    allids=sorted(set().union(*(set(next(iter(x.values())).query_id) for x in collection.values())))
    lookup={q:i for i,q in enumerate(allids)};rng=np.random.default_rng(2026101605)
    draws=rng.multinomial(len(allids),np.full(len(allids),1/len(allids)),size=1000).astype(np.float32)
    intervals=[]
    for protocol in ['prospective','fixed_batch']:
        for reference in ['ce','original_scet','original_ce']:
            diffs={metric:[] for metric in ['precision','complete_set_recall','coverage','no_evidence_admission']}
            points={metric:[] for metric in diffs}
            for seed,methods in collection.items():
                aa=methods[f'all_{protocol}'].set_index('query_id')
                bb=methods[f'{reference}_{protocol}'].set_index('query_id').reindex(aa.index)
                weights=draws[:,[lookup[q] for q in aa.index]]
                one=np.ones((1,len(aa)))
                for metric_name in diffs:
                    diffs[metric_name].append(metric(aa,weights,metric_name)-metric(bb,weights,metric_name))
                    points[metric_name].append(float((metric(aa,one,metric_name)-metric(bb,one,metric_name))[0]))
            for metric_name,delta in diffs.items():
                values=np.mean(delta,axis=0)
                lo,hi=np.quantile(values,[.025,.975]);sl,sh=np.quantile(values,[.0125,.9875])
                intervals.append({'method':'all','reference':reference,'protocol':protocol,'metric':metric_name,
                    'difference':float(np.mean(points[metric_name])),'ci_low':lo,'ci_high':hi,
                    'two_primary_protocol_simultaneous_low':sl,'two_primary_protocol_simultaneous_high':sh,
                    'bootstrap_repetitions':1000,'conditioning':'Fixed trained models and observed partitions; one query weight shared across partitions.'})
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    (out/'audit.json').write_text(json.dumps({'passed':True,'code_sha256':dev.sha(__file__),
        'model_freeze_verified':True,'partition_separation_verified':True,'invariance':invariance,
        'all_comparisons_retained':True,'elapsed_seconds':time.monotonic()-start},indent=2),encoding='utf-8')
    print(means[['evidence_precision_mean','all_facts_recall_mean','coverage_mean']].to_string(),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
