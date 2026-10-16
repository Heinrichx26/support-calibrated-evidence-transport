"""One frozen, source-separated FEVER confirmation with four primary contrasts."""
from pathlib import Path
import argparse, hashlib, json, pickle, time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import r5_context_gain as dev
import r5_simple_controls as simple
import evidence_set_freeze as setfreeze

R=Path(__file__).resolve().parents[1]
D=R/'data/independent_confirmation'
O=R/'results/independent_confirmation'
F=R/'results/evidence_set_confirmation/balanced/frozen'
SEEDS=[13,17,29,7]
PRIMARY=[
    ('list_all','list_ce','prospective','evidence_precision'),
    ('list_all','list_ce','fixed_batch','evidence_precision'),
    ('set_all','set_ce','fixed_batch','evidence_precision'),
    ('set_all','list_all','fixed_batch','multiple_set_recall')]
PLAN={
    'created':'2026-10-16','seeds':SEEDS,
    'models':'All eight original list-feature and eight fitting-selected evidence-set models are frozen; no new-data model or beta choice.',
    'population':'New FEVER training claims with normalized text distinct from original development and all annotated pages absent from every prior FEVER k100 candidate page; support/refute/NEI equally sampled.',
    'calibration':'The 600 independent calibration claims determine score/tie thresholds at 100 sentences per 100 queries, max five per query. Calibration labels are not used.',
    'evaluation':'The 3000 independent evaluation claims are evaluated once after model, program, data and threshold hashes are recorded. Exact-batch allocation uses scores and tie keys only.',
    'primaries':[dict(method=a,reference=b,protocol=c,metric=d) for a,b,c,d in PRIMARY],
    'bootstrap_repetitions':2000,'bootstrap_seed':2026101609,
    'bootstrap':'Paired annotated-page-component multinomial bootstrap; evidence-free claims are individual components. Draw one shared component count per replicate, assign it to member claims, normalize within verdict to preserve the equally sampled verdict strata, and share the resulting query weights across every method and all four frozen models. Trained models and admission decisions stay fixed.',
    'intervals':'Four primary contrasts use two-sided Bonferroni 95% simultaneous intervals (0.00625,0.99375 quantiles); all other intervals are pointwise 95%.',
    'risk':'Report actual unconditional query error and error among covered queries. No independent-identically-distributed finite-sample risk guarantee is assigned to the new stratified grouped sample.',
    'smoke':'60 separate claims; technical and numerical verification only. The fitting-only development experiment established the direction. Smoke gain never changes model, beta, sample or endpoint choice.',
    'all_results_retained':True}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,obj): Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=True)+'\n',encoding='utf-8')

def identity():
    freeze=load(F/'freeze.json')
    for rec in freeze['models']:
        assert sha(F/rec['path'])==rec['sha256']
    sources=[Path(__file__),Path(setfreeze.__file__),Path(dev.__file__),Path(simple.__file__),Path(setfreeze.base.__file__)]
    return {'code_sha256':{p.name:sha(p) for p in sources},'set_freeze_sha256':sha(F/'freeze.json'),
        'set_models_sha256':{p.name:sha(p) for p in sorted(F.glob('model_*.pkl'))},
        'list_models_sha256':{p.name:sha(p) for p in sorted((dev.O/'development').glob('model_*.pkl'))},
        'sample_plan_sha256':sha(D/'frozen_sample_plan.json'),'lexical_state_sha256':sha(D/'frozen_lexical.pkl')}

def unlabelled_predictions(stage):
    raw=pd.read_parquet(D/f'scores/{stage}/scored.parquet')
    assert not any(c.startswith('gold') or c in ['label','claim_verdict'] for c in raw)
    original=pd.read_parquet(D/f'scores/{stage}/frozen_predictions.parquet')
    original['gold_evidence']=0
    qids=sorted(raw.query_id.unique())
    dummy={q:{'gold_sets':[],'gold_count':0} for q in qids}
    x,keep=dev.list_inputs(raw)
    screened=raw.loc[keep].reset_index(drop=True);x=x[keep]
    screened['gold_evidence']=0
    result={};checks=[]
    for seed in SEEDS:
        for group in ['all','ce']:
            name='list_'+group
            score=original[f'score_{seed}_{group}'].to_numpy(float)
            data=simple.compact(original,score,dummy)
            result[(seed,name)]=(data,data['raw'])
            with (F/f'model_seed{seed}_{group}.pkl').open('rb') as handle:bundle=pickle.load(handle)
            # Cache the label-free list features across all eight frozen utility bundles.
            sdata,features,_=setfreeze.base.prepared(screened,x,bundle['base'],dummy)
            utility=setfreeze.base.scores(bundle['utility'],sdata,features,bundle['beta'])
            assert np.isfinite(utility).all()
            result[(seed,'set_'+group)]=(sdata,utility)
            if stage=='smoke' and seed==13:
                hd,hs=setfreeze.predict_without_labels(bundle,raw)
                assert np.array_equal(hd['qids'],sdata['qids']) and np.array_equal(hd['cids'],sdata['cids'])
                np.testing.assert_array_equal(hs,utility)
                changed=raw.iloc[::-1].reset_index(drop=True).copy();changed['gold_evidence']=1
                rd,rs=setfreeze.predict_without_labels(bundle,changed)
                assert np.array_equal(rd['qids'],sdata['qids']) and np.array_equal(rd['cids'],sdata['cids'])
                np.testing.assert_array_equal(rs,utility)
                checks.append({'seed':seed,'group':group,'helper_exact':True,'label_and_order_invariance':True})
    return result,checks

def attach(data,labels,queries):
    data=dict(data); data.pop('_groups',None)
    keys=list(zip(data['qids'][data['codes']],data['cids']))
    data['y']=np.asarray([labels[k] for k in keys],int)
    data['queries']=queries
    return data

def describe(stat):
    return {**simple.summary(stat),
        'conditional_query_error':float(stat.has_error.sum()/stat.covered.sum()) if stat.covered.sum() else np.nan,
        'multiple_set_queries':int((stat.minimum_set_size>1).sum())}

def weight_matrix(queries,qids,repetitions):
    groups=[queries[q]['annotation_page_group'] for q in qids]
    ordered=sorted(set(groups));lookup={g:i for i,g in enumerate(ordered)}
    codes=np.asarray([lookup[g] for g in groups],int)
    verdict=np.asarray([queries[q]['claim_verdict'] for q in qids])
    rng=np.random.default_rng(PLAN['bootstrap_seed']);chunks=[];got=0
    while got<repetitions:
        size=repetitions-got
        counts=rng.multinomial(len(ordered),np.full(len(ordered),1/len(ordered)),size=size)
        w=counts[:,codes].astype(float)
        valid=np.ones(size,bool)
        for value in np.unique(verdict):valid&=w[:,verdict==value].sum(axis=1)>0
        w=w[valid]
        for value in np.unique(verdict):
            mask=verdict==value; w[:,mask]*=(mask.sum()/w[:,mask].sum(axis=1))[:,None]
        chunks.append(w.astype(np.float32));got+=len(w)
    return np.concatenate(chunks),len(ordered)

def weighted(stat,w,name):
    if name=='evidence_precision':num,den=stat.correct,stat.admitted
    elif name=='all_facts_recall':num,den=stat.all_facts*(stat.gold_count>0),stat.gold_count>0
    elif name=='multiple_set_recall':num,den=stat.all_facts*(stat.minimum_set_size>1),stat.minimum_set_size>1
    elif name=='coverage':num,den=stat.covered,np.ones(len(stat))
    elif name=='query_error_rate':num,den=stat.has_error,np.ones(len(stat))
    elif name=='conditional_query_error':num,den=stat.has_error,stat.covered
    elif name=='false_admission_no_evidence':num,den=stat.covered*(stat.gold_count==0),stat.gold_count==0
    elif name=='volume_per_100':num,den=100*stat.admitted,np.ones(len(stat))
    elif name=='evidence_recall':
        den=stat.gold_count>0;num=np.divide(stat.correct,stat.gold_count,out=np.zeros(len(stat),float),where=den)*den
    else:raise ValueError(name)
    a=w@np.asarray(num,float);b=w@np.asarray(den,float)
    return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)

def intervals(collection,queries,out,repetitions):
    qids=sorted(queries);w,components=weight_matrix(queries,qids,repetitions)
    ones=np.ones((1,len(qids)),float)
    metrics=['evidence_precision','all_facts_recall','multiple_set_recall','coverage','query_error_rate',
             'conditional_query_error','false_admission_no_evidence','volume_per_100','evidence_recall']
    cache={};points={}
    for key,stat in collection.items():
        stat=stat.set_index('query_id').reindex(qids).reset_index();assert stat.admitted.notna().all()
        for metric in metrics:
            cache[key+(metric,)]=weighted(stat,w,metric)
            points[key+(metric,)]=float(weighted(stat,ones,metric)[0])
    pairs=[('list_all','list_ce'),('set_all','set_ce'),('set_all','list_all'),('set_ce','list_ce'),('set_all','list_ce'),('list_all','set_ce')]
    rows=[]
    for method,reference in pairs:
        for protocol in ['prospective','fixed_batch']:
            for metric in metrics:
                samples=np.mean([cache[(s,method,protocol,metric)]-cache[(s,reference,protocol,metric)] for s in SEEDS],axis=0)
                point=float(np.mean([points[(s,method,protocol,metric)]-points[(s,reference,protocol,metric)] for s in SEEDS]))
                finite=samples[np.isfinite(samples)]
                if len(finite):
                    low,high=np.quantile(finite,[.025,.975]);sl,sh=np.quantile(finite,[.00625,.99375])
                else:low=high=sl=sh=np.nan
                primary=(method,reference,protocol,metric) in PRIMARY
                rows.append({'method':method,'reference':reference,'protocol':protocol,'metric':metric,
                    'difference':point,'ci_low':low,'ci_high':high,'primary':primary,
                    'four_primary_simultaneous_low':sl if primary else np.nan,
                    'four_primary_simultaneous_high':sh if primary else np.nan,
                    'bootstrap_repetitions':repetitions,'finite_repetitions':len(finite),'source_components':components,
                    'conditioning':'Fixed models and decisions; paired source-component draws with verdict-stratum normalization shared across seeds.'})
    frame=pd.DataFrame(rows);frame.to_csv(out/'paired_intervals.csv',index=False)
    frame[frame.primary].to_csv(out/'primary_intervals.csv',index=False)
    return frame

def run(smoke):
    started=time.monotonic();O.mkdir(exist_ok=True,parents=True)
    plan=O/'evaluation_plan.json'
    if plan.exists():assert load(plan)==PLAN
    else:dump(plan,PLAN)
    record=identity();stage='smoke' if smoke else 'evaluation'
    out=O/('evaluation_smoke' if smoke else 'evaluation_full');out.mkdir(exist_ok=True)
    assert not (out/'audit.json').exists(),'Saved evaluation exists; preserve the recorded evaluation.'
    if not smoke:
        gate=load(O/'evaluation_smoke/audit.json');assert gate['passed']
        assert gate['identity']==record and gate['plan_sha256']==sha(plan)
    for part in ['calibration',stage]:assert load(O/f'scoring_{part}_audit.json')['passed']
    before={'identity':record,'plan_sha256':sha(plan),'stage':stage,'smoke':smoke,
        'scored_sha256':{part:sha(D/f'scores/{part}/scored.parquet') for part in ['calibration',stage]},
        'evaluation_labels_opened':False,'model_or_endpoint_selection_permitted':False}
    marker=out/'before_prediction.json'
    if marker.exists():assert load(marker)==before
    else:dump(marker,before)
    cal,checks1=unlabelled_predictions('calibration');test,checks2=unlabelled_predictions(stage)
    thresholds=[];decisions={};predicted=[]
    for key,(data,score) in test.items():
        seed,name=key;cd,cs=cal[key]
        for protocol,rule in [('prospective',simple.threshold(cd,cs,100)),('fixed_batch',simple.threshold(data,score,100))]:
            thresholds.append({'seed':seed,'method':name,'protocol':protocol,**rule})
            mask=simple.admitted(data,score,rule)
            assert np.bincount(data['codes'],weights=mask,minlength=len(data['qids'])).max()<=5
            if protocol=='fixed_batch':assert mask.sum()==len(data['qids'])
            decisions[(seed,name,protocol)]=mask
        predicted.append(pd.DataFrame({'seed':seed,'method':name,'query_id':data['qids'][data['codes']],
            'candidate_id':data['cids'],'tie_key':data['ties'],'rank':data['ranks'],'score':score}))
    pd.DataFrame(thresholds).to_csv(out/'thresholds.csv',index=False)
    pd.concat(predicted,ignore_index=True).to_parquet(out/'candidate_predictions.parquet',index=False)
    dump(out/'before_labels.json',{**before,'thresholds_sha256':sha(out/'thresholds.csv'),
        'predictions_sha256':sha(out/'candidate_predictions.parquet'),'all_decisions_fixed':True})
    labelsframe=pd.read_parquet(D/f'{stage}_labels.parquet')
    labels={(q,c):int(y) for q,c,y in labelsframe.itertuples(index=False,name=None)}
    queries=load(D/f'{stage}_queries.json')
    collection={};rows=[];strata=[];sensitivity=[]
    for (seed,name),(data,score) in test.items():
        data=attach(data,labels,queries)
        assert set(data['qids'])==set(queries)
        for protocol in ['prospective','fixed_batch']:
            stat=simple.statistics(data,decisions[(seed,name,protocol)])
            stat['claim_verdict']=stat.query_id.map({q:v['claim_verdict'] for q,v in queries.items()})
            stat['annotation_page_group']=stat.query_id.map({q:v['annotation_page_group'] for q,v in queries.items()})
            stat['fully_verified']=stat.query_id.map({q:v['fully_verified'] for q,v in queries.items()})
            stat['strict_100']=stat.query_id.map({q:v['present_candidates']==100 for q,v in queries.items()})
            stat.to_csv(out/f'query_seed{seed}_{name}_{protocol}.csv',index=False)
            collection[(seed,name,protocol)]=stat
            row={'seed':seed,'method':name,'protocol':protocol,**describe(stat)};rows.append(row)
            for verdict,part in stat.groupby('claim_verdict'):
                strata.append({'seed':seed,'method':name,'protocol':protocol,'verdict':verdict,**describe(part)})
            for field in ['fully_verified','strict_100']:
                for value in [True,False]:
                    part=stat[stat[field]==value]
                    if len(part):sensitivity.append({'seed':seed,'method':name,'protocol':protocol,'subset':field,'value':value,**describe(part)})
    frame=pd.DataFrame(rows);frame.to_csv(out/'by_model.csv',index=False)
    numeric=[c for c in frame.select_dtypes(include='number') if c!='seed']
    means=frame.groupby(['method','protocol'])[numeric].agg(['mean','std'])
    means.columns=['_'.join(c) for c in means.columns];means.to_csv(out/'mean_sd.csv')
    pd.DataFrame(strata).to_csv(out/'by_verdict.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'text_availability_sensitivity.csv',index=False)
    paired=intervals(collection,queries,out,200 if smoke else PLAN['bootstrap_repetitions'])
    audit={'passed':True,'smoke':smoke,'identity':record,'plan_sha256':sha(plan),'thresholds_without_labels':True,
        'all_decisions_fixed_before_labels':True,'all_results_retained':True,'models_retrained':False,
        'model_selection_on_confirmation':False,'finite_order_label_checks':checks1+checks2,
        'queries':len(queries),'elapsed_seconds':time.monotonic()-started}
    dump(out/'audit.json',audit)
    print(means[['evidence_precision_mean','all_facts_recall_mean','multiple_set_recall_mean','coverage_mean']].to_string(),flush=True)
    print(paired[paired.primary].to_string(index=False),flush=True)
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--smoke',action='store_true');g.add_argument('--full',action='store_true')
    with threadpool_limits(limits=2):run(p.parse_args().smoke)
