"""R3 analyses of frozen R2 outputs. Run --smoke before the full analysis."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'2')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

R3=Path(__file__).resolve().parents[1]
R2=Path(os.environ.get('R2_DIR',R3.parent/'r2')).resolve()
sys.path.insert(0,str(R2/'src'))
import r2_admission as admission
from r2_statistics import query_statistics
import analyze_fever_missingness as missing
SEEDS=[13,17,29,7]
OUT=R3/'results/analysis'
OUT.mkdir(parents=True,exist_ok=True)

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
def read(p):return pd.read_csv(p,dtype={'query_id':str})
def describe(s):
    gold=s.gold_count.gt(0); n=len(s); total=s.admitted.sum()
    return dict(queries=n,admitted=int(total),correct=int(s.correct.sum()),precision=s.correct.sum()/total if total else np.nan,
        items=100*total/n,coverage=s.covered.mean(),zero=1-s.covered.mean(),
        evidence_recall=(s.loc[gold,'correct']/s.loc[gold,'gold_count']).mean(),
        set_recall=s.loc[gold,'all_facts'].mean(),no_evidence=s.loc[~gold,'covered'].mean(),
        unannotated_fraction=1-s.correct.sum()/total if total else np.nan,
        evidence_bearing_precision=s.loc[gold,'correct'].sum()/s.loc[gold,'admitted'].sum() if s.loc[gold,'admitted'].sum() else np.nan,
        nei_item_share=s.loc[~gold,'admitted'].sum()/total if total else np.nan)
def aggregate(f,keys):
    nums=[x for x in f.select_dtypes('number').columns if x not in keys+['seed']]
    a=f.groupby(keys,dropna=False)[nums].agg(['mean','std']); a.columns=['_'.join(x) for x in a.columns]
    return a.reset_index()
def group_stats(s,meta):
    result=[{**meta,'verdict':'ALL',**describe(s)}]
    if 'claim_verdict' in s:
        result += [{**meta,'verdict':v,**describe(g)} for v,g in s.groupby('claim_verdict')]
    return result
def exact_rule(frame,score,budget,cap=5):
    rank=admission.ranks(frame,score); ix=np.flatnonzero(rank<=cap)
    order=ix[np.lexsort((frame.tie_key.to_numpy(str)[ix],-score[ix]))]
    b=order[budget-1]
    return {'threshold':float(score[b]),'tie_cutoff':str(frame.iloc[b].tie_key),'eligible_rank':cap},order

def smoke():
    m=load(R2/'results/experiments/fever_full_k100/model_seed13.json')
    qs=load(R2/'data/processed/fever_full_queries.json')
    av=read(R2/'results/evidence_completion/fever_missingness/full/query_availability.csv')
    ids=[]
    for flag in [True,False]:
        for verdict in ['SUPPORTS','REFUTES','NOT_ENOUGH_INFO']:
            ids+=sorted(set(av.loc[av.all_candidates_complete.eq(flag)&av.claim_verdict.eq(verdict),'query_id'])&set(m['evaluation_query_ids']))[:2]
    f=pd.read_parquet(R2/'data/scored/fever_full_k100.parquet',filters=[('query_id','in',ids)])
    scores,stress,_=admission.score_pipeline(f,m)
    checks=[]
    for method in ['scet','ce']:
        got=query_statistics(f,scores[method],m['thresholds'][method],stress,{i:qs[i] for i in ids}).set_index('query_id')
        expected=read(R2/f'results/experiments/fever_full_k100/query_seed13_{method}.csv').set_index('query_id').loc[got.index]
        for col in ['admitted','correct','covered','all_facts']:
            np.testing.assert_array_equal(got[col],expected[col])
        rule,_=exact_rule(f,scores[method] if method=='scet' else f.raw_ce.to_numpy(float),len(ids))
        s=query_statistics(f,scores[method] if method=='scet' else f.raw_ce.to_numpy(float),rule,stress,{i:qs[i] for i in ids})
        assert s.admitted.sum()==len(ids)
        checks.append({'method':method,'cached_outcomes_match':True,'exact_budget':int(s.admitted.sum())})
    dump(OUT/'smoke.json',{'passed':True,'queries':len(ids),'checks':checks,'models_fitted':False,'neural_inference':False})
    print('Smoke passed: frozen outcomes reproduced; exact-budget selection verified.',flush=True)

def main_tables():
    records=[]; selected=[]
    simple=R2/'results/r5_simple_controls/full'
    for ds in ['scifact','fever','hover']:
        qs=load(R2/f'data/processed/{ds}_full_queries.json')
        for seed in SEEDS:
            model=load(R2/f'results/experiments/{ds}_full_k100/model_seed{seed}.json')
            selected.append({'dataset':ds,'seed':seed,'form':model['selected_form'],'penalty':model['penalty'],'selected_single':model['primary_single'],'fitting_n':len(model['fitting_query_ids']),'calibration_n':len(model['threshold_query_ids'])})
            qdir=OUT/'query'/ds/f'seed{seed}';qdir.mkdir(parents=True,exist_ok=True)
            for method,label in [('frozen_scet','Admission'),('raw_cap5','Cross-encoder'),('raw_top1','Cross-encoder top-one'),('list_logit_cap5','Cross-encoder list-cutoff')]:
                s=pd.read_parquet(simple/f'{ds}_ce/seed{seed}/query_{method}_exact_budget_100.parquet')
                s['claim_verdict']=s.query_id.map({q:v['claim_verdict'] for q,v in qs.items()})
                s.to_csv(qdir/f'{method}.csv',index=False)
                records+=group_stats(s,{'dataset':ds,'seed':seed,'method':label,'protocol':'fixed_batch'})
            ids=model['evaluation_query_ids']
            columns=['query_id','candidate_id','tie_key','gold_evidence','raw_e5','raw_nli_entailment','raw_'+model['primary_single']]
            f=pd.read_parquet(R2/f'data/scored/{ds}_full_k100.parquet',columns=list(dict.fromkeys(columns)),filters=[('query_id','in',ids)])
            for method,col in [('Selected single','raw_'+model['primary_single']),('E5','raw_e5'),('NLI entailment','raw_nli_entailment')]:
                score=f[col].to_numpy(float);rule,_=exact_rule(f,score,len(ids))
                stat=query_statistics(f,score,rule,np.zeros(len(f),bool),{q:qs[q] for q in ids})
                assert stat.admitted.sum()==len(ids)
                records+=group_stats(stat,{'dataset':ds,'seed':seed,'method':method,'protocol':'fixed_batch'})
                stat.to_csv(qdir/f'{method.replace(" ","_")}.csv',index=False)
            for method in ['scet','ce']:
                s=read(R2/f'results/experiments/{ds}_full_k100/query_seed{seed}_{method}.csv')
                records+=group_stats(s,{'dataset':ds,'seed':seed,'method':{'scet':'Admission','ce':'Cross-encoder'}[method],'protocol':'prospective'})
        print('Summarized',ds,flush=True)
    for stage,prefix,methods in [('development',R2/'results/r5_context_gain/evaluation', [('all','Ranked-list'),('ce','Cross-encoder feature')]),('confirmation',R2/'results/independent_confirmation/evaluation_full',[('list_all','Ranked-list'),('list_ce','Cross-encoder feature'),('set_all','Evidence-set'),('set_ce','Cross-encoder evidence-set')])]:
        for seed in SEEDS:
            for protocol in ['fixed_batch','prospective']:
                for name,label in methods:
                    s=read(prefix/f'query_seed{seed}_{name}_{protocol}.csv')
                    records+=group_stats(s,{'dataset':'fever_'+stage,'seed':seed,'method':label,'protocol':protocol})
    table=pd.DataFrame(records);table.to_csv(OUT/'stratified_per_seed.csv',index=False)
    aggregate(table,['dataset','method','protocol','verdict']).to_csv(OUT/'stratified_mean_sd.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'selected_forms.csv',index=False)

def pool_audit():
    f=pd.read_parquet(R2/'data/scored/fever_full_k100.parquet',columns=['query_id','candidate_id','gold_evidence','raw_ce','tie_key','nli_contradiction'])
    qs=load(R2/'data/processed/fever_full_queries.json')
    f['page']=f.candidate_id.str.rsplit('::',n=1).str[0]
    f['rank']=admission.ranks(f,f.raw_ce.to_numpy(float))
    annotated={q for q,v in qs.items() if v['gold_count']>0}
    has={q for q,v in qs.items() if v['verified_gold_count']>0}
    e=f[f.query_id.isin(has)];gold=e[e.gold_evidence.eq(1)]
    first=gold.groupby('query_id')['rank'].min().reindex(sorted(has),fill_value=10000)
    same=set(zip(gold.query_id,gold.page));neg=e[e.gold_evidence.eq(0)]
    samefrac=np.mean([(q,p) in same for q,p in zip(neg.query_id,neg.page)])
    median=float(f.loc[f.gold_evidence.eq(1),'raw_ce'].median())
    nei=f[~f.query_id.isin(annotated)];topnei=nei[nei['rank'].eq(1)]
    found5=gold[gold['rank'].le(5)].groupby('query_id').size().reindex(sorted(has),fill_value=0)
    original=np.array([qs[q]['gold_count'] for q in found5.index])
    retained=np.array([qs[q]['verified_gold_count'] for q in found5.index])
    audit=dict(rows=len(f),distinct_sentences=int(f.candidate_id.nunique()),original_evidence_queries=len(annotated),evidence_available_queries=len(has),queries_with_all_annotated_text_missing=len(annotated-has),top1_hit=float(first.le(1).mean()),recall5_original_denominator=float((found5.to_numpy()/original).mean()),recall5_available_denominator=float((found5.to_numpy()/retained).mean()),recall5_claim_hit=float(first.le(5).mean()),same_page_unannotated_fraction=float(samefrac),annotated_score_median=median,nei_top_above_median=float(topnei.raw_ce.ge(median).mean()),contradiction_gt_half_unannotated=float(f.loc[f.gold_evidence.eq(0),'nli_contradiction'].gt(.5).mean()))
    dump(OUT/'candidate_pool_audit.json',audit)
    # Verify source examples in the frozen input, without claiming human annotation.
    qids=[q for q,v in qs.items() if 'Kellogg' in v['query_text'] or 'Island Records' in v['query_text']]
    ex=pd.read_parquet(R2/'data/scored/fever_full_k100.parquet',columns=['query_id','query_text','candidate_id','evidence_text','raw_ce','tie_key','gold_evidence'],filters=[('query_id','in',qids)])
    ex['rank']=admission.ranks(ex,ex.raw_ce.to_numpy(float))
    ex=ex[ex['rank'].eq(1)];ex.to_csv(OUT/'nei_source_examples.csv',index=False,encoding='utf-8-sig')
    print('Pool audit',audit,flush=True)

def missing_budget():
    assert load(OUT/'smoke.json')['passed']
    av=read(R2/'results/evidence_completion/fever_missingness/full/query_availability.csv')
    ids=set(av.loc[~av.all_candidates_complete,'query_id'])
    qs=load(R2/'data/processed/fever_full_queries.json')
    saved=missing.saved_outcomes()
    dest=OUT/'missing_candidates';dest.mkdir(exist_ok=True)
    missing.METHODS=['scet','ce']
    missing.fixed_budget(dest,qs,ids,saved)
    for p in dest.glob('fixed_budget*.csv'):
        t=read(p)
        if 'subset' in t:t['subset']='candidate_incomplete'
        t.to_csv(p,index=False)

def frontiers():
    # FEVER was completed separately in R2; per-partition records are authoritative.
    files=[R2/f'results/r5_simple_controls/full/{ds}_ce/seed{seed}/summary.csv' for ds in ['scifact','fever','hover'] for seed in SEEDS]
    s=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    s=s[(s.source=='ce')&s.analysis.str.startswith('workload_')&s.method.isin(['frozen_scet','raw_cap5','raw_top1','list_logit_cap5'])].copy()
    s['target']=s.analysis.str.replace('workload_','').astype(int)
    s['method']=s.method.map({'frozen_scet':'Admission','raw_cap5':'Cross-encoder','raw_top1':'Cross-encoder top-one','list_logit_cap5':'Cross-encoder list-cutoff'})
    for ds in ['scifact','fever','hover']:
        qs=load(R2/f'data/processed/{ds}_full_queries.json')
        for seed in SEEDS:
            m=load(R2/f'results/experiments/{ds}_full_k100/model_seed{seed}.json')
            values=np.array([qs[q]['verified_gold_count'] for q in m['evaluation_query_ids']]);pe=(values>0).mean()
            for ix,r in s[(s.dataset==ds)&(s.seed==seed)].iterrows():
                cap=1 if 'top-one' in r.method else 5
                s.loc[ix,'evidence_available_fraction']=pe
                s.loc[ix,'ceiling_from_query_availability']=min(1,cap*pe/(r.volume_per_100/100))
                s.loc[ix,'ceiling_from_annotated_counts']=min(1,np.minimum(values,cap).mean()/(r.volume_per_100/100))
    s.to_csv(OUT/'frontiers_per_seed.csv',index=False)
    aggregate(s,['dataset','method','target']).to_csv(OUT/'frontiers_mean_sd.csv',index=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();start=time.time()
    with threadpool_limits(limits=2):
        if args.smoke:smoke();return
        assert load(OUT/'smoke.json')['passed']
        main_tables();pool_audit();frontiers();missing_budget()
    dump(OUT/'manifest.json',{'partition_seeds':SEEDS,'source_release_commit':'cadee9083971e6d4f980cd8dcd071c3dc2937f5b','description':'The archive contains score tables, query outcomes, and source examples used in the reported analyses.'})
    print('R3 analysis complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
