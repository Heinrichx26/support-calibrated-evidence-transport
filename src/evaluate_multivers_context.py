"""Paired full-abstract versus single-sentence MultiVerS evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
from run_multivers_extension import paired_intervals, SPEC

R2=Path(__file__).resolve().parents[1]
OUT=R2/'results/multivers_extension/context'


def run(smoke):
    out=OUT/('smoke' if smoke else 'full');out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(R2/'data/scored/scifact_full_k100.parquet')
    paths={'multivers_full':R2/'results/strengthening/multivers/candidate_scores.parquet',
           'multivers_single':R2/'results/multivers_extension/inference_single/candidate_scores.parquet'}
    for name,path in paths.items():
        score=pd.read_parquet(path).rename(columns={'raw_multivers':name})
        frame=frame.merge(score,on=['query_id','candidate_id'],how='left',validate='one_to_one',sort=False)
        assert frame[name].notna().all() and np.isfinite(frame[name]).all()
    queries=json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    summaries=[];collections={};rules=[]
    for seed in ([13] if smoke else SPEC['seeds']):
        model=json.loads((R2/f'results/experiments/scifact_full_k100/model_seed{seed}.json').read_text())
        testids=model['evaluation_query_ids']
        if smoke:testids=sorted(testids)[:12]
        test=frame[frame.query_id.isin(testids)].copy().reset_index(drop=True)
        cal=frame[frame.query_id.isin(model['threshold_query_ids'])].copy().reset_index(drop=True)
        qtest={q:queries[q] for q in testids}
        _,stress,_=admission.score_pipeline(test,model)
        for name in paths:
            ts=test[name].to_numpy(float);cs=cal[name].to_numpy(float)
            for analysis,rule in [('prospective_100',admission.fit_threshold(cal,cs,100)),
                                  ('exact_budget',budget_rule(test,ts,len(testids)))]:
                stat=query_statistics(test,ts,rule,stress,qtest)
                if analysis=='exact_budget':assert stat.admitted.sum()==len(testids)
                stat.to_csv(out/f'query_seed{seed}_{name}_{analysis}.csv',index=False)
                summaries.append({'seed':seed,'method':name,'analysis':analysis,**admission.summarize(stat)})
                collections.setdefault(analysis,{}).setdefault(seed,{})[name]=stat
                rules.append({'seed':seed,'method':name,'analysis':analysis,**rule})
    data=pd.DataFrame(summaries);data.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(rules).to_csv(out/'rules.csv',index=False)
    if not smoke:
        metrics=['evidence_precision','annotated_per_100','volume_per_100','coverage','evidence_recall','all_facts_recall','query_error_rate']
        intervals=[]
        for analysis,stats in collections.items():
            for row in paired_intervals(stats,'multivers_full','multivers_single',metrics):
                intervals.append({'analysis':analysis,**row})
        pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
        previous=pd.read_csv(R2/'results/strengthening/multivers/primary.csv').set_index('seed')
        current=data[(data.analysis=='prospective_100')&(data.method=='multivers_full')].set_index('seed')
        for col in ['evidence_precision','volume_per_100','evidence_recall','all_facts_recall']:
            assert np.allclose(previous[col].sort_index(),current[col].sort_index(),atol=1e-12)
    audit={'smoke':smoke,'pairs':len(frame),'same_checkpoint':True,'same_tokenizer_and_forward':True,
           'candidate_identities_identical':True,'full_context_reproduces_previous_result':not smoke,
           'source_score_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
           'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(data[['seed','method','analysis','evidence_precision','volume_per_100','evidence_recall','all_facts_recall']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not a.smoke:assert json.loads((OUT/'smoke/audit.json').read_text())['candidate_identities_identical']
    run(a.smoke)
