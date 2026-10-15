"""Train-only selection followed by one frozen external test evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
import run_multivers_adaptive_calibration as adaptive
import run_multivers_admission_calibration as calibrated
from infer_multivers_local_context import screens
from evaluate_multivers_confirmation import reselected_precision, fixed_precision

R2=Path(__file__).resolve().parents[1]
DATA=R2/'data/multivers_external'
OUT=R2/'results/multivers_extension/external_evaluation'
INFERENCE=R2/'results/multivers_extension/external_inference'
PLAN=R2/'data/multivers_external_analysis.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_groups(claims):
    parent={str(c['id']):str(c['id']) for c in claims}
    def find(q):
        while parent[q]!=q:
            parent[q]=parent[parent[q]]
            q=parent[q]
        return q
    owners={}
    for c in claims:
        q=str(c['id'])
        for d in c['doc_ids']:
            if d in owners:
                parent[find(q)]=find(owners[d])
            else:
                owners[d]=q
    return {q:find(q) for q in parent}


def label_data(name,split):
    # Call with test only beyond the saved, one-shot evaluation gate.
    claims=list(map(json.loads,(DATA/'data'/name/f'claims_{split}.jsonl').read_text().splitlines()))
    families=source_groups(claims)
    queries={}
    for c in claims:
        q=f"{name}:{split}:{c['id']}"
        support,refute,sets=set(),set(),[]
        for d,groups in c['evidence'].items():
            for g in groups:
                ids={f'{d}::{s}' for s in g['sentences']}
                sets.append(sorted(ids))
                (support if g['label']=='SUPPORT' else refute).update(ids)
        gold=support|refute
        queries[q]={'query_text':c['claim'],'gold_ids':sorted(gold),'support_ids':sorted(support),
            'refute_ids':sorted(refute),'gold_sets':sets,'gold_count':len(gold),'verified_gold_count':len(gold),
            'fully_verified':True,'claim_verdict':'SUPPORT' if support else ('CONTRADICT' if refute else 'NOT_ENOUGH_INFO'),
            'source_family':families[str(c['id'])]}
    return queries


def probability_frame():
    cache=DATA/'probabilities.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    frame=pd.read_parquet(DATA/'inference_candidates.parquet')
    docs={(r['query_id'],r['doc_id']):r for r in map(json.loads,(INFERENCE/'full_scores.jsonl').read_text().splitlines())}
    assert set(frame[['query_id','doc_id']].itertuples(index=False,name=None))==set(docs)
    values=[docs[(q,d)] for q,d in zip(frame.query_id,frame.doc_id)]
    frame['raw_multivers_rationale']=[r['rationale_probabilities'][int(s)] for r,s in zip(values,frame.sentence_id)]
    frame['mv_document_evidence']=[1-r['label_probabilities'][1] for r in values]
    frame['mv_support']=[r['label_probabilities'][2] for r in values]
    frame['mv_refute']=[r['label_probabilities'][0] for r in values]
    frame['raw_multivers']=frame.raw_multivers_rationale*frame.mv_document_evidence
    single=pd.read_json(INFERENCE/'single_scores.jsonl',lines=True,dtype={'query_id':str,'candidate_id':str})
    assert len(single)==len(frame) and not single.duplicated(['query_id','candidate_id']).any()
    frame=frame.merge(single[['query_id','candidate_id','single_rationale']],on=['query_id','candidate_id'],validate='one_to_one',sort=False)
    frame=adaptive.olddual.context.add_context(frame,INFERENCE/'full_scores.jsonl')
    frame['pool']='provided_pool'
    frame['sample_id']=frame.query_id+':'+frame.pool
    for col,prefix in [('raw_multivers_rationale','full'),('single_rationale','single')]:
        s=frame[col].to_numpy(float)
        frame[prefix+'_log_rank']=np.log(admission.ranks(frame,s))
        frame[prefix+'_gap_to_max']=frame.groupby('query_id')[col].transform('max')-s
    assert np.isfinite(frame[['raw_multivers_rationale','single_rationale']].to_numpy()).all()
    frame.to_parquet(cache,index=False)
    return frame


def with_labels(part,queries):
    available=set(part.query_id)
    queries={q:v for q,v in queries.items() if q in available}
    p=part[part.query_id.isin(queries)].copy().reset_index(drop=True)
    p['gold_evidence']=[int(cid in queries[q]['gold_ids']) for q,cid in zip(p.query_id,p.candidate_id)]
    p['family']=p.query_id.map(lambda q:queries[q]['source_family'])
    assert set(p.query_id)==set(queries)
    return p


def smoke():
    out=OUT/'smoke'
    out.mkdir(parents=True,exist_ok=True)
    frame=adaptive.prepare()
    keep=screens(frame)
    part=frame[keep].reset_index(drop=True)
    fold=adaptive.folds(part)
    train,valid=part[fold!=0].reset_index(drop=True),part[fold==0].reset_index(drop=True)
    for name in ['boost:context:2:0:0','boost:full_context:2:0:0','mean:0.5:1']:
        model=calibrated.fit(train,name)
        values=calibrated.score(model,valid)
        modified=valid.copy()
        modified['gold_evidence']=1-modified.gold_evidence
        assert np.allclose(calibrated.score(model,modified.iloc[::-1])[::-1],values,rtol=0,atol=1e-12)
        for row in adaptive.quality(valid,values):
            assert np.isfinite(row['precision'])
    record={'passed':True,'code_sha256':digest(Path(__file__)),'plan_sha256':digest(PLAN),
        'model_family_sha256':digest(Path(calibrated.__file__)),'external_test_outcomes_accessed':False,
        'family_names':calibrated.candidates()}
    (OUT/'study_specification.json').write_text(json.dumps(record,indent=2))
    (out/'audit.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


def train():
    start=time.monotonic()
    spec=json.loads((OUT/'study_specification.json').read_text())
    assert spec['passed'] and spec['code_sha256']==digest(Path(__file__)) and spec['plan_sha256']==digest(PLAN)
    assert not (OUT/'test_started.json').exists()
    frame=probability_frame()
    names=calibrated.candidates()
    for name in ['healthver','covidfact']:
        out=OUT/name
        out.mkdir(parents=True,exist_ok=True)
        if (out/'freeze.json').exists():
            print(name,'already frozen; preserved',flush=True)
            continue
        queries=label_data(name,'train')
        if name=='healthver':
            calibration=label_data(name,'dev')
        else:
            calibration={q:v for q,v in queries.items() if int(hashlib.sha256(f"84:cal:{v['source_family']}".encode()).hexdigest(),16)%5==0}
            queries={q:v for q,v in queries.items() if q not in calibration}
        # Deduplicate using public claim text. This does not open test outcomes.
        normalize=lambda s:' '.join(re.findall(r'\w+',s.lower()))
        test_text={normalize(s) for s in frame.loc[(frame.dataset==name)&(frame.split=='test'),'query_text']}
        original_fitting,original_calibration=len(queries),len(calibration)
        calibration={q:v for q,v in calibration.items() if normalize(v['query_text']) not in test_text}
        reserved_text=test_text|{normalize(v['query_text']) for v in calibration.values()}
        queries={q:v for q,v in queries.items() if normalize(v['query_text']) not in reserved_text}
        (out/'text_exclusion_audit.json').write_text(json.dumps({
            'fitting_before':original_fitting,'fitting_after':len(queries),
            'calibration_before':original_calibration,'calibration_after':len(calibration),
            'test_outcomes_accessed':False,'rule':'Exact equality after lowercase word normalization.'},indent=2))
        fitting_all=with_labels(frame,queries)
        fitting=fitting_all[screens(fitting_all)].reset_index(drop=True)
        cal_all=with_labels(frame,calibration)
        cal=cal_all[screens(cal_all)].reset_index(drop=True)
        folds=adaptive.folds(fitting)
        rows=[]
        for k in range(4):
            a,b=fitting[folds!=k].reset_index(drop=True),fitting[folds==k].reset_index(drop=True)
            assert not set(a.family)&set(b.family)
            for method in names:
                model=calibrated.fit(a,method)
                score=calibrated.score(model,b)
                q=adaptive.quality(b,score)[0]
                rows.append({'fold':k,'method':method,**q})
            print(name,'train fold',k,'elapsed',round(time.monotonic()-start,1),'s',flush=True)
        cv=pd.DataFrame(rows)
        cv.to_csv(out/'train_cv.csv',index=False)
        means=cv.groupby('method').precision.mean()
        means.sort_values(ascending=False).to_csv(out/'train_cv_mean.csv')
        key=lambda n:(round(means[n],12),-names.index(n))
        contextual=[n for n in names if n.startswith('boost:context:')]
        singles=[n for n in names if n in ['full_raw','full_product','single_raw'] or ':full:' in n or ':full_context:' in n]
        blends=[n for n in names if n.startswith('mean:')]
        chosen,control,blend=max(contextual,key=key),max(singles,key=key),max(blends,key=key)
        matching=chosen.replace(':context:',':full_context:')
        no_context=chosen.replace(':context:',':dual:')
        comparisons={'selected':chosen,'best_single':control,'best_blend':blend,'full_raw':'full_raw',
            'full_product':'full_product','single_raw':'single_raw','matched_full':matching,
            'without_context':no_context,'without_focus':chosen}
        freeze={'dataset':name,'chosen':chosen,'single_control':control,'arithmetic_control':blend,
            'fitting_ids':sorted(queries),'calibration_ids':sorted(calibration),'models':{},
            'code_sha256':digest(Path(__file__)),'specification_sha256':digest(OUT/'study_specification.json'),
            'test_outcomes_accessed':False,'frozen_at_utc':pd.Timestamp.now(tz='UTC').isoformat()}
        for label,method in comparisons.items():
            model=calibrated.fit(fitting_all if label=='without_focus' else fitting,method)
            path=out/f'{label}.joblib'
            joblib.dump(model,path)
            rule=admission.fit_threshold(cal,calibrated.score(model,cal),100)
            freeze['models'][label]={'name':method,'path':path.name,'sha256':digest(path),'prospective_rule':rule}
        (out/'freeze.json').write_text(json.dumps(freeze,indent=2))
        print(name,'FROZEN',chosen,'controls',control,blend,flush=True)
    print('All external selections and thresholds frozen without test outcomes.',flush=True)


def evaluate():
    start=time.monotonic()
    assert not (OUT/'test_started.json').exists(),'External confirmation has already started.'
    freezes={n:json.loads((OUT/n/'freeze.json').read_text()) for n in ['healthver','covidfact']}
    for f in freezes.values():
        assert f['code_sha256']==digest(Path(__file__)) and not f['test_outcomes_accessed']
    (OUT/'test_started.json').write_text(json.dumps({'time_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'code_sha256':digest(Path(__file__)),'freeze_sha256':{n:digest(OUT/n/'freeze.json') for n in freezes}},indent=2))
    frame=probability_frame()
    summaries,intervals=[],[]
    for name,freeze in freezes.items():
        queries=label_data(name,'test')
        all_candidates=with_labels(frame,queries)
        part=all_candidates[screens(all_candidates)].reset_index(drop=True)
        families=sorted({q['source_family'] for q in queries.values()})
        lookup={f:i for i,f in enumerate(families)}
        rng=np.random.default_rng(2026101584)
        draws=rng.multinomial(len(families),np.full(len(families),1/len(families)),size=2000)
        boot,point={},{}
        for label,info in freeze['models'].items():
            path=OUT/name/info['path']
            assert digest(path)==info['sha256']
            model=joblib.load(path)
            values=calibrated.score(model,part)
            assert np.isfinite(values).all()
            modified=part.copy()
            modified['gold_evidence']=1-modified.gold_evidence
            assert np.allclose(calibrated.score(model,modified.iloc[::-1])[::-1],values,rtol=0,atol=1e-12)
            part[['query_id','candidate_id']].assign(score=values).to_parquet(OUT/name/f'scores_{label}.parquet',index=False)
            for analysis,rule in [('exact_budget',budget_rule(part,values,len(queries))),('prospective',info['prospective_rule'])]:
                stats=query_statistics(part,values,rule,np.zeros(len(part),bool),queries)
                if analysis=='exact_budget':
                    assert stats.admitted.sum()==len(queries)
                    boot[label]=reselected_precision(part,values,queries,draws,lookup)
                    point[label]=stats.correct.sum()/stats.admitted.sum()
                    # Independently verify exact budgets against an explicit
                    # duplicated-source ordering for three bootstrap samples.
                    rank=admission.ranks(part,values)
                    for draw in draws[:3]:
                        eligible=np.flatnonzero(rank<=5)
                        repeated=np.repeat(eligible,[draw[lookup[queries[q]['source_family']]] for q in part.query_id.iloc[eligible]])
                        budget=sum(draw[lookup[q['source_family']]] for q in queries.values())
                        order=np.lexsort((part.tie_key.to_numpy(str)[repeated],-values[repeated]))
                        direct=part.gold_evidence.to_numpy(int)[repeated[order[:budget]]].mean()
                        fast=reselected_precision(part,values,queries,draw[None,:],lookup)[0]
                        assert np.isclose(direct,fast)
                stats.to_csv(OUT/name/f'query_{label}_{analysis}.csv',index=False)
                summaries.append({'dataset':name,'method':label,'selected_form':info['name'],'analysis':analysis,
                    'source_groups':len(families),**admission.summarize(stats)})
        for reference in ['best_single','best_blend','full_raw','full_product','single_raw','matched_full','without_context','without_focus']:
            delta=boot['selected']-boot[reference]
            lo,hi=np.quantile(delta,[.025,.975])
            slo,shi=np.quantile(delta,[.00625,.99375])
            intervals.append({'dataset':name,'reference':reference,'delta_precision':point['selected']-point[reference],
                'ci95_low':lo,'ci95_high':hi,'simultaneous_low':slo,'simultaneous_high':shi,
                'primary':reference in ['best_single','best_blend'],'source_groups':len(families)})
        print(name,'test evaluated once',flush=True)
    summary=pd.DataFrame(summaries)
    comparison=pd.DataFrame(intervals)
    summary.to_csv(OUT/'summary.csv',index=False)
    comparison.to_csv(OUT/'paired_intervals.csv',index=False)
    primary=comparison[comparison.primary]
    assessment={'supported_general_increment':bool((primary.simultaneous_low>0).all()),
        'all_primary_point_gains_positive':bool((primary.delta_precision>0).all()),
        'primary_comparisons':primary.to_dict(orient='records'),
        'one_shot_test':True,'training_thresholds_unchanged_after_test':True,
        'source_sha256':digest(Path(__file__)),'elapsed_seconds':time.monotonic()-start}
    (OUT/'assessment.json').write_text(json.dumps(assessment,indent=2))
    print(summary[summary.analysis=='exact_budget'][['dataset','method','evidence_precision','coverage','evidence_recall']].to_string(index=False),flush=True)
    print(comparison.to_string(index=False),flush=True)
    print(json.dumps(assessment),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['smoke','train','evaluate'])
    with threadpool_limits(limits=2):
        a=p.parse_args()
        {'smoke':smoke,'train':train,'evaluate':evaluate}[a.stage]()
