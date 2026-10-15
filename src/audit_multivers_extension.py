"""Independent integrity checks for the new scientific comparisons."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import r2_admission as admission
from r2_statistics import query_statistics
from run_multivers_extension import extra_view, SPEC

R2=Path(__file__).resolve().parents[1]
OUT=R2/'results/multivers_extension'


def main(context=False):
    root=OUT/'k100'
    summary=pd.read_csv(root/'summary.csv')
    forms=pd.read_csv(root/'selected_forms.csv')
    assert set(forms.seed)==set(SPEC['seeds'])
    assert set(summary.method)==set(SPEC['methods'])
    exact=summary[summary.analysis=='exact_budget']
    assert (exact.admitted_total==210).all() and (exact.volume_per_100==100).all()
    old=pd.read_csv(R2/'results/strengthening/multivers/primary.csv').set_index('seed')
    new=summary[(summary.method=='multivers')&(summary.analysis=='prospective_100')].set_index('seed')
    assert np.allclose(old.evidence_precision.sort_index(),new.evidence_precision.sort_index(),atol=1e-12)
    frame=pd.read_parquet(R2/'data/scored/scifact_full_k100.parquet').merge(
        pd.read_parquet(R2/'results/strengthening/multivers/candidate_scores.parquet'),
        on=['query_id','candidate_id'],validate='one_to_one',sort=False)
    model=json.loads((root/'model_seed13.json').read_text())
    test=frame[frame.query_id.isin(model['evaluation_query_ids'])].reset_index(drop=True)
    queries=json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    qtest={q:queries[q] for q in model['evaluation_query_ids']}
    with extra_view():
        scores,stress,_=admission.score_pipeline(test,model)
        changed=test.copy()
        changed['gold_evidence']=1-changed.gold_evidence
        changed['gold_directional']=1-changed.gold_directional
        changed['claim_verdict']='PERMUTED'
        mutated=admission.score_pipeline(changed,model)[0]['scet']
        assert np.array_equal(scores['scet'],mutated)
        order=np.arange(len(test))[::-1]
        reversed_scores=admission.score_pipeline(test.iloc[order],model)[0]['scet']
        assert np.array_equal(scores['scet'],reversed_scores[order])
    from strengthen_r2_analysis import budget_rule
    rule=budget_rule(test,scores['scet'],210)
    fast=query_statistics(test,scores['scet'],rule,stress,qtest)
    direct=admission.query_statistics(test,scores['scet'],rule,stress,qtest)
    for col in ['admitted','correct','covered','all_facts','has_error','gold_count']:
        assert np.array_equal(fast[col],direct[col]),col
    # Recompute displayed differences directly from the frozen query outcomes.
    deltas=[]
    for seed in SPEC['seeds']:
        a=pd.read_csv(root/f'query_seed{seed}_scet_multivers_exact_budget.csv')
        b=pd.read_csv(root/f'query_seed{seed}_multivers_exact_budget.csv')
        deltas.append(a.correct.sum()/a.admitted.sum()-b.correct.sum()/b.admitted.sum())
    interval=pd.read_csv(root/'paired_intervals.csv')
    value=interval[(interval.analysis=='exact_budget')&(interval.reference=='multivers')&(interval.metric=='evidence_precision')].iloc[0]
    assert np.isclose(np.mean(deltas),value.difference,atol=1e-12)
    for p in root.glob('risk_seed*.json'):
        config=json.loads(p.read_text())
        passing=[x for x in config['tests'] if x['passes'] and x['admitted']]
        assert bool(passing)==config['nonempty_feasible']
        assert all(x['upper_bound']<=config['alpha']+1e-10 for x in passing)
    report={'four_partitions_complete':True,'same_candidates':True,'previous_multivers_reproduced':True,
            'exact_budget_verified':True,'test_outcome_invariance':True,'candidate_order_invariance':True,
            'independent_query_metrics_match':True,'risk_test_and_upper_bound_agree':True,
            'combination_precision_gain_over_multivers':float(value.difference),
            'protocol_sha256':hashlib.sha256((R2/'data/multivers_extension_protocol.json').read_bytes()).hexdigest()}
    if context:
        cf=OUT/'context/full'
        a=json.loads((cf/'audit.json').read_text())
        infer=json.loads((OUT/'inference_single/audit.json').read_text())
        assert a['full_context_reproduces_previous_result'] and infer['pairs']==30000
        c=pd.read_csv(cf/'summary.csv')
        assert len(c)==16 and set(c.seed)==set(SPEC['seeds'])
        assert (c[c.analysis=='exact_budget'].admitted_total==210).all()
        single=pd.read_parquet(OUT/'inference_single/candidate_scores.parquet').set_index(['query_id','candidate_id'])
        raw=[json.loads(line) for line in (OUT/'inference_single/scores.jsonl').read_text().splitlines()]
        assert len(raw)==30000
        for row in raw:
            assert np.isclose(sum(row['label_probabilities']),1,atol=.002)
            expected=row['rationale_probability']*(1-row['label_probabilities'][1])
            observed=single.loc[(row['query_id'],row['candidate_id']),'raw_multivers']
            assert np.isclose(observed,expected,atol=1e-7)
        smoke=pd.read_parquet(OUT/'inference_single/candidate_scores_smoke.parquet').set_index(['query_id','candidate_id'])
        drift=np.abs(single.reindex(smoke.index).raw_multivers-smoke.raw_multivers)
        assert drift.notna().all() and drift.max()<.002
        full_manifest=json.loads((R2/'data/models/multivers/inference_manifest.json').read_text())
        single_manifest=json.loads((OUT/'inference_single/model_manifest.json').read_text())
        for key in ['sha256','encoder_revision','tokenizer_files','official_forward_sha256']:
            assert full_manifest[key]==single_manifest[key],key
        single_manifest['batching']='Length ordering; 8192 padded tokens for initial cached predictions and 16384 after a successful 64-pair smoke test; halve batch size after memory pressure.'
        (OUT/'inference_single/model_manifest.json').write_text(json.dumps(single_manifest,indent=2),encoding='utf-8')
        report.update(context_comparison_complete=True,context_pair_count=30000,same_checkpoint_verified=True)
        report.update(raw_probability_score_construction_verified=True,smoke_prediction_max_difference=float(drift.max()))
    (OUT/('final_audit.json' if context else 'combination_audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',action='store_true');a=p.parse_args();main(a.context)
