"""Hand-built verification of prefix allocation without confirmation labels."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import evidence_set_confirmation as method
import r5_simple_controls as simple
import independent_confirmation_evaluate as evaluation

R=Path(__file__).resolve().parents[1]
O=R/'results/evidence_set_confirmation/static_verification'


class FixedDistribution:
    classes_=np.arange(6)
    def predict_proba(self,x):
        return np.asarray([[.85,0,.15,0,0,0],[1,0,0,0,0,0]])


class FixedCalibration:
    def predict_proba(self,x):
        p=np.asarray([.1,.1,0,0,0,.3,0,0,0,0])
        return np.column_stack([1-p,p])


def run():
    O.mkdir(exist_ok=True,parents=True)
    data={'qids':np.asarray(['A','B']), 'codes':np.repeat(np.arange(2),5),
        'ranks':np.tile(np.arange(1,6),2), 'cids':np.asarray([f'{q}{i}' for q in ['A','B'] for i in range(5)]),
        'ties':np.asarray([f'{i:016x}' for i in range(10)]),
        'raw':np.tile(np.asarray([.9,.8,.7,.6,.5]),2),'y':np.zeros(10,int),
        'queries':{'A':{'gold_count':0,'gold_sets':[]},'B':{'gold_count':0,'gold_sets':[]}}}
    model={'model':FixedDistribution(),'calibrator':FixedCalibration()}
    score=method.scores(model,data,np.zeros((2,1)),4)
    np.testing.assert_allclose(score.reshape(2,5),[[.4,.4,0,0,0],[.3,0,0,0,0]],atol=1e-12)
    raw=np.asarray([[0,.1,.8,.8,.8,.8],[0,.3,.3,.3,.3,.3]])
    hull=np.asarray([[0,.4,.8,.8,.8,.8],[0,.3,.3,.3,.3,.3]])
    checks=[]
    for budget in range(11):
        mask=simple.admitted(data,score,simple.threshold(data,score,50*budget))
        count=np.bincount(data['codes'],weights=mask,minlength=2).astype(int)
        assert mask.sum()==budget
        for qid in range(2):
            np.testing.assert_array_equal(mask[data['codes']==qid],np.arange(1,6)<=count[qid])
        feasible=[pair for pair in itertools.product(range(6),repeat=2) if sum(pair)==budget]
        best_hull=max(sum(hull[q,k] for q,k in enumerate(pair)) for pair in feasible)
        best_raw=max(sum(raw[q,k] for q,k in enumerate(pair)) for pair in feasible)
        actual_hull=sum(hull[q,k] for q,k in enumerate(count))
        actual_raw=sum(raw[q,k] for q,k in enumerate(count))
        assert abs(best_hull-actual_hull)<1e-12
        checks.append({'budget':budget,'prefix_lengths':count.tolist(),'concave_envelope_value':actual_hull,
            'maximum_concave_envelope_value':best_hull,'original_utility_value':actual_raw,
            'maximum_original_utility_value':best_raw})
    assert checks[1]['original_utility_value']==.1 and checks[1]['maximum_original_utility_value']==.3
    # Scramble label insertion order and provide a stale dummy-group cache.
    truth={('A','A0'):1,('A','A1'):1,('B','B0'):1}
    labels={(q,c):truth.get((q,c),0) for q,c in reversed(list(zip(data['qids'][data['codes']],data['cids'])))}
    queries={'A':{'gold_count':2,'gold_sets':[['A0','A1']]},'B':{'gold_count':1,'gold_sets':[['B0']]}}
    data['_groups']=('stale dummy cache',None)
    attached=evaluation.attach(data,labels,queries)
    assert '_groups' not in attached
    np.testing.assert_array_equal(attached['y'],[1,1,0,0,0,1,0,0,0,0])
    for budget,expected_correct,expected_complete in [(1,[1,0],[0,0]),(2,[2,0],[1,0]),(3,[2,1],[1,1])]:
        mask=simple.admitted(attached,score,simple.threshold(attached,score,50*budget))
        stat=simple.statistics(attached,mask)
        np.testing.assert_array_equal(stat.correct,expected_correct)
        np.testing.assert_array_equal(stat.all_facts,expected_complete)
    record={'passed':True,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'actual_confirmation_labels_or_results_read':False,'hand_built_queries':2,'sentences_per_query':5,
        'budgets_checked':list(range(11)),'prefix_and_exact_budget_verified':True,
        'optimal_for_concave_envelope_verified_by_exhaustive_enumeration':True,
        'optimal_for_original_nonconcave_utility':False,
        'original_utility_counterexample_budget':1,'original_utility_gap':.2,
        'true_label_lookup_and_stale_group_cache_removal_verified':True,'checks':checks}
    (O/'prefix_allocation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':run()
