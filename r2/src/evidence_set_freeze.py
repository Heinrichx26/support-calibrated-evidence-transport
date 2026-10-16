"""Freeze fitting-selected prefix utility models for one new-data confirmation."""
from pathlib import Path
import hashlib
import json
import pickle
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import r5_context_gain as dev
import evidence_set_balanced_confirmation as balanced

base = balanced.base
O = base.O


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_without_labels(bundle, raw_frame):
    """Return screened top-five prefix data and utilities using identifiers/scores."""
    frame = raw_frame.copy()
    frame['gold_evidence'] = 0
    x, keep = dev.list_inputs(frame)
    frame, x = frame[keep].reset_index(drop=True), x[keep]
    dummy = {q: {'gold_sets': [], 'gold_count': 0} for q in sorted(frame.query_id.unique())}
    data, features, unused = base.prepared(frame, x, bundle['base'], dummy)
    utility = base.scores(bundle['utility'], data, features, bundle['beta'])
    return data, utility


def run():
    started = time.monotonic()
    gate = json.loads((O/'development/audit.json').read_text(encoding='utf-8'))
    assert gate['technical_passed'] and gate['direction_passed']
    assert gate['protocol_sha256']==sha(O/'protocol.json')
    target = O/'frozen'
    target.mkdir(exist_ok=True)
    assert not (target/'freeze.json').exists()
    cv = pd.read_csv(O/'development/cv.csv')
    means = cv.groupby(['group','beta']).mean(numeric_only=True)
    threshold = float(means.loc[('ce',0.0),'evidence_precision']-.01)
    selected = {}
    for group in ['ce','all']:
        pool = means.loc[group].reset_index()
        feasible = pool[pool.evidence_precision>=threshold]
        assert len(feasible)
        chosen = feasible.sort_values(['all_facts_recall','multiple_set_recall','evidence_precision','beta'],
            ascending=[False,False,False,True],kind='stable').iloc[0]
        selected[group] = {'beta':float(chosen.beta), 'fitting_only_cv':chosen.to_dict()}
    plan = {'fitting_protocol_sha256':sha(O/'protocol.json'),'development_audit_sha256':sha(O/'development/audit.json'),
        'base_model_source_sha256':sha(dev.__file__),'utility_source_sha256':sha(base.__file__),
        'balanced_source_sha256':sha(balanced.__file__),'freeze_source_sha256':sha(__file__),
        'selection': 'Both feature groups use the same predeclared precision feasibility floor and complete-set/multiple-set/precision ordering. Selection uses only fitting out-of-fold outcomes.',
        'selected':selected, 'comparisons':['all_selected_minus_ce_selected','all_selected_minus_all_beta_zero','ce_selected_minus_ce_beta_zero'],
        'new_confirmation_outcomes_read':False}
    base.dump(target/'before_fitting.json',plan)
    queries = json.loads((dev.R/'data/processed/fever_full_queries.json').read_text(encoding='utf-8'))
    models = []
    for seed in base.PLAN['seeds']:
        frame, original = dev.load(seed)
        x, keep = dev.list_inputs(frame)
        frame, x = frame[keep].reset_index(drop=True), x[keep]
        for group in ['ce','all']:
            bp = dev.O/f'development/model_seed{seed}_{group}.pkl'
            with bp.open('rb') as h:
                bm = pickle.load(h)
            data, features, y = base.prepared(frame,x,bm,queries)
            um = balanced.fit(data,features,y)
            bundle = {'base':bm,'utility':um,'beta':selected[group]['beta'],'seed':seed,'group':group}
            dest = target/f'model_seed{seed}_{group}.pkl'
            with dest.open('wb') as h:
                pickle.dump(bundle,h)
            # Exact predictions must ignore supplied labels and row order.
            small = frame[frame.query_id.isin(data['qids'][:20])].copy()
            d1,p1 = predict_without_labels(bundle,small)
            small.gold_evidence = 1-small.gold_evidence
            d2,p2 = predict_without_labels(bundle,small.iloc[::-1].reset_index(drop=True))
            assert np.array_equal(d1['qids'],d2['qids']) and np.array_equal(d1['cids'],d2['cids'])
            np.testing.assert_array_equal(p1,p2)
            models.append({'seed':seed,'group':group,'path':dest.name,'sha256':sha(dest),
                'base_model_sha256':sha(bp),'beta':selected[group]['beta'],'fitting_queries':len(data['qids']),
                'label_order_invariance':True})
            print(seed,group,'frozen',time.monotonic()-started,flush=True)
    base.dump(target/'freeze.json',{**plan,'models':models,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
