"""Frozen context-model score caches and predeclared operating-point controls.

No models are fitted or selected here. First run --smoke, then --full. Every
new artifact stays in results/r5_context_gain/controls.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import r5_context_gain as dev
import r5_simple_controls as simple

R = dev.R
O = dev.O/'controls'
PLAN = {
    'created': '2026-10-16 before inspection of new context held-out results',
    'primary': 'Frozen all-view list-feature model minus frozen equal-capacity CE-only list-feature model.',
    'exploratory': 'All-view comparisons with existing single-scorer raw cap-five/top-one, logistic list cap-five/top-one and original frozen SCET. All outcomes retained.',
    'models': 'Use every seed13/17/29/7 ce/all model and split ID from the saved development freeze; no refitting, feature or model selection.',
    'screen': dev.PLAN['screen'],
    'score_cache': 'Save original identifiers, fixed tie keys, evidence labels, both frozen model scores and split membership for every screened fitting/calibration/test candidate.',
    'features': dev.PLAN['features'],
    'workload_targets': [25,50,100,200,300,400,500],
    'workload_threshold': 'Independent original calibration queries choose score/tie order statistic; evaluate frozen threshold on original held-out queries, max five outputs/query.',
    'fixed_batch': '100 sentences per 100 held-out queries, label-free score/tie order, max five/query; separate diagnostic.',
    'risk_targets': [0.01,0.025,0.05,0.1,0.2,0.3],
    'risk_metrics': ['query_error_rate','false_admission_no_evidence'],
    'risk_grid': '101 quantiles of each fixed model fitting-sample eligible top-five predictions plus +/-infinity, frozen before calibration labels are used.',
    'risk_selection': 'Calibration alone chooses maximum output count satisfying the empirical error target; no finite-sample guarantee is claimed.',
    'outcomes': ['evidence_precision','volume_per_100','annotated_per_100','coverage','query_error_rate','conditional_query_error','false_admission_no_evidence','evidence_recall','all_facts_recall','singleton_set_recall','multiple_set_recall'],
    'interval_points': ['exact_budget_100','query_error_rate_0.1','false_admission_no_evidence_0.1'],
    'intervals': '1000 paired query-cluster bootstrap draws, same query weight across overlapping partitions, frozen decisions; pointwise 95% intervals, no family-wide simultaneous claim.',
    'primary_interval_metrics': ['evidence_precision','all_facts_recall','coverage','query_error_rate','conditional_query_error','false_admission_no_evidence','singleton_set_recall','multiple_set_recall'],
    'exploratory_interval_metrics': ['evidence_precision','all_facts_recall','coverage','conditional_query_error','false_admission_no_evidence'],
    'threads': 2,
    'smoke': 'Seed13 first200 fitting/100 calibration/200 held-out queries; check scores, split disjointness, original identifiers, cap, complete-set invariants and source-feature label/order invariance. Full gated on structural validity only, with unchanged code and model hashes.',
    'scope': 'Exploratory extension on previously studied official development collections; empirical calibration constraints and observed held-out risks are reported separately.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metadata(models, freezes):
    return {'training_source_sha256':sha(dev.__file__),
            'freeze_sha256':sha(dev.O/'development/freeze.json'),
            'models':{f'{s}_{g}':sha(dev.O/f'development/model_seed{s}_{g}.pkl') for s in freezes for g in ['ce','all']},
            'scored_data_sha256':sha(R/'data/scored/fever_full_k100.parquet'),
            'simple_control_source_sha256':sha(simple.__file__)}


def score_cache(ids, split, models, out, smoke):
    frame = pd.read_parquet(R/'data/scored/fever_full_k100.parquet', columns=dev.COLS,
                            filters=[('query_id','in',ids)])
    assert set(frame.query_id)==set(ids)
    result = []
    for start in range(0,len(ids),1000):
        part = frame[frame.query_id.isin(ids[start:start+1000])].copy().reset_index(drop=True)
        x, keep = dev.list_inputs(part)
        if smoke and start == 0:
            altered = part.iloc[::-1].reset_index(drop=True).copy()
            altered.gold_evidence = 1-altered.gold_evidence
            rx, rk = dev.list_inputs(altered)
            np.testing.assert_array_equal(keep,rk[::-1])
            np.testing.assert_allclose(x,rx[::-1],rtol=0,atol=0)
        cache = part.loc[keep,['query_id','candidate_id','tie_key','gold_evidence']].copy()
        for name, model in models.items():
            values = dev.predict(model,x[keep])
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(values,dev.predict(model,x[keep][::-1])[::-1])
            cache['score_'+name] = values
        cache['split'] = split
        result.append(cache)
        del part,x,cache
    data = pd.concat(result,ignore_index=True)
    assert not data.duplicated(['query_id','candidate_id']).any()
    data.to_parquet(out/f'candidate_scores_{split}.parquet',index=False)
    del frame,result
    gc.collect()
    return data


def summary(stats):
    result = simple.summary(stats)
    result['conditional_query_error'] = stats.has_error.sum()/stats.covered.sum() if stats.covered.any() else np.nan
    return result


def evaluation(f,c,t,method,out):
    rows, saved, rules = [],{},[]
    fs,cs,ts = [x['raw'] for x in [f,c,t]]
    for target in PLAN['workload_targets']:
        rule = simple.threshold(c,cs,target)
        stat = simple.statistics(t,simple.admitted(t,ts,rule))
        rows.append({'method':method,'analysis':f'workload_{target}',**summary(stat)})
        stat.to_parquet(out/f'query_{method}_workload_{target}.parquet',index=False)
        rules.append({'method':method,'analysis':f'workload_{target}',**rule})
    rule = simple.threshold(t,ts,100)
    stat = simple.statistics(t,simple.admitted(t,ts,rule))
    assert stat.admitted.sum()==len(t['qids'])
    saved[(method,'exact_budget_100')] = stat
    rows.append({'method':method,'analysis':'exact_budget_100',**summary(stat)})
    stat.to_parquet(out/f'query_{method}_exact_budget_100.parquet',index=False)
    rules.append({'method':method,'analysis':'exact_budget_100',**rule})
    grid, choices = simple.risk_choices(f,fs,c,cs)
    pd.DataFrame(grid).to_csv(out/f'calibration_risk_grid_{method}.csv',index=False)
    for analysis, entry in choices:
        rule = {'threshold':entry['threshold'],'tie_cutoff':'ffffffffffffffff'}
        stat = simple.statistics(t,simple.admitted(t,ts,rule))
        rows.append({'method':method,'analysis':analysis,**summary(stat)})
        stat.to_parquet(out/f'query_{method}_{analysis}.parquet',index=False)
        rules.append({'method':method,'analysis':analysis,**entry})
        if analysis in PLAN['interval_points']:
            saved[(method,analysis)] = stat
    for stat in saved.values():
        assert stat.admitted.max()<=5 and stat.correct.le(stat.admitted).all()
        assert stat.all_facts.le(stat.gold_count.gt(0)).all()
    pd.DataFrame(rules).to_csv(out/f'rules_{method}.csv',index=False)
    return rows,saved


def parts(s,name):
    if name=='conditional_query_error':
        return s.has_error.to_numpy(float),s.covered.to_numpy(float)
    if name in ['singleton_set_recall','multiple_set_recall']:
        keep = s.minimum_set_size.eq(1).to_numpy() if name.startswith('singleton') else s.minimum_set_size.gt(1).to_numpy()
        return s.all_facts.to_numpy(float)*keep,keep.astype(float)
    return simple.metric_parts(s,name)


def bootstrap(collection,out):
    ids = sorted(set().union(*(set(x[('context_all','exact_budget_100')].query_id) for x in collection.values())))
    lookup = {q:i for i,q in enumerate(ids)}
    rng = np.random.default_rng(2026101619)
    draws = rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=1000).astype(np.float32)
    records = []
    refs = ['context_ce','simple_raw_cap5','simple_raw_top1','simple_list_logit_cap5','simple_list_logit_top1','original_scet']
    for reference in refs:
        metrics = PLAN['primary_interval_metrics'] if reference=='context_ce' else PLAN['exploratory_interval_metrics']
        for analysis in PLAN['interval_points']:
            for name in metrics:
                samples, points = [],[]
                included = []
                for seed, rows in collection.items():
                    a = rows[('context_all',analysis)].set_index('query_id')
                    b = rows[(reference,analysis)].set_index('query_id').reindex(a.index)
                    an,ad = parts(a,name);bn,bd = parts(b,name)
                    if ad.sum()==0 or bd.sum()==0:
                        continue
                    w = draws[:,[lookup[q] for q in a.index]]
                    aa,bb = w@np.column_stack([an,ad]),w@np.column_stack([bn,bd])
                    av = np.divide(aa[:,0],aa[:,1],out=np.full(len(w),np.nan),where=aa[:,1]>0)
                    bv = np.divide(bb[:,0],bb[:,1],out=np.full(len(w),np.nan),where=bb[:,1]>0)
                    samples.append(av-bv)
                    points.append(an.sum()/ad.sum()-bn.sum()/bd.sum())
                    included.append(seed)
                if samples:
                    distribution = np.nanmean(samples,axis=0)
                    low,high = np.nanquantile(distribution,[.025,.975])
                    records.append({'method':'context_all','reference':reference,'analysis':analysis,'metric':name,
                        'difference':np.mean(points),'ci_low':low,'ci_high':high,
                        'comparison_status':'primary' if reference=='context_ce' else 'exploratory',
                        'included_partitions':json.dumps(included),'bootstrap_repetitions':1000,
                        'scope':'pointwise; frozen models and decisions; shared query weights across observed partitions'})
    pd.DataFrame(records).to_csv(out/'paired_intervals.csv',index=False)


def report(data,intervals,out):
    means = data.groupby(['method','analysis'])[PLAN['outcomes']].mean()
    lines = ['# Frozen context-form controls','',
        'Primary comparison: all-view list features versus the same-capacity CE-only model, selected using fitting queries before this evaluation. Other comparisons with existing simple controls and original SCET are exploratory. The complete predeclared workload and empirical calibration-risk grids are retained.', '',
        '| Policy | Precision | Items/100 | Coverage | Query error | Conditional error | No-evidence admission | Set recall | Singleton | Multiple |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for analysis in PLAN['interval_points']:
        for method in ['context_all','context_ce','simple_list_logit_cap5','simple_list_logit_top1','original_scet']:
            row = means.loc[(method,analysis)]
            fields = ['evidence_precision','volume_per_100','coverage','query_error_rate','conditional_query_error','false_admission_no_evidence','all_facts_recall','singleton_set_recall','multiple_set_recall']
            lines.append('| '+method+' / '+analysis+' | '+' | '.join(f'{row[f]:.4f}' for f in fields)+' |')
    lines += ['', 'Primary paired differences are all-view minus capacity-matched CE:', '',
              '| Analysis | Metric | Difference | Pointwise 95% interval |',
              '|---|---|---:|---|']
    for row in intervals[intervals.comparison_status=='primary'].itertuples():
        lines.append(f'| {row.analysis} | {row.metric} | {row.difference:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] |')
    lines += ['',
        'All thresholds for prospective workload and empirical risk constraints come from the independent calibration queries; fitting predictions fix the risk grids. Fixed-batch allocation uses scores and tie keys without held-out labels. Calibration empirical error targets need not equal held-out error. Conditional query error divides error queries by covered queries and is reported separately from the unconditional event.',
        '',
        'The score cache preserves each screened fitting/calibration/test candidate identity, fixed tie key and both frozen-model scores. Candidate-list summaries and the common twelve-view top-five union use prediction values only. CE-only uses this shared screen, so it describes a controlled scorer comparison within that screen. No new model or point is selected from these outputs.',
        '',
        'Intervals use 1,000 query-cluster bootstrap draws shared across overlapping partitions, conditional on fixed models and observed splits. They are pointwise intervals. The extra forms remain an exploratory extension evaluated on a previously studied official development collection. No family-wide simultaneous or finite-sample risk-control claim is made for this analysis.',
        '',
        'Files: summary.csv, mean_sd.csv, paired_intervals.csv; seed*/candidate_scores_{fitting,calibration,evaluation}.parquet; seed*/rules_*.csv; seed*/calibration_risk_grid_*.csv; seed*/query_*.parquet; analysis_plan.json and audit.json.',
    ]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(smoke):
    started = time.monotonic()
    O.mkdir(parents=True,exist_ok=True)
    plan = O/'analysis_plan.json'
    if plan.exists():
        assert load(plan)==PLAN
    else:
        dump(plan,PLAN)
    freezes = load(dev.O/'development/freeze.json')
    assert len(freezes)==4
    identities = metadata(None,[f['seed'] for f in freezes])
    out = O/('smoke' if smoke else 'full')
    out.mkdir(exist_ok=True)
    if not smoke:
        gate = load(O/'smoke/audit.json')
        assert gate['passed'] and gate['code_sha256']==sha(__file__) and gate['inputs']==identities
    dump(out/'before_evaluation.json',{'plan_sha256':sha(plan),'inputs':identities,
         'code_sha256':sha(__file__),'status':'frozen before this stage reads candidate outcome labels'})
    queries = load(R/'data/processed/fever_full_queries.json')
    summaries,collection = [],{}
    for freeze in freezes[:1] if smoke else freezes:
        seed = freeze['seed']
        fitids,calids,testids = [sorted(freeze[k]) for k in ['fitting_query_ids','threshold_query_ids','evaluation_query_ids']]
        assert not(set(fitids)&set(calids) or set(fitids)&set(testids) or set(calids)&set(testids))
        if smoke:
            fitids,calids,testids = fitids[:200],calids[:100],testids[:200]
        target = out/f'seed{seed}'
        target.mkdir(exist_ok=True)
        models = {}
        for name in ['ce','all']:
            with (dev.O/f'development/model_seed{seed}_{name}.pkl').open('rb') as handle:
                models[name] = pickle.load(handle)
        caches = [score_cache(ids,split,models,target,smoke) for ids,split in [
            (fitids,'fitting'),(calids,'calibration'),(testids,'evaluation')]]
        saved = {}
        for name in ['ce','all']:
            parts = [simple.compact(f,f['score_'+name].to_numpy(float),queries) for f in caches]
            rows,records = evaluation(*parts,'context_'+name,target)
            summaries += [dict(row,seed=seed) for row in rows]
            saved.update(records)
        if not smoke:
            simple_dir = R/f'results/r5_simple_controls/full/fever_ce/seed{seed}'
            simple_rows = pd.read_csv(simple_dir/'summary.csv')
            for name in ['raw_cap5','raw_top1','list_logit_cap5','list_logit_top1','frozen_scet']:
                display = 'original_scet' if name=='frozen_scet' else 'simple_'+name
                for row in simple_rows[simple_rows.method==name].to_dict('records'):
                    stat = pd.read_parquet(simple_dir/f'query_{name}_{row["analysis"]}.parquet')
                    assert set(stat.query_id)==set(testids)
                    summaries.append({'seed':seed,'method':display,'analysis':row['analysis'],**summary(stat)})
                    if row['analysis'] in PLAN['interval_points']:
                        saved[(display,row['analysis'])] = stat
                        stat.to_parquet(target/f'query_{display}_{row["analysis"]}.parquet',index=False)
        pd.DataFrame([r for r in summaries if r['seed']==seed]).to_csv(target/'summary.csv',index=False)
        collection[seed] = saved
        del caches,models,saved
        gc.collect()
        print('context controls',seed,'complete;',round(time.monotonic()-started,1),'seconds',flush=True)
    data = pd.DataFrame(summaries)
    data.to_csv(out/'summary.csv',index=False)
    aggregate = data.groupby(['method','analysis'])[PLAN['outcomes']].agg(['mean','std'])
    aggregate.columns = ['_'.join(x) for x in aggregate.columns]
    aggregate.to_csv(out/'mean_sd.csv')
    if not smoke:
        bootstrap(collection,out)
        report(data,pd.read_csv(out/'paired_intervals.csv'),out)
    dump(out/'audit.json',{'passed':True,'smoke':smoke,'code_sha256':sha(__file__),
         'inputs':identities,'plan_sha256':sha(plan),'new_fitting':False,'test_point_selection':False,
         'label_and_order_invariance':smoke,'split_disjointness':True,'threads':2,
         'elapsed_seconds':time.monotonic()-started,'executable':sys.executable})
    print('Structural checks passed;',stage if (stage:='smoke' if smoke else 'full') else '',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
