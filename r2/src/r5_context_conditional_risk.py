"""Fixed context scorers: jointly bounded conditional query risk and coverage."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import beta
from threadpoolctl import threadpool_limits

import r5_context_gain as dev
import r5_context_controls as controls
import r5_simple_controls as simple

O = dev.O / 'conditional_risk'
CACHE = dev.O / 'controls/full'
PLAN = {
    'version': '20261016_fixed_context_two_scorer_conditional_risk',
    'methods': ['ce', 'all'],
    'seeds': [13, 17, 29, 7],
    'fitting_top_one_quantiles': (np.arange(20) * .05).tolist(),
    'conditional_query_error_target': .20,
    'coverage_lower_target': .05,
    'family_failure_probability': .05,
    'family_size': 80,
    'one_sided_failure_probability': .05 / 80,
    'family': '2 fixed scorers x 20 fitting-defined thresholds x 2 one-sided bounds, separately within each fixed partition',
    'risk_event': 'At least one admitted candidate without annotated evidence, conditional on any output for a query.',
    'coverage_event': 'At least one admitted candidate for a query.',
    'bounds': 'One-sided exact Clopper-Pearson upper limit for error count among covered queries; lower limit for covered count among all queries.',
    'selection': 'Among rules with conditional-error upper limit <= .20 and coverage lower limit >= .05, maximize coverage lower limit; ties choose lower threshold.',
    'admission': 'Frozen model score >= threshold, maximum five candidates per query, fixed saved tie keys.',
    'empty_rule': 'When no rule is jointly feasible, output nothing; conditional error is undefined and the coverage requirement fails.',
    'no_fitting': True,
    'no_test_selection': True,
    'smoke': 'Seed13 complete fitting/calibration score caches only; do not open evaluation caches; check order invariance, bounds, risk identity and split disjointness.',
    'assumptions': 'Independent identically distributed future queries under the same candidate protocol, with fixed scorers and fitting-defined grids independent of calibration; guarantee is per partition and conditional on its fixed fit.',
    'scope': 'Independent two-scorer exploratory family; distinct from the original three-scorer family and from the descriptive empirical-risk controls.',
    'threads': 2,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bounds(n, covered, errors):
    assert 0 <= errors <= covered <= n
    eta = PLAN['one_sided_failure_probability']
    upper = float(beta.ppf(1-eta, errors+1, covered-errors)) if errors < covered else 1.
    lower = float(beta.ppf(eta, covered, n-covered+1)) if covered else 0.
    return upper, lower


def calibrate(fit, cal, method):
    f = simple.compact(fit, fit['score_'+method].to_numpy(float), {})
    c = simple.compact(cal, cal['score_'+method].to_numpy(float), {})
    grid = np.quantile(f['raw'][f['ranks']==1], PLAN['fitting_top_one_quantiles'])
    tests = []
    for i, threshold in enumerate(grid):
        mask = c['raw'] >= threshold
        n = len(c['qids'])
        covered = int((np.bincount(c['codes'],weights=mask,minlength=n)>0).sum())
        errors = int((np.bincount(c['codes'],weights=mask*(1-c['y']),minlength=n)>0).sum())
        if covered:
            assert np.isclose(errors/n,(covered/n)*(errors/covered))
        upper, lower = bounds(n, covered, errors)
        tests.append({'method': method, 'grid_index': i,
                      'fitting_quantile': PLAN['fitting_top_one_quantiles'][i],
                      'threshold': float(threshold), 'queries': n,
                      'covered_queries': covered, 'error_queries': errors,
                      'admitted': int(mask.sum()),
                      'conditional_error': errors/covered if covered else np.nan,
                      'coverage': covered/n,
                      'conditional_error_upper': upper, 'coverage_lower': lower,
                      'one_sided_failure_probability': PLAN['one_sided_failure_probability'],
                      'passes': bool(covered and upper <= .20 and lower >= .05)})
    passing = [r for r in tests if r['passes']]
    choice = max(passing, key=lambda r: (r['coverage_lower'], -r['threshold'])) if passing else None
    return tests, choice


def report(data, selected, out):
    means = data.groupby('method').mean(numeric_only=True)
    lines = ['# Fixed context scorers: conditional query risk', '',
        'The two fixed scorers share a separate family of 80 one-sided bounds per partition. Twenty thresholds use fitting top-one quantiles from 0 to 0.95. Independent calibration chooses maximum certified coverage, subject to a conditional query-error upper bound of 0.20 and a coverage lower bound of 0.05. The family failure probability is 0.05.', '',
        '| Method | Feasible partitions | Precision | Coverage | Conditional query error | Unconditional query error | Items/100 | Set recall | Singleton | Multiple | No-evidence admission |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in PLAN['methods']:
        r = means.loc[method]
        names = ['evidence_precision','coverage','conditional_query_error','query_error_rate','volume_per_100','all_facts_recall','singleton_set_recall','multiple_set_recall','false_admission_no_evidence']
        lines.append('| '+method+' | '+str(int(data.loc[data.method==method,'feasible'].sum()))+'/4 | '+' | '.join(f'{r[n]:.4f}' for n in names)+' |')
    lines += ['', '| Seed | Method | Feasible | Calibration conditional-error upper | Calibration coverage lower | Test conditional error | Test coverage |',
              '|---:|---|---|---:|---:|---:|---:|']
    for r in data.merge(selected,on=['seed','method','feasible']).itertuples():
        lines.append(f'| {r.seed} | {r.method} | {r.feasible} | {r.calibration_conditional_error_upper:.4f} | {r.calibration_coverage_lower:.4f} | {r.conditional_query_error:.4f} | {r.coverage:.4f} |')
    lines += ['',
        'Validity requires independent identically distributed future queries under the same candidate protocol and a fixed scorer/grid independent of calibration. The family correction covers both bound types and both scorers within each partition. Repeated partitions are reported as sensitivity analyses; no simultaneous guarantee across all four partitions is asserted.', '',
        'The conditional event uses annotation-based candidate errors among covered queries. Empty output has undefined conditional error and fails the minimum-coverage requirement. Every chosen rule is saved before its held-out cache is opened. All 40 candidate rules and their two bounds per partition are retained. This family is distinct from the original three-scorer analysis and the empirical risk-target scans. No model is refitted.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(smoke):
    start = time.monotonic()
    O.mkdir(parents=True,exist_ok=True)
    plan = O/'analysis_plan.json'
    if plan.exists():
        assert load(plan) == PLAN
    else:
        dump(plan,PLAN)
    out = O/('smoke' if smoke else 'full')
    out.mkdir(exist_ok=True)
    freezes = load(dev.O/'development/freeze.json')
    identities = {'model_freeze_sha256':sha(dev.O/'development/freeze.json'),
                  'cached_score_generation_code_sha256':sha(controls.__file__),
                  'source_simple_statistics_sha256':sha(simple.__file__),
                  'cache_audit_sha256':sha(CACHE/'audit.json')}
    if not smoke:
        gate = load(O/'smoke/audit.json')
        assert gate['passed'] and gate['code_sha256']==sha(__file__) and gate['inputs']==identities
    all_tests, all_selected, results, cache_hashes, fixed = [],[],[],{},[]
    for freeze in freezes[:1] if smoke else freezes:
        seed = freeze['seed']
        ids = [sorted(freeze[k]) for k in ['fitting_query_ids','threshold_query_ids','evaluation_query_ids']]
        assert not(set(ids[0])&set(ids[1]) or set(ids[0])&set(ids[2]) or set(ids[1])&set(ids[2]))
        fitpath, calpath = [CACHE/f'seed{seed}/candidate_scores_{split}.parquet' for split in ['fitting','calibration']]
        fit, cal = pd.read_parquet(fitpath), pd.read_parquet(calpath)
        assert set(fit.query_id)==set(ids[0]) and set(cal.query_id)==set(ids[1])
        cache_hashes[str(seed)] = {'fitting':sha(fitpath),'calibration':sha(calpath)}
        choices = {}
        for method in PLAN['methods']:
            tests,choice = calibrate(fit,cal,method)
            all_tests += [dict(r,seed=seed) for r in tests]
            choices[method] = choice
            if smoke:
                reverse_tests, reverse_choice = calibrate(fit.iloc[::-1],cal.iloc[::-1],method)
                assert tests==reverse_tests and choice==reverse_choice
        dump(out/f'freeze_seed{seed}.json',{'seed':seed,'chosen':choices,'fitting_ids':ids[0],
              'calibration_ids':ids[1],'evaluation_ids':ids[2],'input_hashes':cache_hashes[str(seed)],
              'test_cache_opened_at_freeze':False,'models_refitted':False})
        fixed.append((seed,ids[2],choices))
        print('conditional risk calibration',seed,'smoke' if smoke else 'full', {k:bool(v) for k,v in choices.items()},round(time.monotonic()-start,1),'seconds',flush=True)
    if not smoke:
        # All four calibration choices are serialized before opening evaluation
        # annotations or evaluation score caches.
        dump(out/'all_choices_frozen_before_evaluation.json',{
            'seeds':[x[0] for x in fixed],'all_choices_saved':True,
            'test_caches_or_query_annotations_opened':False})
        queries = load(dev.R/'data/processed/fever_full_queries.json')
        for seed,testids,choices in fixed:
            testpath = CACHE/f'seed{seed}/candidate_scores_evaluation.parquet'
            test = pd.read_parquet(testpath)
            assert set(test.query_id)==set(testids)
            cache_hashes[str(seed)]['evaluation'] = sha(testpath)
            for method, choice in choices.items():
                t = simple.compact(test,test['score_'+method].to_numpy(float),queries)
                threshold = choice['threshold'] if choice else np.inf
                stat = simple.statistics(t,t['raw']>=threshold)
                stat.to_parquet(out/f'query_seed{seed}_{method}.parquet',index=False)
                r = controls.summary(stat)
                if stat.covered.any():
                    assert np.isclose(r['query_error_rate'],r['coverage']*r['conditional_query_error'])
                results.append({'seed':seed,'method':method,'feasible':bool(choice),**r})
                all_selected.append({'seed':seed,'method':method,'feasible':bool(choice),
                    'threshold':threshold,
                    'calibration_conditional_error_upper':choice['conditional_error_upper'] if choice else np.nan,
                    'calibration_coverage_lower':choice['coverage_lower'] if choice else 0.,
                    'calibration_covered_queries':choice['covered_queries'] if choice else 0})
            print('conditional risk evaluation',seed,round(time.monotonic()-start,1),'seconds',flush=True)
    assert bounds(100,0,0)==(1.,0.)
    u,l = bounds(100,50,0)
    assert 0<u<1 and 0<l<.5
    pd.DataFrame(all_tests).to_csv(out/'calibration_tests.csv',index=False)
    if not smoke:
        data, selected = pd.DataFrame(results),pd.DataFrame(all_selected)
        data.to_csv(out/'evaluation_by_partition.csv',index=False)
        selected.to_csv(out/'selected_rules.csv',index=False)
        mean = data.groupby('method')[PLAN_METRICS].agg(['mean','std'])
        mean.columns = ['_'.join(x) for x in mean.columns]
        mean.to_csv(out/'evaluation_mean_sd.csv')
        report(data,selected,out)
    dump(out/'audit.json',{'passed':True,'smoke':smoke,'code_sha256':sha(__file__),
         'plan_sha256':sha(plan),'inputs':identities,'cache_hashes':cache_hashes,
         'new_fitting':False,'test_point_selection':False,'smoke_test_cache_access':False,
         'split_disjointness':True,'order_invariance_checked_in_smoke':True,
         'threads':2,'elapsed_seconds':time.monotonic()-start})


PLAN_METRICS = ['evidence_precision','volume_per_100','coverage','query_error_rate',
                'conditional_query_error','false_admission_no_evidence','all_facts_recall',
                'singleton_set_recall','multiple_set_recall']

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().smoke)
