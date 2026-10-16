"""Recreate supplementary Tables S32-S35 from archived numeric results."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def only(rows, **criteria):
    values = [r for r in rows if all(r[k] == str(v) for k, v in criteria.items())]
    assert len(values) == 1, criteria
    return values[0]

def mean_sd(row, metric, digits=3):
    return f'{float(row[metric+"_mean"]):.{digits}f}\n({float(row[metric+"_std"]):.{digits}f})'

def tables():
    evaluation = read('results/r5_context_gain/evaluation/mean_sd.csv')
    controls = read('results/r5_context_gain/controls/full/mean_sd.csv')
    names = {'context_all': 'SCET, list features', 'context_ce': 'Cross-encoder features',
             'simple_list_logit_cap5': 'Cross-encoder list cutoff',
             'simple_list_logit_top1': 'Cross-encoder top-one gate',
             'original_scet': 'SCET (Table 3)'}
    rows = []
    for method in names:
        if method == 'simple_list_logit_top1': continue
        row = only(controls, method=method, analysis='exact_budget_100')
        rows.append([names[method]+'\nFixed batch', mean_sd(row, 'volume_per_100', 1),
                     mean_sd(row, 'evidence_precision'), mean_sd(row, 'all_facts_recall'),
                     mean_sd(row, 'false_admission_no_evidence')])
    row = only(evaluation, method='original_ce', protocol='fixed_batch')
    rows.append(['Raw cross-encoder\nFixed batch', mean_sd(row, 'volume_per_100', 1),
                 mean_sd(row, 'evidence_precision'), mean_sd(row, 'all_facts_recall'),
                 mean_sd(row, 'false_admission_no_evidence')])
    for group, label in [('all', names['context_all']), ('ce', names['context_ce'])]:
        row = only(evaluation, method=group, protocol='prospective')
        rows.append([label+'\nCalibrated threshold', mean_sd(row, 'volume_per_100', 1),
                     mean_sd(row, 'evidence_precision'), mean_sd(row, 'all_facts_recall'),
                     mean_sd(row, 'false_admission_no_evidence')])
    result = {'S32': {'title': 'FEVER list features at a common sentence budget',
        'headers': ['Scoring form and rule', 'Items/100 (SD)', 'Precision (SD)', 'Set recall (SD)', 'No-evidence admission (SD)'],
        'rows': rows, 'widths': [.30, .14, .17, .18, .20],
        'bold': [[0,2], [0,3], [3,4], [5,2], [6,3], [5,4]],
        'note': 'Means (sample standard deviations) over four partitions. Fixed batch admits exactly 100 sentences per 100 queries; calibrated thresholds target this workload on independent calibration queries. All rules admit at most five sentences per query. The two feature models share the same twelve-view candidate screen and model capacity. Higher precision and set recall and lower no-evidence admission are preferable. Bold identifies the best displayed value within each rule. The cross-encoder feature control uses cross-encoder predictors on the shared screen.'}}
    rows=[]
    for target, label in [('query_error_rate_0.1','Query error\n0.10'), ('false_admission_no_evidence_0.1','No-evidence\n0.10')]:
        for method in names:
            row=only(controls, method=method, analysis=target)
            error='query_error_rate' if target.startswith('query') else 'false_admission_no_evidence'
            rows.append([label,names[method],mean_sd(row,'evidence_precision'),
                         mean_sd(row,'volume_per_100',1),mean_sd(row,'all_facts_recall'),mean_sd(row,error)])
    result['S33']={'title':'FEVER admission under empirical calibration error targets',
        'headers':['Calibration target','Scoring form','Precision (SD)','Items/100 (SD)','Set recall (SD)','Test error (SD)'],
        'rows':rows,'widths':[.14,.27,.145,.145,.145,.145],
        'bold':[[0,2],[0,4],[5,2],[5,4]],
        'note':'Means (sample standard deviations) over four partitions. Independent calibration queries select the greatest-workload threshold satisfying the stated empirical error target, from a fitting-defined grid. Test error uses all queries for query error and only queries without annotated evidence for no-evidence admission. Higher precision and set recall and lower error are preferable; workload identifies each operating point. Bold identifies the largest precision and set recall within each target. These empirical targets are distinct from the simultaneous conditional-risk bounds in Table S34.'}
    rules=read('results/r5_risk_conflict/risk/full/selected_rules.csv')
    outcomes=read('results/r5_risk_conflict/risk/full/evaluation_by_partition.csv')
    rows=[]
    for method, seed, label in [('scet',29,'SCET (29)'),('ce',17,'Cross-encoder (17)'),('ce',7,'Cross-encoder (7)')]:
        row=only(rules,dataset='fever',method=method,seed=seed)
        test=only(outcomes,dataset='fever',method=method,seed=seed)
        rows.append([label]+[f'{float(row[k]):.4f}' for k in ['calibration_conditional_error_upper','calibration_coverage_lower']]
                    +[f'{float(test[k]):.4f}' for k in ['coverage','conditional_query_error']])
    new_rules=read('results/r5_context_gain/conditional_risk/full/selected_rules.csv')
    new_outcomes=read('results/r5_context_gain/conditional_risk/full/evaluation_by_partition.csv')
    for method,label in [('all','SCET, list features'),('ce','Cross-encoder features')]:
        for seed in [13,17,29,7]:
            row=only(new_rules,method=method,seed=seed)
            test=only(new_outcomes,method=method,seed=seed)
            assert row['feasible']=='True'
            rows.append([f'{label} ({seed})']+[f'{float(row[k]):.4f}' for k in ['calibration_conditional_error_upper','calibration_coverage_lower']]
                        +[f'{float(test[k]):.4f}' for k in ['coverage','conditional_query_error']])
    result['S34']={'title':'Joint conditional-risk and coverage calibration',
        'headers':['Method (seed)','Calibration risk upper bound','Calibration coverage lower bound','Test coverage','Test conditional error'],
        'rows':rows,'widths':[.28,.19,.19,.16,.18],
        'bold':[[i,j] for i in range(len(rows)) for j in [1,2]],
        'note':'Nonempty FEVER rules satisfying a conditional-risk upper bound of 0.20 and a coverage lower bound of 0.05. One-sided Clopper–Pearson bounds use separate within-partition Bonferroni families: 120 bounds for the three original scorers and 80 for the two feature models. Each scorer has 20 fitting-defined thresholds. Bold bounds jointly pass the requirements. Other original dataset–method–partition combinations return empty output. Lower conditional error and higher coverage are preferable.'}
    ablation=read('results/r5_risk_conflict/conflict/full/paired_intervals.csv')
    ablation_means=read('results/r5_risk_conflict/conflict/full/evaluation_mean_sd.csv')
    rows=[]
    for dataset,label in [('scifact','SciFact'),('fever','FEVER'),('hover','HoVer')]:
        row=only(ablation,analysis='exact_budget',dataset=dataset,method='without_explicit_conflict_indicator',metric='evidence_precision')
        original=only(ablation_means,dataset=dataset,analysis='exact_budget',method='original')
        rows.append([label,f'{float(original["evidence_precision_mean"]):.4f}',
            f'{float(row["difference"]):.4f}\n[{float(row["ci_low"]):.4f}, {float(row["ci_high"]):.4f}]'])
    result['S35']={'title':'Refitted ablation of the explicit conflict indicator',
        'headers':['Dataset','Original precision','Precision difference after removal (95% interval)'],
        'rows':rows,'widths':[.20,.25,.55],'bold':[],
        'note':'Ablated minus original precision at exactly 100 sentences per 100 queries. The ablation removes the explicit indicator from logistic inputs, the utility score and the external penalty, then repeats fitting, form selection and independent threshold calibration. Component scores remain available. Intervals use 1,000 paired query-cluster resamples shared across overlapping partitions, conditional on the fitted rules and decisions. Higher precision is preferable. All three intervals include zero.'}
    return result

def main():
    values=tables()
    dest=ROOT/'results/tables/additional_validation'
    dest.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for key,value in values.items():
        path=dest/f'{key}.csv'
        with path.open('w',encoding='utf-8',newline='') as handle:
            csv.writer(handle,lineterminator='\n').writerows([value['headers']]+value['rows'])
        manifest.append({'table':key,'file':str(path.relative_to(ROOT)).replace('\\','/'),
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(value['rows']),
                         'columns':len(value['headers']),'title':value['title']})
    out=ROOT/'results/r5_strengthening';out.mkdir(parents=True,exist_ok=True)
    (out/'additional_table_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'table_content.json').write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Recreated S32-S35 from numeric results.')

if __name__=='__main__':main()
