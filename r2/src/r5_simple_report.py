"""Combine and independently check the frozen simple-control outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
O = R/'results/r5_simple_controls'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    paths = sorted((O/'full').glob('*_*/seed*/summary.csv'))
    data = pd.concat([pd.read_csv(x) for x in paths], ignore_index=True)
    assert not data.duplicated(['dataset','source','seed','method','analysis']).any()
    assert len(data.groupby(['dataset','source']).seed.unique()) == 5
    assert data.groupby(['dataset','source']).seed.nunique().eq(4).all()
    columns = [c for c in data.select_dtypes(include='number') if c != 'seed']
    aggregate = data.groupby(['dataset','source','method','analysis'])[columns].agg(['mean','std'])
    aggregate.columns = ['_'.join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index()
    data.to_csv(O/'combined_by_partition.csv', index=False)
    aggregate.to_csv(O/'combined_mean_sd.csv', index=False)
    interval_paths = sorted((O/'full').glob('*_*/paired_intervals.csv'))
    intervals = pd.concat([pd.read_csv(x) for x in interval_paths], ignore_index=True)
    intervals.to_csv(O/'combined_paired_intervals.csv', index=False)
    checks = []
    for path in paths:
        source = path.parent.parent.name
        dataset = source.split('_')[0]
        seed = int(path.parent.name[4:])
        for method, reference in [('frozen_scet','scet'), ('raw_cap5',source.split('_',1)[1])]:
            saved = pd.read_parquet(path.parent/f'query_{method}_exact_budget_100.parquet').set_index('query_id')
            if reference == 'multivers':
                oldpath = R/f'results/multivers_extension/k100/query_seed{seed}_multivers_exact_budget.csv'
            else:
                oldpath = R/f'results/strengthening/{dataset}_full_k100/query_seed{seed}_{reference}.csv'
            old = pd.read_csv(oldpath, dtype={'query_id':str}).set_index('query_id').reindex(saved.index)
            assert old.correct.notna().all()
            for c in ['admitted','correct','covered','all_facts']:
                np.testing.assert_array_equal(saved[c].to_numpy(int), old[c].to_numpy(int))
            checks.append({'dataset': dataset, 'source': source, 'seed': seed, 'method': method,
                           'same_query_counts_correct_coverage_complete_sets_as_prior': True})
        for path1 in path.parent.glob('query_*top1*.parquet'):
            stat = pd.read_parquet(path1)
            assert stat.admitted.max() <= 1
            assert not stat.loc[stat.minimum_set_size > 1,'all_facts'].any()
        for path1 in path.parent.glob('query_*.parquet'):
            stat = pd.read_parquet(path1)
            assert stat.admitted.max() <= 5
            assert stat.correct.le(stat.admitted).all()
            assert stat.all_facts.le(stat.gold_count.gt(0)).all()
    (O/'independent_result_checks.json').write_text(json.dumps({'passed':True,
        'old_result_matches': checks, 'all_top1_multisentence_completions_zero':True,
        'all_admissions_at_most_five':True,'report_code_sha256':sha(__file__)}, indent=2),encoding='utf-8')
    def val(ds, source, method, analysis, field):
        row = aggregate[(aggregate.dataset==ds)&(aggregate.source==source)&(aggregate.method==method)&(aggregate.analysis==analysis)]
        return float(row.iloc[0][field+'_mean'])
    def ci(ds, source, reference, analysis, metric):
        row = intervals[(intervals.dataset==ds)&(intervals.source==source)&(intervals.reference==reference)&(intervals.analysis==analysis)&(intervals.metric==metric)].iloc[0]
        return f"{row.difference:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]"
    lines = [
        '# Simple single-scorer allocation and abstention controls',
        '',
        'The frozen original SCET does not retain an advantage over a simple single-scorer list calibration at the common FEVER sentence budget. Under an independently calibrated query-error target, the simple controls also retain competitive or higher precision and complete-set recall. Under a no-evidence-query admission constraint, SCET retains more complete evidence sets while the simple controls have higher evidence precision. These results distinguish the constraints; they do not support a general independent multi-view gain from the original FEVER comparison.',
        '',
        'All original source candidates and four fitting/calibration/test partitions are preserved. A single raw scorer supplies candidate scores and eight query-list summaries: maximum, first–second and first–fifth gaps, first-five mean/SD, full-pool mean/SD, and candidate count. The cap-five logistic model additionally uses candidate score, gap to the maximum, and rank indicators. A cumulative minimum preserves the raw candidate ordering, so this control changes the per-query admitted prefix length. A separate logistic gate predicts membership of the top-ranked sentence and permits zero output. The target is annotated verification-evidence membership.',
        '',
        'Only fitting queries train models and select C from 0.1/1/10 using three inner query folds. List-model selection uses OOF precision at 100/100; top-one gate selection uses OOF Brier score. Independent calibration queries fix every prospective workload threshold. All workload targets 25/50/100/200/300/400/500 are retained; top-one policies saturate at 100. Separate empirical calibration constraints use query-error and no-evidence admission targets 0.01/0.025/0.05/0.1/0.2/0.3. Their threshold grids use fitting predictions. No held-out outcome selects a model or threshold, and no finite-sample guarantee is claimed for these empirical risk constraints.',
        '',
        'The common fixed-batch budget is a separate label-free score-order diagnostic. The inference grids and models remained fixed after the successful smoke run. Every specified source and operating point, including negative results, is saved.',
        '',
        '## FEVER',
        '',
        '| Policy | Precision | Items/100 | Coverage | Set recall | Query error | No-evidence admission |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    primary = [('frozen_scet','exact_budget_100'),('raw_cap5','exact_budget_100'),
               ('list_logit_cap5','exact_budget_100'),('list_logit_top1','exact_budget_100'),
               ('frozen_scet','query_error_rate_0.1'),('list_logit_cap5','query_error_rate_0.1'),
               ('list_logit_top1','query_error_rate_0.1'),
               ('frozen_scet','false_admission_no_evidence_0.1'),
               ('list_logit_cap5','false_admission_no_evidence_0.1'),
               ('list_logit_top1','false_admission_no_evidence_0.1')]
    for method, analysis in primary:
        values = [val('fever','ce',method,analysis,f) for f in ['evidence_precision','volume_per_100','coverage','all_facts_recall','query_error_rate','false_admission_no_evidence']]
        lines.append('| '+method+' / '+analysis+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
    lines += ['',
        'At the exact sentence budget, SCET minus CE-list calibration precision is '+ci('fever','ce','list_logit_cap5','exact_budget_100','evidence_precision')+'. Its complete-set recall difference is '+ci('fever','ce','list_logit_cap5','exact_budget_100','all_facts_recall')+'.',
        '',
        'At the calibration query-error target 0.10, the corresponding precision difference is '+ci('fever','ce','list_logit_cap5','query_error_rate_0.1','evidence_precision')+', complete-set difference '+ci('fever','ce','list_logit_cap5','query_error_rate_0.1','all_facts_recall')+', and held-out query-error difference '+ci('fever','ce','list_logit_cap5','query_error_rate_0.1','query_error_rate')+'. This comparison has similar measured error rates and does not depend on forced top-one output.',
        '',
        'At the calibration no-evidence-admission target 0.10, SCET minus top-one logistic gate precision is '+ci('fever','ce','list_logit_top1','false_admission_no_evidence_0.1','evidence_precision')+', complete-set difference '+ci('fever','ce','list_logit_top1','false_admission_no_evidence_0.1','all_facts_recall')+', and actual no-evidence-admission difference '+ci('fever','ce','list_logit_top1','false_admission_no_evidence_0.1','false_admission_no_evidence')+'. SCET retains more annotated material under this constraint; its precision is lower.',
        '', '## Other collections', '',
        '| Collection/source | Policy at exact 100/100 | Precision | Coverage | Set recall |',
        '|---|---|---:|---:|---:|']
    for ds, source in [('hover','ce'),('hover','e5'),('scifact','ce'),('scifact','multivers')]:
        for method in ['frozen_scet','raw_cap5','list_logit_cap5','list_logit_top1']:
            numbers = [val(ds,source,method,'exact_budget_100',f) for f in ['evidence_precision','coverage','all_facts_recall']]
            lines.append('| '+ds+'/'+source+' | '+method+' | '+' | '.join(f'{x:.4f}' for x in numbers)+' |')
    lines += ['',
        'HoVer list controls substantially improve sentence precision, while original SCET retains more complete multi-sentence sets. The raw E5 control remains competitive on complete sets. SciFact CE-list calibration improves over original SCET; MultiVerS raw scores retain higher sentence precision than the additional list calibration. Here frozen_scet denotes the original twelve-view family, whose SciFact result is separate from the already reported SCET-with-MultiVerS equality.',
        '',
        '## Integrity, scope, and files', '',
        'The independent result check reproduces every original exact-budget SCET and raw-scorer query-level admitted count, correct count, coverage, and complete-set outcome for all 20 dataset/source/seed combinations. Every top-one policy has zero complete-set success on claims requiring several sentences. Every control respects its per-query cap.',
        '',
        'Intervals use 1,000 paired query-cluster bootstrap draws shared across overlapping partitions, conditional on frozen models and decisions. They are pointwise intervals. Empty-output precision is undefined and excluded where it occurs; coverage, output, and error remain explicitly reported. Empirical calibration targets need not equal held-out error rates. These additional analyses use previously studied official development collections and remain exploratory.',
        '',
        'The experiment reuses .venv_scet without installations, performs no neural inference, and limits numerical libraries to two threads. Source code, analysis plan, environment versions, model pickles, selected C, all OOF candidates, split IDs, threshold grids, per-query outcomes, means/SD and paired intervals are retained under this results directory. No paper or core admission implementation was modified.',
        '',
        '- `combined_by_partition.csv`: every specified point for all sources and partitions.',
        '- `combined_mean_sd.csv`: four-partition means and sample SD.',
        '- `combined_paired_intervals.csv`: SCET-minus-control intervals for predeclared primary comparisons.',
        '- `full/{dataset}_{source}/seed{seed}/`: models, fitting choices, thresholds, and query outputs.',
        '- `smoke/audit.json`, `fever_audit.json`, and `full/audit.json`: smoke and phase-specific execution records.',
        '- `independent_result_checks.json`: independent equality checks against original outputs.',
    ]
    (O/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest = {str(p.relative_to(O)): sha(p) for p in [O/'report.md',O/'combined_by_partition.csv',
        O/'combined_mean_sd.csv',O/'combined_paired_intervals.csv',O/'independent_result_checks.json']}
    manifest['source/r5_simple_controls.py'] = sha(R/'src/r5_simple_controls.py')
    manifest['source/r5_simple_report.py'] = sha(__file__)
    (O/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Validated original outcomes for',len(checks),'method/partition comparisons; report saved.')
    print(aggregate[(aggregate.dataset=='fever')&(aggregate.analysis=='query_error_rate_0.1')][
        ['method','evidence_precision_mean','coverage_mean','all_facts_recall_mean','query_error_rate_mean']].to_string(index=False))


if __name__ == '__main__':
    run()
