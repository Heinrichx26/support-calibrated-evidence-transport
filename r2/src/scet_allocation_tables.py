"""Publication table specifications from frozen allocation-analysis results."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results/scet_allocation/full'


def display(value):
    return '—' if pd.isna(value) else f'{value:.3f}'


def allocation_table_specs():
    """Return S28-S29, with every displayed point checked from query counts."""
    summary = pd.read_csv(RES/'summary.csv')
    metrics = ['evidence_precision', 'coverage', 'evidence_recall', 'all_facts_recall',
               'false_admission_no_evidence']
    labels = {'scifact': 'SciFact', 'fever': 'FEVER', 'hover': 'HoVer'}
    methods = {'scet_cap5': 'SCET, cap 5', 'ce_cap1': 'Cross-encoder, cap 1',
               'ce_cap5': 'Cross-encoder, cap 5', 'multivers_cap1': 'MultiVerS, cap 1',
               'multivers_cap5': 'MultiVerS, cap 5'}
    chosen = {
        'scifact': ['scet_cap5', 'ce_cap1', 'ce_cap5', 'multivers_cap1', 'multivers_cap5'],
        'fever': ['scet_cap5', 'ce_cap1', 'ce_cap5'],
        'hover': ['scet_cap5', 'ce_cap1', 'ce_cap5']}
    rows = []
    for dataset, selected in chosen.items():
        for method in selected:
            partition_values = []
            for seed in [13, 17, 29, 7]:
                q = pd.read_csv(RES/f'query_{dataset}_seed{seed}_{method}.csv')
                assert not q.query_id.duplicated().any()
                assert q.admitted.sum() == len(q)
                cap = int(method.rsplit('cap', 1)[1])
                assert q.admitted.max() <= cap
                if cap == 1:
                    assert q.admitted.eq(1).all()
                gold = q.gold_count > 0
                values = [q.correct.sum()/q.admitted.sum(), q.covered.mean(),
                          (q.loc[gold, 'correct']/q.loc[gold, 'gold_count']).mean(),
                          q.loc[gold, 'all_facts'].mean(), q.loc[~gold, 'covered'].mean()]
                stored = summary[(summary.dataset == dataset) & (summary.seed == seed) & (summary.method == method)]
                assert len(stored) == 1
                assert np.allclose(values, stored.iloc[0][metrics].to_numpy(float), equal_nan=True, atol=1e-12, rtol=0)
                partition_values.append(values)
            average = pd.DataFrame(partition_values).mean().to_numpy(float)
            rows.append([labels[dataset], methods[method]] + [display(average[j]) for j in [0, 1, 3, 4]])
    table = ('Admission allocation at equal workload',
             ['Dataset', 'Method', 'Precision', 'Coverage', 'Set recall', 'No-evidence admission'],
             rows,
             'Means over four partitions; every policy admits 100 sentences per 100 queries. '
             'Cap is the maximum sentences per query; cap one covers every query. '
             'Set recall measures complete annotated evidence retention. '
             'Higher precision and recall and lower no-evidence admission are desirable. '
             'Cross-encoder ranks use raw scores. HoVer has no query without annotated evidence. '
             'FEVER uses the original June 2017 Wikipedia snapshot.')

    intervals = pd.read_csv(RES/'paired_intervals.csv')
    comparisons = {
        'scifact': [('scet_cap5', 'ce_cap1'), ('scet_cap5', 'ce_cap5'), ('multivers_cap5', 'multivers_cap1')],
        'fever': [('scet_cap5', 'ce_cap1'), ('scet_cap5', 'ce_cap5')],
        'hover': [('scet_cap5', 'ce_cap1'), ('scet_cap5', 'ce_cap5')]}
    names = {'scet_cap5': 'SCET 5', 'ce_cap1': 'Cross-encoder 1', 'ce_cap5': 'Cross-encoder 5',
             'multivers_cap5': 'MultiVerS 5', 'multivers_cap1': 'MultiVerS 1'}
    rows = []
    for dataset, pairs in comparisons.items():
        for method, reference in pairs:
            row = [labels[dataset], names[method]+' − '+names[reference]]
            for metric in ['evidence_precision', 'all_facts_recall', 'false_admission_no_evidence']:
                found = intervals[(intervals.dataset == dataset) & (intervals.method == method)
                                  & (intervals.reference == reference) & (intervals.metric == metric)]
                assert len(found) == 1
                result = found.iloc[0]
                if pd.isna(result.difference):
                    row.append('—')
                else:
                    row.append(f'{result.difference:.3f}\n[{result.ci_low:.3f}, {result.ci_high:.3f}]')
            rows.append(row)
    comparison = ('Paired differences between admission allocations',
                  ['Dataset', 'Comparison', 'Precision difference', 'Set-recall difference', 'No-evidence admission difference'],
                  rows,
                  'Numbers following method names give the per-query cap. Differences are first method minus second. '
                  'Brackets give pointwise 95% intervals from 1,000 paired query-cluster resamples shared across overlapping partitions, '
                  'conditional on frozen admission decisions. Allocation sensitivity is exploratory. '
                  'FEVER uses the original June 2017 Wikipedia snapshot.')
    return {28: table, 29: comparison}


if __name__ == '__main__':
    for number, (_, headers, rows, _) in allocation_table_specs().items():
        print('S'+str(number), len(rows), 'rows', len(headers), 'columns')
