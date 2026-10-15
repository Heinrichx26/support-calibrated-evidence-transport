"""Standalone additional tables from frozen public scientific results."""
from pathlib import Path
import pandas as pd
R2 = Path(__file__).resolve().parents[1]
RES = R2 / 'results/multivers_extension'

def fs(x, d=3):
    return f'{x:.{d}f}'

def ms(frame, column, d=3):
    return fs(frame[column].mean(), d) + '\n(' + fs(frame[column].std(ddof=1), d) + ')'

def data():
    extension = pd.read_csv(RES / 'k100/summary.csv')
    context = pd.read_csv(RES / 'context/full/summary.csv')
    interval = pd.read_csv(RES / 'context/full/paired_intervals.csv')
    exact = context[context.analysis == 'exact_budget']
    full = exact[exact.method == 'multivers_full']
    single = exact[exact.method == 'multivers_single']
    contrast = interval[(interval.analysis == 'exact_budget') & (interval.metric == 'evidence_precision')].iloc[0]
    return (extension, context, full, single, contrast)

def tables():
    extension, context, full, single, c = data()
    exact = extension[extension.analysis == 'exact_budget']
    rows = []
    for name, f, input_context in [('SCET', exact[exact.method == 'scet'], 'Title and sentence'), ('MultiVerS', full, 'Title and abstract'), ('SCET with MultiVerS', exact[exact.method == 'scet_multivers'], 'Title and abstract'), ('MultiVerS', single, 'Title and sentence')]:
        rows.append([name, input_context, ms(f, 'evidence_precision'), fs(f.evidence_recall.mean()), fs(f.all_facts_recall.mean()), fs(f.coverage.mean())])
    first = ('SciFact combinations and context ablation', ['Method', 'Input context', 'Precision (SD)', 'Recall', 'Set recall', 'Coverage'], rows, 'Every method admits 100 sentences per 100 held-out queries. Values average four partitions; SD denotes sample standard deviation. Higher precision, recall, complete-set recall, and coverage indicate more effective evidence retention at this workload.')
    rules = pd.read_csv(RES / 'k100/risk_rules.csv')
    rows = []
    labels = {'scet': 'SCET', 'multivers': 'MultiVerS', 'scet_multivers': 'SCET with MultiVerS'}
    for alpha in [0.2, 0.3]:
        for method in ['scet', 'multivers', 'scet_multivers']:
            f = extension[(extension.analysis == f'risk_{alpha:g}') & (extension.method == method)]
            n = rules[(rules.alpha == alpha) & (rules.method == method)].nonempty_feasible.sum()
            rows.append([fs(alpha, 2), labels[method], f'{n}/4', ms(f, 'coverage'), ms(f, 'annotated_per_100', 1), ms(f, 'query_error_rate')])
    second = ('SciFact comparison under common query-risk constraints', ['Risk target', 'Method', 'Nonempty', 'Coverage (SD)', 'Ann./100 (SD)', 'Query error (SD)'], rows, 'Nonempty counts partitions with a passing rule. Each method tests 20 fitting-derived thresholds using 30 independent calibration queries and failure probability 0.05. Higher coverage and yield, and lower query error, indicate better outcomes. Target 0.10 returns empty output throughout.')
    return {23: first, 24: second}
