"""Data-driven evidence-set-size table, without experiment dependencies."""
from pathlib import Path
import numpy as np
import pandas as pd

R2 = Path(__file__).resolve().parents[1]
OUT = R2 / 'results/substantive_strengthening/set_size'
DATASETS = ['scifact', 'fever', 'hover']
METHODS = ['scet_batch', 'ce_batch', 'ce_top1', 'scet_top1']


def gold_set_table_specs():
    mean = pd.read_csv(OUT / 'full/mean_sd.csv')
    labels = {'scet_batch': 'SCET', 'ce_batch': 'Cross-encoder',
              'ce_top1': 'Cross-encoder top 1', 'scet_top1': 'SCET top 1'}
    datasets = {'scifact': 'SciFact', 'fever': 'FEVER', 'hover': 'HoVer'}
    rows = []
    def fmt(row, col):
        a, b = row[col + '_mean'], row[col + '_std']
        return f'{a:.3f} ({b:.3f})' if np.isfinite(a) else '—'
    for ds in DATASETS:
        for method in METHODS:
            s = mean[(mean.dataset == ds) & (mean.method == method)].set_index('group')
            rows.append([datasets[ds], labels[method], fmt(s.loc['all_queries'], 'precision'),
                         fmt(s.loc['annotated_queries'], 'set_recall_conditional'),
                         fmt(s.loc['singleton_set'], 'set_recall_conditional'),
                         fmt(s.loc['multisentence_set'], 'set_recall_conditional')])
    note = ('Means (sample standard deviations) across four partitions. All methods admit 100 sentences per 100 queries. '
            'Cross-encoder decisions rank its raw scores. '
            'Set recall uses all originally annotated queries. Singleton and multiple use queries whose smallest annotated '
            'evidence set contains one sentence or at least two sentences, respectively. HoVer has no singleton queries. '
            'Higher precision and recall indicate more annotated evidence at the same workload. '
            'FEVER uses the original June 2017 Wikipedia snapshot.')
    return {30: ('Complete evidence sets at equal sentence budgets',
                 ['Dataset', 'Method', 'Precision (SD)', 'Set recall (SD)', 'Singleton (SD)', 'Multiple (SD)'], rows, note)}
