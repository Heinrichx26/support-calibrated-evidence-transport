"""Draw external comparison diagnostics from saved CSV results only.

The figure is a supplementary reproduction diagnostic. No inference, fitting,
or bootstrap resampling is invoked. PDF text remains selectable and the SVG
is editable; all source values are retained in the input CSVs.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R2 = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=R2 / 'results/multivers_extension/external_evaluation')
    parser.add_argument('--output', type=Path, default=R2 / 'results/external_figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.input / 'summary.csv')
    intervals = pd.read_csv(args.input / 'paired_intervals.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), layout='constrained')
    for ax, (dataset, label) in zip(axes, [('healthver', 'HealthVer'), ('covidfact', 'COVID-Fact')]):
        part = summary[(summary.dataset == dataset) & (summary.analysis == 'exact_budget')].set_index('method')
        methods = ['selected', 'best_single', 'best_blend', 'without_context', 'without_focus']
        y = np.arange(len(methods))
        ax.barh(y, part.loc[methods, 'evidence_precision'], color='0.45')
        ax.set_yticks(y, ['SCET calibration', 'Selected single context', 'Selected arithmetic blend',
                          'Without context features', 'All-candidate fitting'])
        ax.invert_yaxis()
        ax.set(xlim=(0, 1), xlabel='Evidence precision', title=label)
        for j, method in enumerate(methods):
            value = part.loc[method, 'evidence_precision']
            ax.text(value + .012, j, f'{value:.3f}', va='center', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    for extension in ['pdf', 'svg']:
        fig.savefig(args.output / f'external_precision.{extension}', bbox_inches='tight')
    plt.close(fig)
    primary = intervals[intervals.primary]
    fig, ax = plt.subplots(figsize=(7.4, 3.0), layout='constrained')
    y = np.arange(len(primary))
    point = primary.delta_precision.to_numpy()
    ax.errorbar(point, y, xerr=np.vstack([point-primary.simultaneous_low, primary.simultaneous_high-point]),
                fmt='o', color='black', capsize=3)
    ax.axvline(0, color='0.6', linewidth=.8)
    ax.set_yticks(y, [f"{'HealthVer' if r.dataset == 'healthver' else 'COVID-Fact'} / " +
                     ('single context' if r.reference == 'best_single' else 'arithmetic blend')
                     for r in primary.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel('Precision difference and simultaneous interval')
    ax.spines[['top', 'right']].set_visible(False)
    for extension in ['pdf', 'svg']:
        fig.savefig(args.output / f'external_primary_intervals.{extension}', bbox_inches='tight')
    plt.close(fig)
    print('External diagnostics drawn from saved results.')


if __name__ == '__main__':
    main()
