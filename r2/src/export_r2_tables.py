"""Export all 26 R2 tables from saved experimental results.

No Word document, manuscript source, or experimental rerun is required.
Run from an extracted release with Python 3.11 and requirements-admission.txt.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import pandas as pd

R2 = Path(__file__).resolve().parents[1]
RES = R2 / 'results'


def evidence(dataset, pool=100):
    root = RES / f'experiments/{dataset}_full_k{pool}'
    return pd.read_csv(root / 'primary_mean_sd.csv').set_index('method')

def selected_single(dataset, pool=100):
    root = RES / f'experiments/{dataset}_full_k{pool}'
    rows = []
    for seed in [13, 17, 29, 7]:
        model = json.loads((root / f'model_seed{seed}.json').read_text())
        data = pd.read_csv(root / 'primary.csv')
        rows.append(data[(data.seed == seed) & (data.method == model['primary_single'])].iloc[0])
    data = pd.DataFrame(rows)
    out = {}
    for col in data.select_dtypes('number'):
        out[col + '_mean'] = data[col].mean()
        out[col + '_std'] = data[col].std(ddof=1)
    return pd.Series(out)

def fs(v, digits=3):
    return '—' if pd.isna(v) else f'{v:.{digits}f}'

def ms(row, col, digits=3, multiline=False):
    sep = '\n' if multiline else ' '
    return fs(row[col + '_mean'], digits) + sep + '(' + fs(row[col + '_std'], digits) + ')'

def main_tables(info):
    t1 = [['SciFact', 'Abstract sentence', 'Annotated evidence; stance groups', 'Original scientific abstract'], ['FEVER', 'Wikipedia sentence', 'Annotated evidence; claim verdict', 'June 2017 snapshot'], ['HoVer', 'Context sentence', 'Annotated multi-hop set; claim verdict', 'Original HotpotQA contexts']]
    t2 = []
    for d, name in [('scifact', 'SciFact'), ('fever', 'FEVER'), ('hover', 'HoVer')]:
        x = info[d]
        t2.append([name, f"{x['claims']:,}", f"{x['rows']:,}", f"{x['candidates_mean']:.1f} ({x['candidates_min']}–{x['candidates_max']})", f"{x['fully_verified']:,} / {x['claims']:,}"])
    t3 = []
    bold = set()
    names = {'scet': 'SCET', 'cal_single': 'Selected single', 'ce': 'Cross-encoder', 'e5': 'E5', 'nli_relevance': 'NLI relevance'}
    for d, name in [('scifact', 'SciFact'), ('fever', 'FEVER'), ('hover', 'HoVer')]:
        dat = evidence(d)
        dat.loc['cal_single'] = selected_single(d)
        sub = dat.loc[list(names)]
        base = len(t3)
        for method, label in names.items():
            r = dat.loc[method]
            t3.append([name, label, ms(r, 'evidence_precision', multiline=True), ms(r, 'annotated_per_100', 1, multiline=True), ms(r, 'volume_per_100', 1, multiline=True), ms(r, 'coverage', multiline=True), ms(r, 'evidence_recall', multiline=True), ms(r, 'stress_precision', multiline=True)])
        for col, j in [('evidence_precision_mean', 2), ('annotated_per_100_mean', 3), ('evidence_recall_mean', 6), ('stress_precision_mean', 7)]:
            high = sub[col].max()
            for k, v in enumerate(sub[col]):
                if np.isclose(v, high, rtol=0, atol=1e-12):
                    bold.add((base + k, j))
    t4 = []
    for d, name in [('scifact', 'SciFact'), ('fever', 'FEVER'), ('hover', 'HoVer')]:
        dat = evidence(d)
        r = dat.loc['scet_risk_control']
        models = [json.loads((RES / f'experiments/{d}_full_k100/model_seed{s}.json').read_text()) for s in [13, 17, 29, 7]]
        feasible = sum((any((t['passes'] and t['admitted'] > 0 for t in m['risk_control']['tests'])) for m in models))
        t4.append([name, f'{feasible}/4', ms(r, 'coverage', multiline=True), ms(r, 'query_error_rate', multiline=True), ms(r, 'volume_per_100', 1, multiline=True), ms(r, 'annotated_per_100', 1, multiline=True)])
    specs = {70: (['Dataset', 'Candidate', 'Evidence target', 'Sentence provenance'], t1, None), 130: (['Dataset', 'Claims', 'Rows', 'Candidates mean (range)', 'Complete evidence queries'], t2, None), 158: (['Dataset', 'Method', 'Precision (SD)', 'Ann./100 (SD)', 'Items/100 (SD)', 'Cov. (SD)', 'Recall (SD)', 'Stress (SD)'], t3, bold), 179: (['Dataset', 'Feasible', 'Coverage (SD)', 'Query error (SD)', 'Items/100 (SD)', 'Ann./100 (SD)'], t4, None)}
    return specs

METHODS = {'bm25': 'BM25', 'tfidf': 'TF-IDF', 'lsa': 'LSA', 'entity': 'Long-token overlap', 'relation': 'Claim-token coverage', 'minilm': 'MiniLM', 'e5': 'E5', 'bge': 'BGE-M3', 'ce': 'Cross-encoder', 'nli_relevance': 'NLI relevance', 'nli_directional': 'NLI directional', 'nli_entailment': 'NLI entailment', 'reliability_mean': 'Reliability mean', 'rag_utility': 'RAG utility', 'retrieval_verification': 'Retrieval-verification', 'conflict_pressure': 'Conflict pressure', 'scet_zero_penalty': 'SCET zero penalty', 'scet': 'SCET', 'scet_risk_control': 'SCET risk control', 'bm25_top1': 'BM25 top 1', 'ce_top1': 'Cross-encoder top 1', 'scet_top1': 'SCET top 1'}

def readable(name):
    if name == 'nli':
        return 'NLI'
    if name.startswith('portfolio_'):
        return 'Logistic portfolio C=' + name.split('_')[-1]
    return METHODS.get(name, name)

def supplementary():
    tables = {}

    def tab(number, title, headers, rows):
        names = {'scifact': 'SciFact', 'fever': 'FEVER', 'hover': 'HoVer'}
        rows = [[names.get(str(row[0]), row[0])] + list(row[1:]) for row in rows]
        if len(headers) == 8:
            headers = [h.replace('Coverage', 'Cov.') for h in headers]
            headers = [{'Items/100': 'Items/\n100', 'Ann./100': 'Ann./\n100', 'Threshold': 'Cutoff'}.get(h, h) for h in headers]
        if number == 7:
            rows = [['Refuted' if x == 'Contradict' else x for x in row] for row in rows]
        tables[f'S{number}'] = (title, headers, rows)
    protocol = json.loads((R2 / 'data/r2_protocol.json').read_text())
    rows = [['Outer split', '30% calibration; 70% evaluation', 'Query-level SHA1 ordering', '13, 17, 29, 7'], ['Calibration allocation', '67% fitting; 33% threshold calibration', 'Disjoint query sets', 'Three inner fitting folds'], ['Workload targets', '25, 50, 100, 200, 300, 400, 500', 'Items per 100 calibration queries', 'Primary target 100'], ['Score transformations', 'Fitting empirical midranks', '1001 quantiles per view', 'Frozen across queries'], ['Eligible rank', '5', 'Fixed identifier-derived random ties', 'Tie seed 73'], ['Conflict penalties', '0, 0.1, 0.25, 0.5', 'Selected on fitting queries', 'Shared stress upper quartile'], ['Logistic regularization', 'C = 0.1, 1, 10', 'Query- and class-balanced weights', 'Inner-fold predictions'], ['Uncertainty', '1000 paired query-cluster draws', 'Mean of four precision differences', 'Bootstrap seed 20261015'], ['Risk testing', 'α = 0.20; δ = 0.05', '20 fixed thresholds; Bonferroni', 'Query-level error event']]
    tab(1, 'Frozen experiment settings', ['Setting', 'Value', 'Procedure', 'Additional detail'], rows)
    rows = []
    for d in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'{d}_full_provenance.csv')
        denominators = dat.groupby(['claim_verdict', 'gold_support']).candidate_rows.sum().to_dict()
        for x in dat.to_dict('records'):
            rows.append([d, x['claim_verdict'].replace('_', ' ').capitalize(), 'Annotated' if x['gold_support'] else 'Other', {'SCIFACT_ORIGINAL_ABSTRACT': 'Original abstract', 'FEVER_JUNE_2017': 'Original 2017 snapshot', 'HOTPOTQA_ORIGINAL_CONTEXT': 'Original context', 'UNVERIFIED_OR_MISSING': 'Unresolved'}.get(x['text_provenance'], x['text_provenance']), f"{x['candidate_rows']:,}\n({100 * x['candidate_rows'] / denominators[x['claim_verdict'], x['gold_support']]:.2f}%)"])
    tab(2, 'Candidate sentence provenance before verification filtering', ['Dataset', 'Verdict', 'Membership', 'Provenance', 'Rows (share)'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        meta = json.loads((R2 / f'data/scored/{ds}_full_k100.json').read_text())
        if ds != 'scifact':
            continue
        for key, m in meta['models'].items():
            revision = '\n'.join((m['revision'][j:j + 12] for j in range(0, len(m['revision']), 12)))
            rows.append([readable(key), m['model'], revision, m['pair_order'], '256'])
    tab(3, 'Exact neural checkpoints', ['View', 'Checkpoint', 'Revision', 'Input order', 'Limit'], rows)
    rows = []
    metrics = ['evidence_precision', 'annotated_per_100', 'volume_per_100', 'coverage', 'evidence_recall', 'stress_precision']
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k100/primary_mean_sd.csv')
        views = ['bm25', 'tfidf', 'lsa', 'entity', 'relation', 'minilm', 'e5', 'bge', 'ce', 'nli_relevance', 'nli_directional', 'nli_entailment', 'reliability_mean', 'rag_utility', 'retrieval_verification', 'portfolio_0.1', 'portfolio_1', 'portfolio_10', 'conflict_pressure', 'scet_zero_penalty', 'scet', 'scet_risk_control', 'bm25_top1', 'ce_top1', 'scet_top1']
        for method in views:
            x = dat[dat.method == method]
            if x.empty:
                continue
            x = x.iloc[0]
            rows.append([ds, readable(method)] + [fs(x[m + '_mean']) + '\n(' + fs(x[m + '_std']) + ')' for m in metrics])
    tab(4, 'All scorer outputs at the common calibration target', ['Dataset', 'Method', 'Precision (SD)', 'Ann./100 (SD)', 'Items/100 (SD)', 'Coverage (SD)', 'Recall (SD)', 'Stress (SD)'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k100/volume_mean_sd.csv')
        dat = dat[dat.method.isin(['scet', 'ce', 'e5', 'bge', 'nli_relevance', 'bm25'])]
        for x in dat.to_dict('records'):
            rows.append([ds, readable(x['method']), str(x['calibration_target']), fs(x['volume_per_100_mean'], 1), fs(x['evidence_precision_mean']), fs(x['evidence_recall_mean']), fs(x['coverage_mean']), fs(x['all_facts_recall_mean'])])
    tab(5, 'Calibration-fixed workload curves', ['Dataset', 'Method', 'Target', 'Items/100', 'Precision', 'Recall', 'Coverage', 'Set recall'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k100/fully_verified_mean_sd.csv')
        for x in dat[dat.method.isin(['scet', 'ce', 'e5', 'bm25', 'nli_relevance'])].to_dict('records'):
            rows.append([ds, readable(x['method']), fs(x['queries_mean'], 0), fs(x['evidence_precision_mean']), fs(x['volume_per_100_mean'], 1), fs(x['evidence_recall_mean']), fs(x['all_facts_recall_mean']), fs(x['stress_precision_mean'])])
    tab(6, 'Queries with complete verified annotated evidence', ['Dataset', 'Method', 'Queries', 'Precision', 'Items/100', 'Recall', 'Set recall', 'Stress'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k100/verdict_mean_sd.csv')
        for x in dat[dat.method.isin(['scet', 'ce', 'e5', 'nli_relevance'])].to_dict('records'):
            rows.append([ds, readable(x['method']), x['claim_verdict'].replace('_', ' ').capitalize(), fs(x['queries_mean'], 0), fs(x['evidence_precision_mean']), fs(x['volume_per_100_mean'], 1), fs(x['coverage_mean']), fs(x['evidence_recall_mean'])])
    tab(7, 'Results by claim verdict', ['Dataset', 'Method', 'Verdict', 'Queries', 'Precision', 'Items/100', 'Coverage', 'Recall'], rows)
    rows = []
    dat = pd.read_csv(RES / 'experiments/scifact_full_k100_directional/primary_mean_sd.csv')
    for x in dat[dat.method.isin(['scet', 'ce', 'e5', 'bge', 'nli_entailment', 'nli_directional', 'nli_relevance'])].to_dict('records'):
        rows.append([readable(x['method']), fs(x['evidence_precision_mean']), fs(x['evidence_precision_std']), fs(x['annotated_per_100_mean'], 1), fs(x['volume_per_100_mean'], 1), fs(x['coverage_mean'])])
    tab(8, 'SciFact directional-support analysis', ['Method', 'Precision', 'SD', 'Support/100', 'Items/100', 'Coverage'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k500/primary_mean_sd.csv')
        for x in dat[dat.method.isin(['scet', 'ce', 'e5', 'bge', 'bm25', 'nli_relevance', 'conflict_pressure'])].to_dict('records'):
            rows.append([ds, readable(x['method']), fs(x['evidence_precision_mean']) + '\n(' + fs(x['evidence_precision_std']) + ')', fs(x['volume_per_100_mean'], 1), fs(x['annotated_per_100_mean'], 1), fs(x['evidence_recall_mean']), fs(x['stress_precision_mean']), fs(x['all_facts_recall_mean'])])
    tab(9, 'Larger candidate-pool audit', ['Dataset', 'Method', 'Precision (SD)', 'Items/100', 'Ann./100', 'Recall', 'Stress', 'Set recall'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        root = RES / f'experiments/{ds}_full_k100'
        dat = pd.read_csv(root / 'paired_bootstrap.csv')
        for x in dat.to_dict('records'):
            rows.append([ds, readable(x['comparator']), fs(x['delta_precision_mean']), f"[{x['ci_low']:.3f}, {x['ci_high']:.3f}]", str(x['unique_query_clusters'])])
        x = json.loads((root / 'primary_comparison.json').read_text())
        rows.append([ds, 'Calibration-selected single', fs(x['delta_precision']), f"[{x['ci_low']:.3f}, {x['ci_high']:.3f}]", 'Shared query clusters'])
    tab(10, 'Paired precision differences from SCET', ['Dataset', 'Comparator', 'Difference', '95% interval', 'Query clusters'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        for seed in [13, 17, 29, 7]:
            x = json.loads((RES / f'experiments/{ds}_full_k100/model_seed{seed}.json').read_text())
            rows.append([ds, str(seed), readable(x['selected_form']), str(x['penalty']), readable(x['primary_single']), fs(x['thresholds']['scet']['threshold'], 6)])
    tab(11, 'Selected forms and frozen thresholds', ['Dataset', 'Seed', 'SCET form', 'Penalty', 'Selected single', 'Threshold'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        root = RES / f'experiments/{ds}_full_k100'
        dat = pd.read_csv(root / 'primary.csv')
        for seed in [13, 17, 29, 7]:
            model = json.loads((root / f'model_seed{seed}.json').read_text())
            x = dat[(dat.seed == seed) & (dat.method == 'scet_risk_control')].iloc[0]
            feasible = model['risk_control']['nonempty_feasible']
            rows.append([ds, str(seed), 'Yes' if feasible else 'Empty rule', fs(model['risk_control']['threshold'], 3) if feasible else '∞', fs(x['query_error_rate']), fs(x['coverage']), fs(x['volume_per_100'], 1), fs(x['annotated_per_100'], 1)])
    tab(12, 'Learn-then-Test query-risk results', ['Dataset', 'Seed', 'Feasible', 'Threshold', 'Query error', 'Coverage', 'Items/100', 'Ann./100'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k100/probability_diagnostic.csv')
        for x in dat.to_dict('records'):
            rows.append([ds, str(x['seed']), fs(x['brier'], 4), fs(x['ece_10_equal_width'], 4), str(x['pairs']), str(x['calibration_pairs'])])
    tab(13, 'Separate monotone probability diagnostic', ['Dataset', 'Seed', 'Brier', 'ECE', 'Test pairs', 'Cal. pairs'], rows)
    dat = pd.read_csv(R2 / 'data/bts_protocol_summary.csv')
    rows = [[{'airport_holdout': 'Airport holdout', 'carrier_holdout': 'Carrier holdout', 'cross_year': 'Cross-year', 'rolling': 'Rolling'}[x['protocol']], f"{x['rows']:,}", f"{x['reviewed']:,}", fs(x['teacher_support_precision'], 4), fs(x['mvec_tg_support_precision'], 4), fs(x['precision_change'], 4)] for x in dat.to_dict('records')]
    tab(14, 'BTS protocol units and precision', ['Protocol', 'Evaluation rows', 'Queue entries', 'Reference', 'Structured', 'Change'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k100/primary_mean_sd.csv')
        for x in dat[dat.method.isin(['scet', 'ce', 'e5', 'bge', 'nli_relevance', 'conflict_pressure', 'scet_zero_penalty'])].to_dict('records'):
            rows.append([ds, readable(x['method']), fs(x['stress_admitted_mean'], 1) + '\n(' + fs(x['stress_admitted_std'], 1) + ')', fs(x['stress_precision_mean']) + '\n(' + fs(x['stress_precision_std']) + ')'])
    tab(15, 'Shared-stress admission counts', ['Dataset', 'Method', 'Admitted (SD)', 'Precision (SD)'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'experiments/{ds}_full_k100/primary_mean_sd.csv')
        for x in dat[dat.method.isin(['scet', 'bm25', 'tfidf', 'lsa', 'minilm', 'e5', 'bge', 'ce', 'nli_relevance', 'nli_directional', 'nli_entailment'])].to_dict('records'):
            rows.append([ds, readable(x['method']), fs(x['ranking_recall_at_5_mean']), fs(x['mrr_mean']), fs(x['ndcg_at_10_mean']), fs(x['all_facts_recall_mean'])])
    tab(16, 'Ranking quality and complete-set retention', ['Dataset', 'Method', 'Recall@5', 'MRR', 'nDCG@10', 'Set recall'], rows)
    for number, pool in [(17, 100), (18, 500)]:
        rows = []
        order = ['scet', 'selected_single', 'bm25', 'tfidf', 'lsa', 'entity', 'relation', 'minilm', 'e5', 'bge', 'ce', 'nli_relevance', 'nli_directional', 'nli_entailment', 'conflict_pressure']
        for ds in ['scifact', 'fever', 'hover']:
            dat = pd.read_csv(RES / f'strengthening/{ds}_full_k{pool}/exact_budget_mean_sd.csv').set_index('method')
            for method in order:
                x = dat.loc[method]
                rows.append([ds, 'Selected single' if method == 'selected_single' else readable(method), fs(x.evidence_precision_mean) + '\n(' + fs(x.evidence_precision_std) + ')', fs(x.coverage_mean), fs(x.evidence_recall_mean), fs(x.all_facts_recall_mean)])
            if ds == 'scifact' and pool == 100:
                extra = pd.read_csv(RES / 'strengthening/multivers/exact_budget.csv')
                rows.append([ds, 'MultiVerS context', fs(extra.evidence_precision.mean()) + '\n(' + fs(extra.evidence_precision.std(ddof=1)) + ')', fs(extra.coverage.mean()), fs(extra.evidence_recall.mean()), fs(extra.all_facts_recall.mean())])
        tab(number, f'Fixed-batch comparison with a {pool}-candidate cap', ['Dataset', 'Method', 'Precision (SD)', 'Coverage', 'Recall', 'Set recall'], rows)
    rows = []
    mv = pd.read_csv(RES / 'strengthening/multivers/primary.csv')
    primary = pd.read_csv(RES / 'experiments/scifact_full_k100/primary.csv')
    for name, dat, context in [('SCET', primary[primary.method == 'scet'], 'Title and sentence'), ('Cross-encoder', primary[primary.method == 'ce'], 'Title and sentence'), ('MultiVerS', mv, 'Title and full abstract')]:
        rows.append([name, fs(dat.evidence_precision.mean()) + '\n(' + fs(dat.evidence_precision.std(ddof=1)) + ')', fs(dat.volume_per_100.mean(), 1), fs(dat.evidence_recall.mean()), fs(dat.all_facts_recall.mean()), context])
    tab(19, 'SciFact document-context comparison', ['Method', 'Precision (SD)', 'Items/100', 'Recall', 'Set recall', 'Input context'], rows)
    interval = json.loads((RES / 'strengthening/multivers/paired_interval.json').read_text())
    batch_interval = json.loads((RES / 'strengthening/multivers/exact_budget_interval.json').read_text())
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        dat = pd.read_csv(RES / f'strengthening/{ds}_full_k100/probability_reference.csv')
        rows.append([ds, fs(dat.test_prevalence.mean(), 4), fs(dat.mapped_brier.mean(), 4), fs(dat.prior_brier.mean(), 4), fs(dat.mapped_ece.mean(), 4), fs(dat.prior_ece.mean(), 4)])
    tab(20, 'Probability diagnostics with a prevalence reference', ['Dataset', 'Prevalence', 'Mapped Brier', 'Prior Brier', 'Mapped ECE', 'Prior ECE'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        for pool in [100, 500]:
            dat = pd.read_csv(RES / f'strengthening/{ds}_full_k{pool}/retention_decomposition.csv')
            for method in ['scet', 'selected_single', 'ce']:
                x = dat[dat.method == method].mean(numeric_only=True)
                rows.append([ds, str(pool), 'Selected single' if method == 'selected_single' else readable(method), fs(x.pool_set_recall), fs(x.eligible_set_recall), fs(x.admitted_set_recall)])
    tab(21, 'Complete-set recall through admission stages', ['Dataset', 'Pool cap', 'Method', 'Available', 'Eligible', 'Admitted'], rows)
    rows = []
    for ds in ['scifact', 'fever', 'hover']:
        for x in pd.read_csv(RES / f'strengthening/{ds}_full_k100/risk_bound.csv').to_dict('records'):
            rows.append([ds, str(x['seed']), str(x['calibration_queries']), str(x['error_queries']), 'Nonempty' if x['nonempty_feasible'] else 'Empty rule', fs(x['simultaneous_upper_bound'], 3)])
    tab(22, 'Calibration-side query-risk bounds', ['Dataset', 'Seed', 'Queries', 'Errors', 'Rule', 'Upper bound'], rows)
    return tables


def export(smoke=False):
    destination = RES / ('tables_smoke' if smoke else 'tables')
    destination.mkdir(parents=True, exist_ok=True)
    specs = main_tables(json.loads((RES / 'dataset_facts.json').read_text(encoding='utf-8')))
    tables = {f'Table_{i}': (f'Manuscript Table {i}', headers, rows)
              for i, (headers, rows, bold) in enumerate(specs.values(), 1)}
    tables.update(supplementary())
    assert len(tables) == 26
    if smoke:
        tables = {key: tables[key] for key in ['Table_1', 'S1']}
    index = []
    for name, (title, headers, rows) in tables.items():
        assert all(len(row) == len(headers) for row in rows), name
        path = destination / (name + '.csv')
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerows([headers] + rows)
        index.append({'table': name, 'title': title, 'file': path.name,
                      'rows': len(rows), 'columns': len(headers),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print(f'Exported {len(index)} tables to results/{destination.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Write two representative tables first')
    export(parser.parse_args().smoke)

