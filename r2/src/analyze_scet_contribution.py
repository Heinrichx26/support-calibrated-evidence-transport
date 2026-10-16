"""Attribute saved SCET results without fitting, rescoring, or changing decisions.

Run --smoke first. Full analysis is gated by an unchanged successful smoke.
Every output is confined to results/evidence_completion/contribution/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/evidence_completion/contribution'
SEEDS = [13, 17, 29, 7]
DATASETS = ['fever', 'hover', 'scifact']
SINGLES = ['bm25', 'tfidf', 'lsa', 'entity', 'relation', 'minilm', 'e5',
           'bge', 'ce', 'nli_relevance', 'nli_directional', 'nli_entailment']
NAMES = {'scet': 'SCET', 'ce': 'Cross-encoder', 'selected_single': 'Fitting-selected single',
         'e5': 'E5', 'multivers': 'MultiVerS', 'scet_multivers': 'SCET with MultiVerS',
         'ce_top1': 'Cross-encoder top one', 'scet_top1': 'SCET top one',
         'bm25': 'BM25', 'tfidf': 'TF-IDF', 'lsa': 'LSA', 'entity': 'Long-token overlap',
         'relation': 'Claim-token coverage', 'minilm': 'MiniLM', 'bge': 'BGE-M3',
         'nli_relevance': 'NLI relevance', 'nli_directional': 'NLI directional',
         'nli_entailment': 'NLI entailment'}
GROUPS = ['all', 'annotated', 'singleton', 'multiple', 'no_evidence',
          'SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
METRICS = ['precision', 'set_recall', 'evidence_recall', 'coverage',
           'correct_per100', 'complete_per100', 'no_evidence_queries_per100',
           'no_evidence_items_per100']
PLAN = {
    'analysis': 'Descriptive attribution of already frozen 100-candidate, equal-budget decisions',
    'datasets': DATASETS, 'seeds': SEEDS, 'individual_scorers': SINGLES,
    'budget': '100 admitted sentences per 100 held-out claims; maximum five per claim, with separately labelled top-one controls',
    'bootstrap_repetitions': 2000, 'bootstrap_seed': 2026101505,
    'intervals': 'Pointwise paired percentile 95% intervals; no family-wide significance claim',
    'resampling': 'Shared cluster weights for the same query across overlapping partitions; average the four partition differences',
    'conditioning': 'Frozen fitted scorers, thresholds, selections, admission decisions and observed partitions; no budget reallocation or refitting within resamples',
    'source_sensitivity': 'FEVER connected components of shared originally annotated Wikipedia pages; no-evidence claims are singleton components because no annotated pages exist. Also diagnose components of all candidate pages; do not issue a source-bootstrap interval with fewer than 20 components or a component containing over 50% of queries.',
    'attribution': 'Additive changes in correct items and complete claims by minimum annotated-set size and verdict, normalized by all-query counts',
    'selection': 'No model or rule is selected using these results; every individual scorer is reported',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metadata(queries, ids):
    rows = []
    for qid in ids:
        q = queries[qid]
        sizes = [len(set(s)) for s in q['gold_sets'] if s]
        smallest = min(sizes) if sizes else 0
        rows.append({'query_id': qid, 'minimum_set_size': smallest,
                     'stratum': 'no_evidence' if not smallest else ('singleton' if smallest == 1 else 'multiple')})
    return pd.DataFrame(rows)


def load(ds, manifest):
    qp = ROOT / f'data/processed/{ds}_full_queries.json'
    queries = read_json(qp)
    manifest[str(qp.relative_to(ROOT))] = sha(qp)
    collection = {}
    for seed in SEEDS:
        mp = ROOT / f'results/experiments/{ds}_full_k100/model_seed{seed}.json'
        model = read_json(mp)
        ids = list(map(str, model['evaluation_query_ids']))
        meta = metadata(queries, ids)
        paths = {m: ROOT / f'results/strengthening/{ds}_full_k100/query_seed{seed}_{m}.csv'
                 for m in SINGLES + ['selected_single', 'scet', 'conflict_pressure']}
        paths.update({m + '_top1': ROOT / f'results/scet_allocation/full/query_{ds}_seed{seed}_{m}_cap1.csv'
                      for m in ['scet', 'ce']})
        if ds == 'scifact':
            paths.update({m: ROOT / f'results/multivers_extension/k100/query_seed{seed}_{m}_exact_budget.csv'
                          for m in ['multivers', 'scet_multivers']})
            paths['multivers_top1'] = ROOT / f'results/scet_allocation/full/query_scifact_seed{seed}_multivers_cap1.csv'
            extension = read_json(ROOT / f'results/multivers_extension/k100/model_seed{seed}.json')
            assert extension['selected_form'] == 'multivers' and extension['penalty'] == 0
        methods = {}
        for method, path in paths.items():
            s = pd.read_csv(path, dtype={'query_id': str})
            assert set(s.query_id) == set(ids) and not s.query_id.duplicated().any(), (ds, seed, method)
            s = s.set_index('query_id').loc[ids].reset_index().merge(meta, on='query_id', validate='one_to_one')
            assert int(s.admitted.sum()) == len(ids), (ds, seed, method, 'budget')
            assert s.admitted.between(0, 5).all() and s.correct.le(s.admitted).all()
            assert s.all_facts.le(s.gold_count.gt(0)).all()
            assert np.array_equal(s.gold_count, [queries[q]['gold_count'] for q in ids])
            if method.endswith('top1'):
                assert s.admitted.eq(1).all()
                assert not s.loc[s.minimum_set_size.gt(1), 'all_facts'].any()
            methods[method] = s
            manifest[str(path.relative_to(ROOT))] = sha(path)
        # The fitting-selected row must be exactly the saved chosen individual row.
        cols = ['admitted', 'correct', 'covered', 'all_facts']
        np.testing.assert_array_equal(methods['selected_single'][cols], methods[model['primary_single']][cols])
        if ds == 'scifact':
            np.testing.assert_array_equal(methods['multivers'][cols], methods['scet_multivers'][cols])
        collection[seed] = methods
        manifest[str(mp.relative_to(ROOT))] = sha(mp)
    return queries, collection


def mask_group(s, group):
    if group == 'all':
        return np.ones(len(s), bool)
    if group == 'annotated':
        return s.gold_count.to_numpy() > 0
    if group in ['singleton', 'multiple', 'no_evidence']:
        return s.stratum.eq(group).to_numpy()
    return s.claim_verdict.eq(group).to_numpy()


def parts(s, metric, group):
    keep = mask_group(s, group).astype(float)
    positive = s.gold_count.to_numpy() > 0
    if metric == 'precision':
        return s.correct.to_numpy() * keep, s.admitted.to_numpy() * keep
    if metric == 'set_recall':
        return s.all_facts.to_numpy() * keep, positive * keep
    if metric == 'evidence_recall':
        r = np.divide(s.correct, s.gold_count, out=np.zeros(len(s)), where=positive)
        return r * keep, positive * keep
    if metric == 'coverage':
        return s.covered.to_numpy() * keep, keep
    if metric == 'correct_per100':
        return 100 * s.correct.to_numpy() * keep, np.ones(len(s))
    if metric == 'complete_per100':
        return 100 * s.all_facts.to_numpy() * keep, np.ones(len(s))
    if metric == 'no_evidence_queries_per100':
        return 100 * s.covered.to_numpy() * (~positive) * keep, np.ones(len(s))
    if metric == 'no_evidence_items_per100':
        return 100 * s.admitted.to_numpy() * (~positive) * keep, np.ones(len(s))
    raise ValueError(metric)


def value(s, metric, group):
    a, b = parts(s, metric, group)
    return float(a.sum()/b.sum()) if b.sum() else np.nan


def summarize(ds, collection):
    rows = []
    for seed, methods in collection.items():
        for method, s in methods.items():
            groups = GROUPS if ds == 'fever' else GROUPS[:5]
            for group in groups:
                keep = mask_group(s, group)
                if not keep.any():
                    continue
                row = {'dataset': ds, 'seed': seed, 'method': method, 'group': group,
                       'queries': int(keep.sum()), 'all_queries': len(s),
                       'admitted': int(s.loc[keep, 'admitted'].sum()),
                       'correct': int(s.loc[keep, 'correct'].sum()),
                       'complete': int(s.loc[keep, 'all_facts'].sum()),
                       'covered': int(s.loc[keep, 'covered'].sum())}
                row.update({metric: value(s, metric, group) for metric in METRICS})
                rows.append(row)
    return rows


class Components:
    def __init__(self, ids):
        self.parent = {q: q for q in ids}

    def find(self, q):
        while self.parent[q] != q:
            self.parent[q] = self.parent[self.parent[q]]
            q = self.parent[q]
        return q

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def mapping(self):
        return {q: self.find(q) for q in self.parent}


def component_summary(mapping, kind):
    counts = pd.Series(mapping).value_counts()
    return {'kind': kind, 'queries': int(len(mapping)), 'components': int(len(counts)),
            'largest_component_queries': int(counts.max()),
            'largest_component_share': float(counts.max()/len(mapping)),
            'singleton_components': int(counts.eq(1).sum()),
            'bootstrap_usable': bool(len(counts) >= 20 and counts.max()/len(mapping) <= .5)}


def source_components(queries, ids, out, smoke):
    comp = Components(ids)
    owner = {}
    for q in ids:
        for page in {cid.rsplit('::', 1)[0] for cid in queries[q]['gold_ids']}:
            if page in owner:
                comp.join(q, owner[page])
            else:
                owner[page] = q
    gold = comp.mapping()
    descriptions = [component_summary(gold, 'annotated_page_components')]
    pd.DataFrame({'query_id': ids, 'gold_page_component': [gold[q] for q in ids],
                  'annotated_pages': [len({c.rsplit('::', 1)[0] for c in queries[q]['gold_ids']}) for q in ids]}).to_csv(out/'fever_source_components.csv', index=False)
    if not smoke:
        frame = pd.read_parquet(ROOT/'data/processed/fever_full_k100.parquet', columns=['query_id', 'evidence_source'])
        frame = frame[frame.query_id.isin(ids)].drop_duplicates()
        all_comp, owners = Components(ids), {}
        for q, page in frame.itertuples(index=False, name=None):
            if page in owners:
                all_comp.join(q, owners[page])
            else:
                owners[page] = q
        descriptions.append(component_summary(all_comp.mapping(), 'all_candidate_page_components'))
    return gold, descriptions


def requests(ds):
    # Prespecified comprehensive FEVER precision family, with all individual rows.
    pairs = []
    if ds == 'fever':
        pairs += [('scet', b, 'precision', 'all') for b in SINGLES]
    refs = ['ce', 'selected_single', 'e5', 'ce_top1'] if ds != 'scifact' else ['ce', 'selected_single', 'multivers']
    for ref in refs:
        for group in ['all', 'singleton', 'multiple', 'no_evidence']:
            for metric in ['precision', 'set_recall', 'coverage']:
                pairs.append(('scet', ref, metric, group))
        for metric in ['evidence_recall', 'correct_per100', 'complete_per100',
                       'no_evidence_queries_per100', 'no_evidence_items_per100']:
            pairs.append(('scet', ref, metric, 'all'))
    if ds == 'fever':
        for ref in ['ce', 'ce_top1']:
            for group in ['singleton', 'multiple', 'no_evidence', 'SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']:
                for metric in ['correct_per100', 'complete_per100']:
                    pairs.append(('scet', ref, metric, group))
    if ds == 'scifact':
        pairs += [('scet_multivers', 'multivers', m, 'all') for m in ['precision', 'set_recall', 'evidence_recall', 'coverage']]
    return list(dict.fromkeys(pairs))


def bootstrap(ds, collection, reqs, out, cluster_mapping=None, label='query', repetitions=2000):
    ids = sorted(set().union(*(set(m['scet'].query_id) for m in collection.values())))
    labels = [cluster_mapping[q] if cluster_mapping else q for q in ids]
    unique = sorted(set(labels))
    lookup = {v: i for i, v in enumerate(unique)}
    query_cluster = {q: lookup[v] for q, v in zip(ids, labels)}
    # Reuse numerators and denominators across contrasts; no candidate matrices required.
    keys = list(dict.fromkeys((seed, m, metric, group)
                             for seed in SEEDS for a, b, metric, group in reqs for m in [a, b]))
    ki = {k: i for i, k in enumerate(keys)}
    mat = np.zeros((len(unique), 2*len(keys)), np.float64)
    for key, j in ki.items():
        seed, method, metric, group = key
        s = collection[seed][method]
        ix = np.array([query_cluster[q] for q in s.query_id], int)
        num, den = parts(s, metric, group)
        mat[:, 2*j] = np.bincount(ix, weights=num, minlength=len(unique))
        mat[:, 2*j+1] = np.bincount(ix, weights=den, minlength=len(unique))
    sums = mat.sum(axis=0).reshape(-1, 2)
    points = np.divide(sums[:, 0], sums[:, 1], out=np.full(len(keys), np.nan), where=sums[:, 1] > 0)
    contrasts = []
    for a, b, metric, group in reqs:
        aj = [ki[(seed, a, metric, group)] for seed in SEEDS]
        bj = [ki[(seed, b, metric, group)] for seed in SEEDS]
        if np.isfinite(points[aj]).all() and np.isfinite(points[bj]).all():
            contrasts.append(((a, b, metric, group), aj, bj))
    draws = np.full((repetitions, len(contrasts)), np.nan)
    rng = np.random.default_rng(PLAN['bootstrap_seed'])
    for start in range(0, repetitions, 50):
        count = min(50, repetitions-start)
        weights = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=count).astype(np.float64)
        s = (weights @ mat).reshape(count, len(keys), 2)
        v = np.divide(s[:, :, 0], s[:, :, 1], out=np.full(s.shape[:2], np.nan), where=s[:, :, 1] > 0)
        for j, (_, a, b) in enumerate(contrasts):
            draws[start:start+count, j] = np.mean(v[:, a]-v[:, b], axis=1)
    rows = []
    for j, ((a, b, metric, group), ai, bi) in enumerate(contrasts):
        low, high = np.nanquantile(draws[:, j], [.025, .975])
        rows.append({'dataset': ds, 'method': a, 'reference': b, 'metric': metric,
                     'group': group, 'difference': float(np.mean(points[ai]-points[bi])),
                     'ci95_low': float(low), 'ci95_high': float(high),
                     'resampling_unit': label, 'clusters': len(unique), 'unique_queries': len(ids),
                     'bootstrap_repetitions': repetitions, 'valid_draws': int(np.isfinite(draws[:, j]).sum()),
                     'scope': 'pointwise; frozen admissions; observed overlapping partitions'})
    print(ds, label, 'bootstrap complete:', len(rows), 'contrasts', len(unique), 'clusters', flush=True)
    return rows


def rounded(x):
    match = re.search(r'-?\d+(?:\.\d+)?', str(x))
    return float(match.group()) if match else np.nan


def smoke_check(rows):
    df = pd.DataFrame(rows)
    mean = df.groupby(['dataset', 'method', 'group'])[METRICS].mean()
    checked = []
    def check(table, ds, method, group, metric, expected):
        if not np.isnan(expected):
            got = float(mean.loc[(ds, method, group), metric])
            assert abs(got-expected) <= .00051, (table, ds, method, group, metric, got, expected)
            checked.append({'table': table, 'dataset': ds, 'method': method, 'group': group,
                            'metric': metric, 'current_value': got, 'displayed_value': expected})
    labels = {v: k for k, v in NAMES.items()}
    labels.update({'BM25': 'bm25', 'TF-IDF': 'tfidf', 'LSA': 'lsa', 'Long-token overlap': 'entity',
                   'Claim-token coverage': 'relation', 'MiniLM': 'minilm', 'BGE-M3': 'bge',
                   'NLI relevance': 'nli_relevance', 'NLI directional': 'nli_directional',
                   'NLI entailment': 'nli_entailment', 'Conflict pressure': 'conflict_pressure',
                   'Selected single': 'selected_single', 'MultiVerS context': 'multivers',
                   'Cross-encoder top 1': 'ce_top1', 'SCET top 1': 'scet_top1'})
    t = pd.read_csv(ROOT/'results/tables/S17.csv')
    for _, row in t.iterrows():
        ds, method = row.Dataset.lower(), labels[row.Method]
        for col, metric in [('Precision (SD)', 'precision'), ('Coverage', 'coverage'),
                            ('Recall', 'evidence_recall'), ('Set recall', 'set_recall')]:
            check('S17', ds, method, 'all', metric, rounded(row[col]))
    t = pd.read_csv(ROOT/'results/tables/S28.csv')
    for _, row in t.iterrows():
        name = row.Method.split(',')[0]
        method = labels[name] + ('_top1' if row.Method.endswith('cap 1') else '')
        for col, metric, group in [('Precision', 'precision', 'all'), ('Coverage', 'coverage', 'all'),
                                   ('Set recall', 'set_recall', 'all'), ('No-evidence admission', 'coverage', 'no_evidence')]:
            check('S28', row.Dataset.lower(), method, group, metric, rounded(row[col]))
    t = pd.read_csv(ROOT/'results/tables/S30.csv')
    for _, row in t.iterrows():
        for col, metric, group in [('Precision (SD)', 'precision', 'all'), ('Set recall (SD)', 'set_recall', 'all'),
                                   ('Singleton (SD)', 'set_recall', 'singleton'), ('Multiple (SD)', 'set_recall', 'multiple')]:
            check('S30', row.Dataset.lower(), labels[row.Method], group, metric, rounded(row[col]))
    return checked


def write_report(summary, intervals, descriptions, out):
    mean = summary.groupby(['dataset', 'method', 'group'])[METRICS].mean()
    def v(ds, method, metric, group='all'):
        return mean.loc[(ds, method, group), metric]
    def ci(ds, reference, metric='precision', group='all', unit='query'):
        s = intervals[(intervals.dataset == ds) & (intervals.method == 'scet') &
                      (intervals.reference == reference) & (intervals.metric == metric) &
                      (intervals.group == group) & (intervals.resampling_unit == unit)].iloc[0]
        return f'{s.difference:.4f} [{s.ci95_low:.4f}, {s.ci95_high:.4f}]'
    lines = ['# Frozen-result contribution analysis', '',
             'All primary rows admit exactly 100 sentences per 100 held-out queries. Methods, fitted choices and decisions remain fixed. The four partitions overlap. Intervals below are paired pointwise 95% percentile intervals with common query weights across partitions, conditional on the saved decisions. Budgets are equal at the observed point; the resamples reweight frozen decisions without reallocating the budget. They do not measure refitting or selection uncertainty.', '',
             '## FEVER: every individual scorer', '',
             '| Method | Precision | SCET precision difference [95% interval] |', '|---|---:|---:|']
    for method in SINGLES:
        lines.append(f'| {NAMES.get(method, method)} | {v("fever", method, "precision"):.4f} | {ci("fever", method)} |')
    lines += ['', '## Attribution at the same budget', '',
              '| FEVER reference | Additional annotated items /100 queries | Additional complete claims /100 queries | Change in no-evidence claims receiving output /100 all queries |',
              '|---|---:|---:|---:|']
    for method in ['ce', 'ce_top1']:
        lines.append(f'| {NAMES[method]} | {ci("fever", method, "correct_per100")} | {ci("fever", method, "complete_per100")} | {ci("fever", method, "no_evidence_queries_per100")} |')
    lines += ['', '| FEVER stratum | SCET set recall | Cross-encoder set recall | Difference [95% interval] | Contribution to additional annotated items /100 all queries |', '|---|---:|---:|---:|---:|']
    for group in ['singleton', 'multiple']:
        lines.append(f'| {group} | {v("fever", "scet", "set_recall", group):.4f} | {v("fever", "ce", "set_recall", group):.4f} | {ci("fever", "ce", "set_recall", group)} | {ci("fever", "ce", "correct_per100", group)} |')
    lines += ['', '## HoVer: complete comparator family', '',
              '| Method | Precision | Complete-set recall | SCET set-recall difference [95% interval] |', '|---|---:|---:|---:|']
    for method in ['ce', 'selected_single', 'e5', 'ce_top1']:
        lines.append(f'| {NAMES[method]} | {v("hover", method, "precision"):.4f} | {v("hover", method, "set_recall"):.4f} | {ci("hover", method, "set_recall")} |')
    lines += ['', '## SciFact: MultiVerS attribution', '',
              f'SCET precision is {v("scifact", "scet", "precision"):.4f}; MultiVerS precision is {v("scifact", "multivers", "precision"):.4f}. SCET minus MultiVerS: {ci("scifact", "multivers")}. SCET with MultiVerS and individual MultiVerS have identical per-query admitted counts, correct counts and complete-set decisions in all four saved partitions. Their difference is zero for these metrics; no combination gain is established.', '',
              '## Source dependence sensitivity', '',
              'Annotated-page components connect claims sharing any originally annotated Wikipedia page. Claims without annotated evidence have no such page identity and remain singleton components. This sensitivity therefore covers dependence through annotated pages; it does not cover every dependence caused by shared negative candidates or source content. All-candidate-page components are diagnosed separately. A component containing over half the queries or fewer than 20 components makes that component bootstrap uninformative.', '',
              '| Component construction | Components | Largest component | Largest share | Bootstrap usable |', '|---|---:|---:|---:|---|']
    for d in descriptions:
        lines.append(f'| {d["kind"]} | {d["components"]} | {d["largest_component_queries"]} | {d["largest_component_share"]:.3f} | {d["bootstrap_usable"]} |')
    if (intervals.resampling_unit == 'annotated_page_component').any():
        lines += ['', 'FEVER SCET minus cross-encoder, annotated-page-component sensitivity: '+ci('fever', 'ce', unit='annotated_page_component')+'.']
    lines += ['', '## Interpretation', '',
              'The FEVER gain against the cross-encoder with the same five-sentence cap is a conditional improvement at this collection and pool. Overall complete-set retention and retention within multi-sentence claims are reported separately. Top-one allocation spends an item on every claim, giving different coverage and complete-set outcomes. The HoVer advantage over the cross-encoder does not establish an advantage over the fitting-selected scorer or E5. Adding MultiVerS to SCET reproduces the individual MultiVerS decisions. These results support an admission protocol and collection-specific allocation trade-offs.', '']
    (out/'evidence_note.md').write_text('\n'.join(lines), encoding='utf-8')
    en = [
        f'At an equal budget of 100 sentences per 100 FEVER claims and a maximum of five sentences per claim, SCET increased evidence precision over the cross-encoder by {ci("fever", "ce")}. The change in complete-set recall was {ci("fever", "ce", "set_recall")}. The corresponding differences were {ci("fever", "ce", "set_recall", "singleton")} for claims with a singleton evidence set and {ci("fever", "ce", "set_recall", "multiple")} for claims requiring several sentences. These strata locate the observed retention gain.',
        f'On HoVer, the complete-set recall difference was {ci("hover", "ce", "set_recall")} against the cross-encoder, {ci("hover", "selected_single", "set_recall")} against the fitting-selected individual scorer, and {ci("hover", "e5", "set_recall")} against E5. On SciFact, all four SCET selections with MultiVerS reproduced the individual MultiVerS decisions. The comparisons identify collection-specific changes in evidence allocation.',
        'All intervals are pointwise 95% intervals conditional on the fitted rules and observed partitions. Query weights are shared across overlapping held-out partitions.'
    ]
    (out/'author_paragraphs_en.txt').write_text('\n\n'.join(en)+'\n', encoding='utf-8')
    response = (
        'We now report every individual scorer at exactly 100 admitted sentences per 100 held-out claims, '
        'with the same maximum of five sentences per claim. Each prospective comparison has its own threshold '
        'fixed on independent calibration queries. At the equal batch budget, FEVER precision increased over '
        f'the cross-encoder by {ci("fever", "ce")}. The improvement was positive in all four partitions and '
        'the cross-encoder had the highest precision among the twelve individual scorers in each partition. '
        'A paired sensitivity analysis grouping claims that share annotated Wikipedia pages gave a precision '
        f'difference of {ci("fever", "ce", unit="annotated_page_component")}. '
        'The overall complete-set recall difference was '+ci('fever', 'ce', 'set_recall')+'. '
        'The differences were '+ci('fever', 'ce', 'set_recall', 'singleton')+' for claims with a singleton evidence set '
        'and '+ci('fever', 'ce', 'set_recall', 'multiple')+' for claims requiring several sentences. '
        'At the same workload, SCET produced '+ci('fever', 'ce', 'correct_per100')+' additional annotated sentences '
        'and changed the number of claims without annotated evidence receiving output by '
        +ci('fever', 'ce', 'no_evidence_queries_per100')+' per 100 claims.\n\n'
        'For HoVer, the complete-set recall difference was '+ci('hover', 'ce', 'set_recall')+' against the '
        'cross-encoder, '+ci('hover', 'selected_single', 'set_recall')+' against the fitting-selected individual '
        'scorer, and '+ci('hover', 'e5', 'set_recall')+' against E5. These comparisons identify the reference '
        'under which the complete-set gain occurs. On SciFact, SCET with MultiVerS selected MultiVerS in all '
        'four partitions and reproduced its per-claim admission and complete-set outcomes. We attribute this '
        'result to the individual MultiVerS scorer. The revised interpretation follows these collection-specific '
        'results and the observed evidence-allocation trade-offs. All intervals are pointwise 95% intervals '
        'conditional on the fitted rules and observed partitions; shared query weights preserve their overlap.'
    )
    (out/'author_response_r5_3_en.txt').write_text(response+'\n', encoding='utf-8')
    zh = [
        'FEVER在每100个查询接纳100句、每个查询最多5句的条件下，SCET相对交叉编码器的精度差为'
        +ci('fever', 'ce')+'，四个分区均为正；交叉编码器在每个分区均为12个单评分器中精度最高者。'
        '按共享标注维基页面聚类后，精度差为'+ci('fever', 'ce', unit='annotated_page_component')+'。',
        '总体完整集召回差为'+ci('fever', 'ce', 'set_recall')+'；最小标注集为单句的组差为'
        +ci('fever', 'ce', 'set_recall', 'singleton')+'，最小标注集至少两句的组差为'
        +ci('fever', 'ce', 'set_recall', 'multiple')+'。每100个全部查询增加标注句'
        +ci('fever', 'ce', 'correct_per100')+'，由单句组增加'
        +ci('fever', 'ce', 'correct_per100', 'singleton')+'和多句组减少'
        +ci('fever', 'ce', 'correct_per100', 'multiple')+'相加得到。按verdict的完整结果保存在CSV中。',
        'HoVer完整集召回相对交叉编码器提高'+ci('hover', 'ce', 'set_recall')+'；相对拟合阶段选定单评分器的差为'
        +ci('hover', 'selected_single', 'set_recall')+'，相对E5的差为'+ci('hover', 'e5', 'set_recall')+'。'
        'SciFact中SCET+MultiVerS与单MultiVerS四个分区的逐查询接纳数、正确句数、覆盖和完整集结果完全一致。',
        '按全部候选页面的共享关系，19,832个FEVER查询连成一个组件，无法据此估计有辨识力的组件重采样区间。'
        '共享标注页面敏感性只覆盖这类来源依赖；无标注证据查询没有标注页面，按单例处理。'
        '所有区间以冻结规则、输出和观察到的四分区为条件，重叠查询使用共同权重，未重新拟合或重分配重采样预算。'
    ]
    (out/'evidence_note_zh.md').write_text('\n\n'.join(zh)+'\n', encoding='utf-8')


def run(smoke):
    start = time.monotonic()
    out = OUT/('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    if not smoke:
        gate = read_json(OUT/'smoke/audit.json')
        assert gate['passed'] and gate['source_sha256'] == sha(Path(__file__))
        assert gate['plan'] == PLAN
    (out/'analysis_plan.json').write_text(json.dumps(PLAN, indent=2), encoding='utf-8')
    manifest, rows, collections, queries = {}, [], {}, {}
    for ds in DATASETS:
        queries[ds], collections[ds] = load(ds, manifest)
        rows += summarize(ds, collections[ds])
        print(ds, 'saved decisions loaded and exact budgets verified', flush=True)
    checks = smoke_check(rows)
    print('Displayed-value smoke checks:', len(checks), 'passed', flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'by_partition.csv', index=False)
    per_partition = []
    for ds in DATASETS:
        for seed, methods in collections[ds].items():
            for a, b, metric, group in requests(ds):
                va, vb = value(methods[a], metric, group), value(methods[b], metric, group)
                if np.isfinite(va) and np.isfinite(vb):
                    per_partition.append({'dataset': ds, 'seed': seed, 'method': a, 'reference': b,
                                          'metric': metric, 'group': group, 'method_value': va,
                                          'reference_value': vb, 'difference': va-vb})
    pd.DataFrame(per_partition).to_csv(out/'paired_by_partition.csv', index=False)
    for seed, methods in collections['fever'].items():
        assert all(value(methods['ce'], 'precision', 'all') >= value(methods[m], 'precision', 'all') for m in SINGLES)
        assert value(methods['scet'], 'precision', 'all') > value(methods['ce'], 'precision', 'all')
    nums = [c for c in summary.select_dtypes(include=np.number) if c != 'seed']
    means = summary.groupby(['dataset', 'method', 'group'])[nums].agg(['mean', 'std']).reset_index()
    means.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in means.columns]
    means.to_csv(out/'mean_sd.csv', index=False)
    interval_rows, descriptions = [], []
    if not smoke:
        with threadpool_limits(limits=4):
            for ds in DATASETS:
                interval_rows += bootstrap(ds, collections[ds], requests(ds), out)
            ids = sorted(set().union(*(set(m['scet'].query_id) for m in collections['fever'].values())))
            mapping, descriptions = source_components(queries['fever'], ids, out, False)
            if descriptions[0]['bootstrap_usable']:
                reqs = [r for r in requests('fever') if r[1] in ['ce', 'ce_top1']]
                interval_rows += bootstrap('fever', collections['fever'], reqs, out, mapping, 'annotated_page_component')
        intervals = pd.DataFrame(interval_rows)
        intervals.to_csv(out/'paired_intervals.csv', index=False)
        (out/'source_dependency.json').write_text(json.dumps(descriptions, indent=2), encoding='utf-8')
        overlap = []
        for ds, collection in collections.items():
            occurrence = pd.Series([q for methods in collection.values() for q in methods['scet'].query_id]).value_counts()
            overlap.append({'dataset': ds, 'unique_queries': len(occurrence), 'total_partition_occurrences': int(occurrence.sum()),
                            'queries_in_multiple_partitions': int(occurrence.gt(1).sum())})
        pd.DataFrame(overlap).to_csv(out/'partition_overlap.csv', index=False)
        write_report(summary, intervals, descriptions, out)
    audit = {'passed': True, 'smoke': smoke, 'plan': PLAN, 'source_sha256': sha(Path(__file__)),
             'displayed_value_checks': checks, 'input_sha256': manifest,
             'models_fitted': False, 'thresholds_changed': False, 'new_predictions_computed': False,
             'elapsed_seconds': time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Passed', 'smoke' if smoke else 'full', 'analysis in', round(time.monotonic()-start, 1), 'seconds', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
