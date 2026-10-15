"""Descriptive evidence-set-size analysis of frozen, equal-budget decisions."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import r2_admission as admission
from scet_set_size_tables import gold_set_table_specs

R2 = Path(__file__).resolve().parents[1]
OUT = R2 / 'results/substantive_strengthening/set_size'
DATASETS = ['scifact', 'fever', 'hover']
SEEDS = [13, 17, 29, 7]
METHODS = {
    'scet_batch': ('strengthening', 'scet'),
    'ce_batch': ('strengthening', 'ce'),
    'ce_top1': ('scet_allocation', 'ce_cap1'),
    'scet_top1': ('experiments', 'scet_top1'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_query(path):
    return pd.read_csv(path, dtype={'query_id': str})


def metadata(queries, ids):
    rows = []
    for qid in ids:
        q = queries[qid]
        sets = [set(s) for s in q['gold_sets'] if s]
        smallest = min(map(len, sets)) if sets else 0
        assert bool(q['gold_count']) == bool(smallest), qid
        group = 'no_annotated_evidence' if not smallest else ('singleton_set' if smallest == 1 else 'multisentence_set')
        rows.append({'query_id': qid, 'minimum_set_size': smallest,
                     'set_size_group': group, 'original_gold_count': q['gold_count']})
    return pd.DataFrame(rows)


def verify_queries(frame, scores, saved, queries):
    rank = admission.ranks(frame, scores)
    by_query = frame.groupby('query_id', sort=False).indices
    ncheck = 0
    for r in saved.itertuples(index=False):
        ix = np.asarray(by_query[r.query_id])
        selected = ix[rank[ix] <= r.admitted]
        ids = set(frame.iloc[selected].candidate_id)
        q = queries[r.query_id]
        actual = any(set(s) <= ids for s in q['gold_sets'] if s)
        assert int(actual) == r.all_facts, (r.query_id, 'all_facts')
        assert int(frame.iloc[selected].gold_evidence.sum()) == r.correct, (r.query_id, 'correct')
        ncheck += 1
    return ncheck


def summaries(stats, ds, seed, method):
    rows = []
    positive = stats.original_gold_count > 0
    groups = {'all_queries': np.ones(len(stats), bool),
              'annotated_queries': positive,
              'singleton_set': stats.set_size_group == 'singleton_set',
              'multisentence_set': stats.set_size_group == 'multisentence_set',
              'no_annotated_evidence': ~positive}
    original_denominator = int(positive.sum())
    for group, mask in groups.items():
        s = stats.loc[mask]
        n = len(s)
        adm, correct, complete = int(s.admitted.sum()), int(s.correct.sum()), int(s.all_facts.sum())
        gold = s.original_gold_count > 0
        recall = np.mean(s.loc[gold, 'correct'] / s.loc[gold, 'original_gold_count']) if gold.any() else np.nan
        rows.append({'dataset': ds, 'seed': seed, 'method': method, 'group': group,
                     'queries': n, 'annotated_queries_original': original_denominator,
                     'admitted': adm, 'correct': correct, 'complete_sets': complete,
                     'precision': correct / adm if adm else np.nan,
                     'query_coverage': float(s.covered.mean()) if n else np.nan,
                     'evidence_recall': recall,
                     'set_recall_conditional': complete / int(gold.sum()) if gold.any() else np.nan,
                     'set_recall_original_denominator': complete / original_denominator,
                     'complete_per_100_all_queries': 100 * complete / len(stats),
                     'items_per_100_all_queries': 100 * adm / len(stats)})
    return rows


def run(smoke=False):
    out = OUT / ('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    if not smoke:
        gate = json.loads((OUT / 'smoke/audit.json').read_text())
        assert gate['passed'] and gate['source_sha256'] == digest(Path(__file__))
    rows, detailed, sources, checks = [], [], {}, {}
    for ds in DATASETS:
        allq_path = R2 / f'data/processed/{ds}_full_queries.json'
        allq = json.loads(allq_path.read_text(encoding='utf-8'))
        sources[str(allq_path.relative_to(R2))] = digest(allq_path)
        for seed in ([13] if smoke else SEEDS):
            model_path = R2 / f'results/experiments/{ds}_full_k100/model_seed{seed}.json'
            model = json.loads(model_path.read_text())
            ids = model['evaluation_query_ids']
            if smoke:
                ids = sorted(ids)[:12]
            meta = metadata(allq, ids)
            selected = {}
            for method, (directory, filename) in METHODS.items():
                path = (R2 / f'results/scet_allocation/full/query_{ds}_seed{seed}_{filename}.csv'
                        if directory == 'scet_allocation' else
                        R2 / f'results/{directory}/{ds}_full_k100/query_seed{seed}_{filename}.csv')
                s = read_query(path)
                if smoke:
                    s = s[s.query_id.isin(ids)]
                assert set(s.query_id) == set(ids)
                s = s.merge(meta, on='query_id', how='left', validate='one_to_one')
                assert s.original_gold_count.eq(s.gold_count).all()
                assert s.all_facts.le(s.original_gold_count.gt(0).astype(int)).all()
                if not smoke:
                    assert s.admitted.sum() == len(ids), (ds, seed, method)
                if method.endswith('top1'):
                    assert s.admitted.eq(1).all()
                    assert not s.loc[s.minimum_set_size > 1, 'all_facts'].any()
                selected[method] = s
                rows.extend(summaries(s, ds, seed, method))
                detailed.append(s.assign(dataset=ds, seed=seed, method=method))
                sources[str(path.relative_to(R2))] = digest(path)
            if smoke:
                columns = ['query_id', 'candidate_id', 'tie_key', 'query_text', 'evidence_text', 'gold_evidence'] + admission.VIEWS
                frame = pd.read_parquet(R2 / f'data/scored/{ds}_full_k100.parquet', columns=columns,
                                        filters=[('query_id', 'in', ids)]).reset_index(drop=True)
                scores, _, _ = admission.score_pipeline(frame, model)
                for method, s in selected.items():
                    sc = scores['scet'] if method.startswith('scet') else frame.raw_ce.to_numpy(float)
                    checks[f'{ds}:{seed}:{method}'] = verify_queries(frame, sc, s, allq)
                reverse = frame.iloc[::-1].reset_index(drop=True)
                reversed_scores, _, _ = admission.score_pipeline(reverse, model)
                np.testing.assert_allclose(reversed_scores['scet'][::-1], scores['scet'], rtol=0, atol=1e-12)
            print(ds, seed, 'set-size summaries complete', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'by_partition.csv', index=False)
    detail = pd.concat(detailed, ignore_index=True)
    detail.to_csv(out / 'query_outcomes.csv', index=False)
    keys = ['dataset', 'method', 'group']
    numeric = [c for c in df.select_dtypes(include=np.number) if c != 'seed']
    mean = df.groupby(keys)[numeric].agg(['mean', 'std']).reset_index()
    mean.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in mean.columns]
    mean.to_csv(out / 'mean_sd.csv', index=False)
    if not smoke:
        seed_occurrences = detail[detail.method == 'scet_batch'].groupby(['dataset', 'query_id']).seed.nunique()
        occurrence = seed_occurrences.groupby('dataset').agg(['count', 'sum'])
        occurrence.to_csv(out / 'partition_overlap.csv')
        paired_intervals(detail).to_csv(out / 'paired_set_intervals.csv', index=False)
        make_report(df, mean, out)
    audit = {'passed': True, 'smoke': smoke, 'pool': 100, 'budget': '100 sentences per 100 held-out queries',
             'methods': list(METHODS), 'source_sha256': digest(Path(__file__)),
             'source_files_sha256': sources, 'independent_smoke_query_checks': checks,
             'definition': 'Minimum size among original nonempty annotated evidence sets; no missing annotations removed from denominators.',
             'analysis_scope': 'Descriptive stratification of already frozen decisions; no model selection or thresholds updated.',
             'partition_counts': 'Per-partition counts are separate; overlapping queries are not summed as independent observations.'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


def paired_intervals(detail):
    """Query-cluster intervals conditional on frozen admissions, matching R2."""
    output = []
    for ds in DATASETS:
        d = detail[detail.dataset == ds]
        ids = sorted(d.query_id.unique())
        lookup = {q: i for i, q in enumerate(ids)}
        for group in ['annotated_queries', 'singleton_set', 'multisentence_set']:
            den = np.zeros((len(ids), len(SEEDS)))
            num = np.zeros((len(ids), len(METHODS) * len(SEEDS)))
            for si, seed in enumerate(SEEDS):
                for mi, method in enumerate(METHODS):
                    s = d[(d.seed == seed) & (d.method == method)]
                    keep = s.original_gold_count.gt(0) if group == 'annotated_queries' else s.set_size_group.eq(group)
                    ix = np.asarray([lookup[q] for q in s.query_id[keep]], dtype=np.int64)
                    den[ix, si] = 1
                    num[ix, mi * len(SEEDS) + si] = s.loc[keep, 'all_facts'].to_numpy(float)
            if not den.sum():
                continue
            point = num.sum(axis=0).reshape(len(METHODS), len(SEEDS)) / den.sum(axis=0)[None, :]
            rng = np.random.default_rng(admission.PROTOCOL['bootstrap_seed'])
            draws = []
            for start in range(0, 1000, 100):
                weight = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=min(100, 1000-start))
                denominator = weight @ den
                numerator = (weight @ num).reshape(len(weight), len(METHODS), len(SEEDS))
                value = np.divide(numerator, denominator[:, None, :], out=np.full_like(numerator, np.nan), where=denominator[:, None, :] > 0)
                draws.append(np.mean(value, axis=2))
            boot = np.concatenate(draws, axis=0)
            for mi, reference in enumerate(list(METHODS)[1:], start=1):
                delta = boot[:, 0] - boot[:, mi]
                lo, hi = np.nanquantile(delta, [.025, .975])
                output.append({'dataset': ds, 'group': group, 'reference': reference,
                               'delta_set_recall': float(np.mean(point[0] - point[mi])),
                               'ci95_low': float(lo), 'ci95_high': float(hi),
                               'bootstrap_repetitions': 1000, 'unique_query_clusters': len(ids),
                               'interpretation': 'Conditional on frozen admissions and observed query partitions.'})
    return pd.DataFrame(output)


def make_report(df, mean, out):
    lines = ['# Evidence-set-size analysis', '',
             'All methods admit exactly 100 sentences per 100 held-out claims. Groups use the smallest original nonempty annotated evidence set. Complete-set recall retains every originally annotated claim in its overall denominator. Counts below are per-partition means (sample standard deviations); partitions overlap.', '',
             '| Dataset | Method | Precision | Overall complete-set recall | Singleton-set recall | Multi-sentence-set recall | Multi-sentence complete claims |',
             '|---|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for method in METHODS:
            p = mean[(mean.dataset == ds) & (mean.method == method)].set_index('group')
            a, one, multi = (p.loc[g] for g in ['annotated_queries', 'singleton_set', 'multisentence_set'])
            allrow = p.loc['all_queries']
            lines.append(f"| {ds} | {method} | {allrow.precision_mean:.3f} | {a.set_recall_conditional_mean:.3f} | {one.set_recall_conditional_mean:.3f} | {multi.set_recall_conditional_mean:.3f} | {multi.complete_sets_mean:.1f} ({multi.complete_sets_std:.1f}) |")
    lines += ['', '## HoVer exact counts', '',
              '| Seed | Method | All annotated claims | Multi-sentence claims | Complete multi-sentence claims | Overall denominator recall | Conditional multi-sentence recall |',
              '|---|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for method in METHODS:
            s = df[(df.dataset == 'hover') & (df.seed == seed) & (df.method == method)].set_index('group')
            a, m = s.loc['annotated_queries'], s.loc['multisentence_set']
            lines.append(f"| {seed} | {method} | {int(a.queries)} | {int(m.queries)} | {int(m.complete_sets)} | {a.set_recall_conditional:.5f} | {m.set_recall_conditional:.5f} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
