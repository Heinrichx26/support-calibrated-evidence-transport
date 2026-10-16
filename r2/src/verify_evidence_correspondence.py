"""Verify saved FEVER evidence correspondence and compare frozen stress outputs.

No scoring or model fitting is performed. Source collections remain read-only.
Run --smoke first, inspect the result, then run --full.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


R2 = Path(__file__).resolve().parents[1]
PROJECT = R2.parent
DEST = R2 / 'results/evidence_completion/correspondence_conflict'
SEEDS = [13, 17, 29, 7]


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def annotations():
    original = {}
    raw_entries = duplicate_within_groups = repeated_groups = 0
    path = PROJECT / 'data/raw_open/evidence_retrieval/fever/shared_task_dev.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        groups = []
        for group in item.get('evidence', []):
            pairs = [f'{x[2]}::{x[3]}' for x in group
                     if len(x) >= 4 and x[2] is not None and x[3] is not None]
            raw_entries += len(pairs)
            duplicate_within_groups += len(pairs) - len(set(pairs))
            if pairs:
                groups.append(tuple(sorted(set(pairs))))
        repeated_groups += len(groups) - len(set(groups))
        original[str(item['id'])] = {
            'claim': item['claim'], 'verdict': item['label'].replace(' ', '_'),
            'gold_ids': sorted({p for g in groups for p in g}), 'groups': groups}
    queries = json.loads((R2 / 'data/processed/fever_full_queries.json').read_text(encoding='utf-8'))
    for qid, q in queries.items():
        assert qid in original
        assert q['query_text'] == original[qid]['claim'], (qid, 'claim mismatch')
        assert q['claim_verdict'] == original[qid]['verdict'], (qid, 'verdict mismatch')
        assert q['gold_ids'] == original[qid]['gold_ids'], (qid, 'annotation identity mismatch')
        assert {tuple(sorted(g)) for g in q['gold_sets']} == set(original[qid]['groups'])
        assert q['gold_count'] == len(original[qid]['gold_ids'])
    meta = {'original_queries': len(original), 'recorded_queries': len(queries),
            'raw_nonnull_annotation_entries': raw_entries,
            'repeated_query_sentence_entries_across_annotation_groups': raw_entries - sum(len(x['gold_ids']) for x in original.values()),
            'repeated_pair_entries_within_groups': duplicate_within_groups,
            'repeated_evidence_groups': repeated_groups,
            'unique_query_sentence_pairs_original': sum(len(x['gold_ids']) for x in original.values()),
            'unique_query_sentence_pairs_recorded': sum(len(x['gold_ids']) for x in queries.values()),
            'queries_absent_from_recorded_pool': [
                {'query_id': q, 'claim': original[q]['claim'], 'verdict': original[q]['verdict'],
                 'gold_ids': original[q]['gold_ids']} for q in sorted(set(original) - set(queries))],
            'annotation_join_mismatches': 0}
    return original, queries, meta


def scan_snapshot(cache: pd.DataFrame, queries: dict, smoke: bool):
    gold_ids = {c for q in queries.values() for c in q['gold_ids']}
    required = set(cache.candidate_id) | gold_ids
    with ZipFile(R2 / 'data/raw/fever/wiki-pages.zip') as archive:
        members = [n for n in archive.namelist() if n.startswith('wiki-pages/') and n.endswith('.jsonl')]
        if smoke:
            selected = sorted(gold_ids & set(cache.candidate_id), key=sha)[:12]
            required = set(selected)
            members = sorted(set(cache.set_index('candidate_id').loc[selected, 'member']))
        wanted = defaultdict(set)
        for cid in required:
            title, sid = cid.rsplit('::', 1)
            wanted[title].add(sid)
        pages = Counter()
        line_counts = Counter()
        blanks = Counter()
        texts = defaultdict(dict)
        started = time.monotonic()
        for j, member in enumerate(members):
            with archive.open(member) as stream:
                for line in stream:
                    page = json.loads(line)
                    title = page['id']
                    if title not in wanted:
                        continue
                    pages[title] += 1
                    for sentence in page['lines'].split('\n'):
                        fields = sentence.split('\t')
                        if not fields or fields[0] not in wanted[title]:
                            continue
                        cid = f'{title}::{fields[0]}'
                        line_counts[cid] += 1
                        text = fields[1].strip() if len(fields) > 1 else ''
                        if text:
                            texts[cid].setdefault(text, []).append(member)
                        else:
                            blanks[cid] += 1
            if j % 10 == 0 or j + 1 == len(members):
                print(f'Snapshot shards {j+1}/{len(members)}; {time.monotonic()-started:.1f}s', flush=True)
    records = []
    resolved = {}
    for cid in sorted(required):
        title, sid = cid.rsplit('::', 1)
        variants = texts[cid]
        status = ('resolved' if len(variants) == 1 else 'ambiguous_text' if len(variants) > 1
                  else 'page_absent' if not pages[title]
                  else 'empty_sentence' if blanks[cid] else 'sentence_id_absent')
        text = next(iter(variants)) if status == 'resolved' else None
        if text is not None:
            resolved[cid] = text
        records.append({'candidate_id': cid, 'page_id': title, 'sentence_id': sid,
                        'status': status, 'page_occurrences': pages[title],
                        'sentence_line_occurrences': line_counts[cid], 'blank_occurrences': blanks[cid],
                        'distinct_nonempty_texts': len(variants),
                        'sentence_sha256': sha(text) if text is not None else '',
                        'source_members': '|'.join(sorted({m for ms in variants.values() for m in ms}))})
    return pd.DataFrame(records), resolved, gold_ids


def verify_pools(original: dict, cache: pd.DataFrame, resolved: dict, smoke: bool):
    cache_lookup = cache.set_index('candidate_id').to_dict('index')
    checked = []
    seen_gold = set()
    for pool in [100] if smoke else [100, 500]:
        rows = label_checks = text_checks = 0
        parquet = pq.ParquetFile(R2 / f'data/processed/fever_full_k{pool}.parquet')
        columns = ['query_id', 'query_text', 'candidate_id', 'evidence_text',
                   'sentence_sha256', 'gold_evidence', 'text_provenance']
        for batch in parquet.iter_batches(batch_size=25000, columns=columns):
            frame = batch.to_pandas()
            for row in frame.itertuples(index=False):
                gold = row.candidate_id in original[row.query_id]['gold_ids']
                assert int(row.gold_evidence) == int(gold), (pool, row.query_id, row.candidate_id, 'label')
                assert row.query_text == original[row.query_id]['claim']
                assert row.text_provenance == 'FEVER_JUNE_2017'
                entry = cache_lookup[row.candidate_id]
                assert row.evidence_text == entry['evidence_text']
                assert row.sentence_sha256 == entry['sentence_sha256']
                label_checks += 1
                if pool == 100 and gold:
                    seen_gold.add((row.query_id, row.candidate_id))
                if row.candidate_id in resolved:
                    sentence = resolved[row.candidate_id]
                    title = row.candidate_id.rsplit('::', 1)[0]
                    assert row.evidence_text == f'{title.replace("_", " ")}. {sentence}'
                    assert row.sentence_sha256 == sha(sentence)
                    text_checks += 1
            rows += len(frame)
            if smoke:
                break
        checked.append({'pool_cap': pool, 'rows': rows, 'label_assertions': label_checks,
                        'source_text_assertions': text_checks, 'mismatches': 0})
    return checked, seen_gold


def stress_analysis(smoke: bool, out: Path):
    dataset_rows, partition_rows, intervals, shared_records = [], [], [], []
    datasets = ['fever'] if smoke else ['scifact', 'fever', 'hover']
    for dataset in datasets:
        frames = {}
        for seed in SEEDS[:1] if smoke else SEEDS:
            root = R2 / f'results/experiments/{dataset}_full_k100'
            shared = pd.read_parquet(root / f'shared_stress_seed{seed}.parquet')
            model = json.loads((root / f'model_seed{seed}.json').read_text(encoding='utf-8'))
            assert np.array_equal(shared.stress.to_numpy(), shared.conflict.to_numpy() >= model['stress_cutoff'])
            assert not shared.duplicated(['query_id', 'candidate_id']).any()
            shared_records.append({'dataset': dataset, 'seed': seed, 'candidate_rows': len(shared),
                                   'stress_candidate_rows': int(shared.stress.sum()),
                                   'stress_cutoff': model['stress_cutoff'], 'shared_boundary_mismatches': 0})
            pair = {}
            for method in ['scet', 'ce']:
                frame = pd.read_csv(root / f'query_seed{seed}_{method}.csv', dtype={'query_id': str}).set_index('query_id')
                pair[method] = frame
                n = len(frame)
                admitted, correct = int(frame.stress_admitted.sum()), int(frame.stress_correct.sum())
                stats = {'dataset': dataset, 'seed': seed, 'method': method, 'queries': n,
                         'stress_admitted': admitted, 'stress_correct': correct,
                         'stress_unannotated': admitted - correct,
                         'stress_precision': correct / admitted if admitted else math.nan,
                         'stress_unannotated_fraction': 1 - correct / admitted if admitted else math.nan,
                         'stress_output_queries': int(frame.stress_admitted.gt(0).sum()),
                         'stress_query_coverage': float(frame.stress_admitted.gt(0).mean()),
                         'stress_unannotated_queries': int(frame.stress_admitted.gt(frame.stress_correct).sum()),
                         'stress_query_error_rate': float(frame.stress_admitted.gt(frame.stress_correct).mean()),
                         'all_admitted': int(frame.admitted.sum()),
                         'stress_fraction_of_output': admitted / frame.admitted.sum()}
                partition_rows.append(stats)
            assert pair['scet'].index.equals(pair['ce'].index)
            frames[seed] = pair
        rows = pd.DataFrame([r for r in partition_rows if r['dataset'] == dataset])
        for method, group in rows.groupby('method'):
            d = {'dataset': dataset, 'method': method, 'partitions': len(group)}
            for col in group.select_dtypes(include='number').columns:
                if col == 'seed':
                    continue
                d[col + '_mean'] = float(group[col].mean())
                d[col + '_sd'] = float(group[col].std()) if len(group) > 1 else 0.0
            dataset_rows.append(d)
        if not smoke:
            ids = sorted({q for p in frames.values() for q in p['scet'].index})
            lookup = {q: i for i, q in enumerate(ids)}
            rng = np.random.default_rng(20261016)
            draws = {k: [] for k in ['stress_unannotated_fraction', 'stress_query_coverage', 'stress_query_error_rate']}
            for _ in range(10):
                weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=100)
                per_partition = {k: [] for k in draws}
                for pair in frames.values():
                    a, b = pair['scet'], pair['ce']
                    w = weights[:, [lookup[q] for q in a.index]]
                    for metric in draws:
                        values = []
                        for f in [a, b]:
                            if metric == 'stress_unannotated_fraction':
                                den = w @ f.stress_admitted.to_numpy()
                                num = w @ (f.stress_admitted - f.stress_correct).to_numpy()
                            else:
                                den = w.sum(axis=1)
                                num = w @ ((f.stress_admitted > 0).to_numpy() if metric == 'stress_query_coverage'
                                           else (f.stress_admitted > f.stress_correct).to_numpy())
                            values.append(np.divide(num, den, out=np.full(len(den), np.nan), where=den > 0))
                        per_partition[metric].append(values[0] - values[1])
                for metric in draws:
                    draws[metric].extend(np.nanmean(per_partition[metric], axis=0).tolist())
            for metric, sample in draws.items():
                a = rows[rows.method == 'scet'].set_index('seed')[metric]
                b = rows[rows.method == 'ce'].set_index('seed')[metric]
                intervals.append({'dataset': dataset, 'metric': metric, 'difference_scet_minus_ce': float((a-b).mean()),
                                  'ci_low': float(np.nanpercentile(sample, 2.5)),
                                  'ci_high': float(np.nanpercentile(sample, 97.5)),
                                  'bootstrap_draws': 1000, 'unit': 'query cluster shared across overlapping partitions',
                                  'scope': 'conditional on frozen shared-stress and admission decisions'})
    pd.DataFrame(partition_rows).to_csv(out / 'stress_by_partition.csv', index=False)
    pd.DataFrame(dataset_rows).to_csv(out / 'stress_mean_sd.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'stress_paired_intervals.csv', index=False)
    pd.DataFrame(shared_records).to_csv(out / 'shared_stress_definitions.csv', index=False)
    return dataset_rows, intervals


def main(smoke: bool):
    out = DEST / ('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    code_hash = sha(Path(__file__).read_bytes())
    if not smoke:
        prior = json.loads((DEST / 'smoke/summary.json').read_text(encoding='utf-8'))
        assert prior['passed'] and prior['code_sha256'] == code_hash, 'Run and inspect the current smoke first.'
    original, queries, meta = annotations()
    cache = pd.read_parquet(R2 / 'data/raw/fever/verified_sentences.parquet')
    assert not cache.candidate_id.duplicated().any(), 'Ambiguous cache identities require explicit review.'
    assert all(sha(row.sentence_text) == row.sentence_sha256 for row in cache.itertuples(index=False))
    source, resolved, gold_ids = scan_snapshot(cache, queries, smoke)
    source.to_csv(out / 'source_identity_audit.csv', index=False)
    checked, pool_gold = verify_pools(original, cache, resolved, smoke)
    source_lookup = source.set_index('candidate_id').to_dict('index')
    occurrence_rows, samples = [], []
    for qid, q in queries.items():
        for cid in q['gold_ids']:
            if cid not in source_lookup:
                continue
            info = source_lookup[cid]
            record = {'query_id': qid, 'claim_verdict': q['claim_verdict'], 'candidate_id': cid, **info}
            record['in_k100_pool'] = (qid, cid) in pool_gold
            occurrence_rows.append(record)
            if len([s for s in samples if s['status'] == info['status'] and s['claim_verdict'] == q['claim_verdict']]) < 3:
                samples.append({**record, 'claim': q['query_text'], 'sentence_text': resolved.get(cid)})
    occurrences = pd.DataFrame(occurrence_rows)
    occurrences.to_csv(out / 'annotation_correspondence.csv', index=False)
    dump(out / 'examples.json', samples)
    if not smoke:
        assert len(occurrences) == meta['unique_query_sentence_pairs_recorded']
        assert all((row.status == 'resolved') == row.in_k100_pool for row in occurrences.itertuples(index=False))
        for row in cache.itertuples(index=False):
            assert row.candidate_id in resolved, ('cached source now unresolved', row.candidate_id)
            assert row.sentence_text == resolved[row.candidate_id]
    unresolved = occurrences[occurrences.status != 'resolved']
    meta.update({'unresolved_distinct_sentence_ids': int(unresolved.candidate_id.nunique()),
                 'unresolved_distinct_pages': int(unresolved.page_id.nunique()),
                 'unresolved_distinct_queries': int(unresolved.query_id.nunique()),
                 'unresolved_occurrences_by_verdict': unresolved.groupby('claim_verdict').size().to_dict(),
                 'full_evidence_queries': sum(q['fully_verified'] for q in queries.values()),
                 'annotation_status_by_verdict': occurrences.groupby(['claim_verdict', 'status']).size().reset_index(name='query_sentence_pairs').to_dict('records')})
    stress, ci = stress_analysis(smoke, out)
    meta.update({'mode': 'smoke' if smoke else 'full', 'passed': True, 'code_sha256': code_hash,
                 'source_identity_count': len(source), 'source_status_unique_id': source.status.value_counts().to_dict(),
                 'annotation_status_query_sentence_pairs': occurrences.status.value_counts().to_dict(),
                 'duplicate_cache_identities': 0,
                 'source_ids_with_repeated_sentence_lines': int(source.sentence_line_occurrences.gt(1).sum()),
                 'source_ids_with_ambiguous_nonempty_text': int(source.distinct_nonempty_texts.gt(1).sum()),
                 'pool_checks': checked, 'stress_mean_sd': stress, 'stress_paired_intervals': ci,
                 'definitions': {'annotation_occurrence': 'One unique query, original page ID and sentence ID tuple; repeated alternative groups share this tuple.',
                                 'unannotated': 'An item outside the supplied annotated verification-evidence set.',
                                 'stress_query_coverage': 'Fraction of all held-out queries receiving any shared-stress item.',
                                 'stress_query_error': 'A held-out query receiving any shared-stress item outside its annotated evidence set.'}})
    dump(out / 'summary.json', meta)
    print(json.dumps({k: meta[k] for k in ['mode', 'passed', 'annotation_status_query_sentence_pairs', 'pool_checks']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    args = parser.parse_args()
    main(args.smoke)
