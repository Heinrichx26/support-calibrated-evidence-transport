"""Prepare previously unscored SciFact claims with source-family separation."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

R2 = Path(__file__).resolve().parents[1]
ROOT = R2.parent
sys.path.insert(0, str(ROOT/'src'))
from ipm_evidence_dataset_adapters import _scifact_corpus, build_candidate_token_index, top_candidates_by_index, lexical_overlap

OUT = R2/'data/multivers_confirmation'


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sources(claim):
    return set(map(str, claim.get('cited_doc_ids', []))) | set(claim.get('evidence', {}))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = ROOT/'data/raw_open/evidence_retrieval/scifact'
    train = [json.loads(l) for l in (raw/'claims_train.jsonl').read_text().splitlines()]
    dev = [json.loads(l) for l in (raw/'claims_dev.jsonl').read_text().splitlines()]
    used_sources = set().union(*(sources(x) for x in dev))
    z = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)).fit_transform([x['claim'] for x in train+dev])
    maximum = (z[:len(train)]@z[len(train):].T).max(axis=1).toarray().ravel()
    claims = [x for i, x in enumerate(train) if not sources(x)&used_sources and maximum[i] < .85]
    assert not {str(x['id']) for x in claims}&{str(x['id']) for x in dev}
    corpus = _scifact_corpus(raw/'corpus.jsonl')
    by_id = {x['candidate_id']: x for x in corpus}
    index = build_candidate_token_index(corpus)
    records, queries = [], {}
    # Connected components preserve related claim families for uncertainty estimates.
    parent = {str(x['id']): str(x['id']) for x in claims}
    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q
    owner = {}
    for x in claims:
        q = str(x['id'])
        for d in sorted(sources(x)):
            if d in owner:
                parent[find(q)] = find(owner[d])
            else:
                owner[d] = q
    for i, claim in enumerate(claims):
        qid, query = str(claim['id']), claim['claim']
        groups, support, refute = [], set(), set()
        for doc, evidence in claim.get('evidence', {}).items():
            for group in evidence:
                ids = {f'{doc}::{s}' for s in group['sentences']}
                groups.append(sorted(ids))
                (support if group['label'] == 'SUPPORT' else refute).update(ids)
        gold = support | refute
        retrieved = top_candidates_by_index(query, corpus, index, 100)
        natural = {x['candidate_id'] for x in retrieved}
        merged = {x['candidate_id']: x for x in retrieved}
        merged.update({c: by_id[c] for c in gold})
        ordered = sorted(merged.values(), key=lambda row: (row['candidate_id'] not in gold, -lexical_overlap(query, row['evidence_text'])))
        controlled = {x['candidate_id'] for x in ordered[:max(100, len(gold))]}
        for cid in sorted(natural | controlled):
            c = by_id[cid]
            records.append({'dataset': 'scifact_confirmation', 'query_id': qid, 'query_text': query,
                            'candidate_id': cid, 'doc_id': c['doc_id'], 'sentence_id': str(c['sentence_id']),
                            'evidence_text': c['evidence_text'], 'evidence_source': c['evidence_source'],
                            'gold_evidence': int(cid in gold), 'gold_directional': int(cid in support),
                            'natural_pool': cid in natural, 'controlled_pool': cid in controlled,
                            'tie_key': sha(f'73:scifact:{qid}:{cid}')[:16]})
        queries[qid] = {'query_text': query, 'claim_verdict': 'SUPPORT' if support else ('CONTRADICT' if refute else 'NOT_ENOUGH_INFO'),
                        'gold_ids': sorted(gold), 'gold_sets': groups, 'support_ids': sorted(support), 'refute_ids': sorted(refute),
                        'gold_count': len(gold), 'verified_gold_count': len(gold), 'fully_verified': True,
                        'source_family': find(qid), 'cited_doc_ids': sorted(sources(claim)), 'num_hops': 1}
        if (i+1) % 100 == 0:
            print('prepared', i+1, '/', len(claims), flush=True)
    frame = pd.DataFrame(records).sort_values(['query_id', 'tie_key']).reset_index(drop=True)
    assert not frame.duplicated(['query_id', 'candidate_id']).any()
    assert frame.groupby('query_id').natural_pool.sum().eq(100).all()
    assert frame.groupby('query_id').controlled_pool.sum().eq(100).all()
    frame.to_parquet(OUT/'candidates.parquet', index=False)
    (OUT/'queries.json').write_text(json.dumps(queries, ensure_ascii=False), encoding='utf-8')
    audit = {'query_count': len(queries), 'source_families': len({x['source_family'] for x in queries.values()}),
             'candidate_rows_union': len(frame), 'claim_document_pairs': len(frame[['query_id', 'doc_id']].drop_duplicates()),
             'official_split': 'SciFact train', 'checkpoint': 'fever_sci, without SciFact training',
             'exclusion': 'Any cited/evidence source shared with the 300 previously evaluated dev claims, or maximum character TF-IDF cosine >=0.85 with dev claims.',
             'natural_pool': 'Top 100 lexical-overlap sentences, independent of gold annotations.',
             'controlled_pool': 'Same original controlled-pool construction: retain all annotated gold sentences and fill to 100 from lexical retrieval.',
             'inference_input': 'Claim text, document title, all abstract sentences; labels, source-family identifiers and pool flags never enter the network.',
             'uncertainty': 'Bootstrap source families, preserving claim variants.',
             'evaluation_gate': 'No outcome evaluation before final method specification and fitted models are frozen.',
             'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw/'claims_train.jsonl', raw/'claims_dev.jsonl', raw/'corpus.jsonl']},
             'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'preparation_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ['query_count', 'source_families', 'candidate_rows_union', 'claim_document_pairs']}), flush=True)


if __name__ == '__main__':
    prepare()
