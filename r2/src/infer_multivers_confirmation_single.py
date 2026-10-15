"""Exact decision-preserving sparse inference for the frozen dual-context control."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd

import r2_admission as admission
from strengthen_r2_analysis import budget_rule
from multivers_short_attention import set_window
import multivers_short_attention

R2 = Path(__file__).resolve().parents[1]
DATA = R2/'data/multivers_confirmation'
OUT = R2/'results/multivers_extension/confirmation_single'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_probability_frame():
    frame = pd.read_parquet(DATA/'candidates.parquet')
    documents = {}
    path = R2/'results/multivers_extension/confirmation_inference/document_scores.jsonl'
    for line in path.read_text().splitlines():
        row = json.loads(line)
        documents[(row['query_id'], row['doc_id'])] = row
    expected = set(frame[['query_id', 'doc_id']].itertuples(index=False, name=None))
    assert expected == set(documents), (len(expected), len(documents))
    frame['raw_multivers_rationale'] = [documents[(q, d)]['rationale_probabilities'][int(s)] for q, d, s in zip(frame.query_id, frame.doc_id, frame.sentence_id)]
    return frame


def required_candidates(frame, threshold, weight=.75):
    required = np.zeros(len(frame), dtype=bool)
    bounds = []
    for pool in ['controlled_pool', 'natural_pool']:
        mask = frame[pool].to_numpy(bool)
        part = frame.loc[mask].reset_index(drop=True)
        lower = weight*part.raw_multivers_rationale.to_numpy(float)
        upper = lower+(1-weight)
        batch_lower_rule = budget_rule(part, lower, part.query_id.nunique())
        boundary = min(batch_lower_rule['threshold'], threshold)
        needed = upper >= boundary-1e-12
        required[np.flatnonzero(mask)[needed]] = True
        bounds.append({'pool': pool, 'batch_threshold_lower_bound': batch_lower_rule['threshold'],
                       'prospective_threshold': threshold, 'required_pairs': int(needed.sum()),
                       'omitted_upper_bound_strictly_below_both_boundaries': bool((upper[~needed] < boundary).all())})
    return required, bounds


def development_smoke():
    import run_multivers_dual_calibration as dual
    OUT.mkdir(parents=True, exist_ok=True)
    frame = dual.prepare()
    # Check the bound against previously computed exhaustive predictions, without
    # accessing evidence annotations for any performance comparison.
    freeze = json.loads((R2/'results/multivers_extension/final_portfolio/full/confirmation_freeze.json').read_text())
    threshold = freeze['models']['dual_blend']['prospective_rule']['threshold']
    frame['controlled_pool'] = True
    frame['natural_pool'] = True
    needed, bounds = required_candidates(frame, threshold)
    true = .75*frame.raw_multivers_rationale.to_numpy(float)+.25*frame.single_rationale.to_numpy(float)
    partial = .75*frame.raw_multivers_rationale.to_numpy(float)+.25*np.where(needed, frame.single_rationale.to_numpy(float), 0)
    for rule in [budget_rule(frame, true, frame.query_id.nunique()), freeze['models']['dual_blend']['prospective_rule']]:
        a, _ = admission.apply_threshold(frame, true, rule)
        b, _ = admission.apply_threshold(frame, partial, rule)
        assert np.array_equal(a, b)
        assert needed[a].all()
    partial_rule = budget_rule(frame, partial, frame.query_id.nunique())
    a, _ = admission.apply_threshold(frame, true, budget_rule(frame, true, frame.query_id.nunique()))
    b, _ = admission.apply_threshold(frame, partial, partial_rule)
    assert np.array_equal(a, b)
    audit = {'passed': True, 'source_sha256': digest(Path(__file__)), 'development_pairs': len(frame),
             'required_pairs': int(needed.sum()), 'all_admission_decisions_identical_to_exhaustive': True,
             'labels_accessed': False, 'bounds': bounds}
    (OUT/'bound_smoke_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


def infer(smoke):
    import torch
    from run_multivers_r2 import load_model, MultiVerSDataset, Collator
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source_sha = digest(Path(__file__))
    bound_audit = json.loads((OUT/'bound_smoke_audit.json').read_text())
    assert bound_audit['passed'] and bound_audit['source_sha256'] == source_sha
    attention_audit = json.loads((OUT/'short_attention_audit.json').read_text())
    assert attention_audit['passed']
    assert attention_audit['implementation_sha256'] == digest(Path(multivers_short_attention.__file__))
    freeze = json.loads((R2/'results/multivers_extension/final_portfolio/full/confirmation_freeze.json').read_text())
    # The development bound retained every pair. Exhaustive inference also
    # permits preparation to overlap the independent full-abstract inference.
    frame = pd.read_parquet(DATA/'candidates.parquet')
    bounds = [{'mode': 'exhaustive', 'pairs': len(frame), 'omitted_pairs': 0}]
    if not smoke:
        a = json.loads((OUT/'smoke_audit.json').read_text())
        assert a['passed'] and a['source_sha256'] == source_sha
    else:
        frame = frame.iloc[:32].copy()
    destination = OUT/('scores_smoke.jsonl' if smoke else 'scores.jsonl')
    done = set()
    if destination.exists() and not smoke:
        done = {(r['query_id'], r['candidate_id']) for r in map(json.loads, destination.read_text().splitlines())}
    corpus = {str(d['doc_id']): d for d in map(json.loads, (R2/'data/raw/scifact/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.65)
    model, tokenizer = load_model(OUT/('smoke_model_manifest.json' if smoke else 'model_manifest.json'))
    entries = []
    for row in frame.itertuples(index=False):
        if (row.query_id, row.candidate_id) in done:
            continue
        doc = corpus[row.doc_id]
        entries.append({'claim_id': int(row.query_id), 'abstract_id': len(entries), 'candidate_id': row.candidate_id,
                        'to_tensorize': {'claim': row.query_text, 'title': doc['title'], 'sentences': [doc['abstract'][int(row.sentence_id)]]}})
    dataset, collator = MultiVerSDataset(entries, tokenizer), Collator(tokenizer)
    items = []
    for i, entry in enumerate(entries):
        item = dataset[i]
        item['candidate_id'] = entry['candidate_id']
        assert len(item['abstract_sent_idx']) == 1
        items.append(item)
    items.sort(key=lambda x: len(x['tokenized'].input_ids))
    processed, limit, last = 0, 96, time.monotonic()
    with destination.open('w' if smoke else 'a', encoding='utf-8') as stream, torch.inference_mode():
        while processed < len(items):
            length = len(items[processed]['tokenized'].input_ids)
            padded = 2*max(1, length-1) if length <= 256 else max(512, ((length+511)//512)*512)
            size = min(limit, max(1, 12288//padded))
            batch = items[processed:processed+size]
            tensor = collator([{k: v for k, v in item.items() if k != 'candidate_id'} for item in batch])
            inputs = {k: v.to('cuda') for k, v in tensor['tokenized'].items()}
            attention_window = set_window(model, inputs)
            try:
                with torch.autocast('cuda', dtype=torch.float16):
                    result = model(inputs, tensor['abstract_sent_idx'].to('cuda'))
            except torch.cuda.OutOfMemoryError:
                assert size > 1
                limit = size//2
                torch.cuda.empty_cache()
                continue
            probs = result['rationale_probs'][:, 0].float().cpu().numpy()
            assert np.isfinite(probs).all()
            for j, item in enumerate(batch):
                stream.write(json.dumps({'query_id': str(item['claim_id']), 'candidate_id': item['candidate_id'],
                                         'single_rationale': float(probs[j]), 'attention_window': attention_window})+'\n')
            stream.flush()
            processed += len(batch)
            if smoke or time.monotonic()-last > 25:
                print(processed, '/', len(items), 'required pairs; elapsed', round(time.monotonic()-start, 1), 's', flush=True)
                last = time.monotonic()
    rows = [json.loads(l) for l in destination.read_text().splitlines()]
    assert len(rows) == len(frame) and len({(r['query_id'], r['candidate_id']) for r in rows}) == len(rows)
    audit = {'passed': True, 'smoke': smoke, 'source_sha256': source_sha, 'required_pairs': len(frame),
             'bounds': bounds, 'score_interval': [0, 1], 'labels_accessed_by_inference': False,
             'short_input_attention_graph_preserved': True,
             'attention_equivalence_audit': attention_audit,
             'elapsed_seconds': time.monotonic()-start}
    (OUT/('smoke_audit.json' if smoke else 'audit.json')).write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--bound-smoke', action='store_true')
    a = p.parse_args()
    if a.bound_smoke:
        development_smoke()
    else:
        infer(a.smoke)
