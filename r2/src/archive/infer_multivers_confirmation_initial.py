"""Resume-safe, label-independent MultiVerS inference on reserved claims."""
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
import torch
from run_multivers_r2 import load_model, MultiVerSDataset, Collator, R2

DATA = R2/'data/multivers_confirmation'
OUT = R2/'results/multivers_extension/confirmation_inference'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(smoke):
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    code_sha, input_sha = digest(Path(__file__)), digest(DATA/'candidates.parquet')
    if not smoke:
        audit = json.loads((OUT/'smoke_audit.json').read_text())
        assert audit['passed'] and audit['code_sha256'] == code_sha and audit['input_sha256'] == input_sha
    torch.set_num_threads(6)
    model, tokenizer = load_model(OUT/('smoke_model_manifest.json' if smoke else 'model_manifest.json'))
    # Read precisely the inference fields. No labels or verdicts are accessed.
    frame = pd.read_parquet(DATA/'candidates.parquet', columns=['query_id', 'query_text', 'doc_id'])
    corpus = {str(d['doc_id']): d for d in [json.loads(l) for l in (R2/'data/raw/scifact/corpus.jsonl').read_text(encoding='utf-8').splitlines()]}
    entries = []
    for (qid, did), part in frame.groupby(['query_id', 'doc_id'], sort=True):
        doc = corpus[did]
        entries.append({'claim_id': int(qid), 'abstract_id': int(did), 'to_tensorize': {
            'claim': part.query_text.iloc[0], 'title': doc['title'], 'sentences': doc['abstract']}})
    if smoke:
        entries = entries[:8]
    output = OUT/('document_scores_smoke.jsonl' if smoke else 'document_scores.jsonl')
    done = set()
    if output.exists() and not smoke:
        done = {(r['query_id'], r['doc_id']) for r in [json.loads(l) for l in output.read_text().splitlines()]}
    dataset, collator = MultiVerSDataset(entries, tokenizer), Collator(tokenizer)
    items, truncated = [], 0
    for i, entry in enumerate(entries):
        if (str(entry['claim_id']), str(entry['abstract_id'])) in done:
            continue
        item = dataset[i]
        if len(item['tokenized'].input_ids) > 4096:
            item['tokenized'], item['abstract_sent_idx'] = dataset._tokenize_truncated(**entry['to_tensorize'])
            truncated += 1
        items.append(item)
    items.sort(key=lambda x: len(x['tokenized'].input_ids))
    processed, limit, last = 0, 16, time.monotonic()
    with output.open('w' if smoke else 'a', encoding='utf-8') as stream, torch.inference_mode():
        while processed < len(items):
            window = max(512, ((len(items[processed]['tokenized'].input_ids)+511)//512)*512)
            size = min(limit, max(1, 8192//window))
            batch = items[processed:processed+size]
            tensor = collator(batch)
            inputs = {k: v.to('cuda') for k, v in tensor['tokenized'].items()}
            try:
                with torch.autocast('cuda', dtype=torch.float16):
                    result = model(inputs, tensor['abstract_sent_idx'].to('cuda'))
            except torch.cuda.OutOfMemoryError:
                assert size > 1
                limit = size//2
                torch.cuda.empty_cache()
                continue
            labels = result['label_probs'].float().cpu().numpy()
            rationale = result['rationale_probs'].float().cpu().numpy()
            label_logits = result['label_logits'].float().cpu().numpy()
            rationale_logits = result['rationale_logits'].float().cpu().numpy()
            assert np.isfinite(labels).all() and np.isfinite(rationale).all()
            assert np.allclose(labels.sum(axis=1), 1, atol=.002)
            for j, item in enumerate(batch):
                count = len(item['abstract_sent_idx'])
                row = {'query_id': str(item['claim_id']), 'doc_id': str(item['abstract_id']),
                       'label_probabilities': labels[j].tolist(), 'rationale_probabilities': rationale[j, :count].tolist(),
                       'label_logits': label_logits[j].tolist(), 'rationale_logits': rationale_logits[j, :count].tolist()}
                stream.write(json.dumps(row)+'\n')
            stream.flush()
            processed += len(batch)
            if smoke or time.monotonic()-last > 25:
                print(processed, '/', len(items), 'new documents; elapsed', round(time.monotonic()-start, 1), 's', flush=True)
                last = time.monotonic()
    audit = {'passed': True, 'smoke': smoke, 'code_sha256': code_sha, 'input_sha256': input_sha,
             'documents': len(entries), 'new_documents': len(items), 'truncated_documents': truncated,
             'all_scores_finite': True, 'labels_accessed_by_inference': False, 'elapsed_seconds': time.monotonic()-start}
    (OUT/('smoke_audit.json' if smoke else 'inference_audit.json')).write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args().smoke)
