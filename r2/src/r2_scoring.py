from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize

R2 = Path(__file__).resolve().parents[1]
CACHE = R2 / 'data/score_cache'
CACHE.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
MODELS = {
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'e5': 'intfloat/e5-small-v2',
    'bge': 'BAAI/bge-m3',
    'ce': 'cross-encoder/ms-marco-MiniLM-L6-v2',
    'nli': 'cross-encoder/nli-MiniLM2-L6-H768'
}


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while x := f.read(2**20):
            h.update(x)
    return h.hexdigest()


def snapshot(name):
    root = Path.home() / '.cache/huggingface/hub' / ('models--' + name.replace('/', '--'))
    ref = root / 'refs/main'
    rev = ref.read_text().strip() if ref.exists() else sorted((root / 'snapshots').iterdir())[-1].name
    path = root / 'snapshots' / rev
    assert (path / 'config.json').exists(), name
    return path, rev


def model_manifest(key):
    path, rev = snapshot(MODELS[key])
    target = CACHE / f'{key}_{rev}_manifest.json'
    if target.exists():
        return json.loads(target.read_text())
    files = {x.name: sha_file(x) for x in path.iterdir() if x.is_file() and x.suffix in {'.json', '.bin', '.safetensors', '.txt', '.model'} and x.name != 'model.onnx'}
    config = json.loads((path / 'config.json').read_text())
    manifest = {'model': MODELS[key], 'revision': rev, 'files_sha256': files, 'id2label': config.get('id2label'), 'pair_order': 'evidence premise then claim hypothesis' if key == 'nli' else 'claim then evidence', 'max_length': 256}
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def pair_digest(frame):
    values = pd.util.hash_pandas_object(frame[['query_text', 'evidence_text']], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def paired_dot(q, d, qi, di):
    out = np.zeros(len(qi), dtype=np.float32)
    for start in range(0, len(qi), 10000):
        stop = min(start + 10000, len(qi))
        if sparse.issparse(q):
            out[start:stop] = np.asarray(q[qi[start:stop]].multiply(d[di[start:stop]]).sum(axis=1)).ravel()
        else:
            out[start:stop] = np.einsum('ij,ij->i', q[qi[start:stop]], d[di[start:stop]])
    return out


def lookups(frame):
    qs = frame[['query_id', 'query_text']].drop_duplicates('query_id')
    docs = frame['evidence_text'].drop_duplicates().tolist()
    qdict = {x: i for i, x in enumerate(qs['query_id'])}
    ddict = {x: i for i, x in enumerate(docs)}
    return qs['query_text'].tolist(), docs, frame['query_id'].map(qdict).to_numpy(int), frame['evidence_text'].map(ddict).to_numpy(int)


def lexical(frame):
    queries, docs, qi, di = lookups(frame)
    print('Fixed lexical corpus', len(docs), 'sentences', flush=True)
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=150000, dtype=np.float32)
    dtf = vectorizer.fit_transform(docs).tocsr()
    qtf = vectorizer.transform(queries).tocsr()
    n = dtf.shape[0]
    df = np.bincount(dtf.indices, minlength=dtf.shape[1])
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    lengths = np.asarray(dtf.sum(axis=1)).ravel()
    rows = np.repeat(np.arange(n), np.diff(dtf.indptr))
    bm = dtf.copy()
    denom = bm.data + 1.5 * (0.25 + 0.75 * lengths[rows] / max(lengths.mean(), 1))
    bm.data = (idf[bm.indices] * bm.data * 2.5 / denom).astype(np.float32)
    out = {'raw_bm25': paired_dot(qtf, bm, qi, di)}
    del rows, bm, denom
    transformer = TfidfTransformer()
    dm = transformer.fit_transform(dtf)
    qm = transformer.transform(qtf)
    out['raw_tfidf'] = paired_dot(qm, dm, qi, di)
    del dtf, qtf, dm, qm
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=100000, dtype=np.float32)
    dchar = char.fit_transform(docs)
    svd = TruncatedSVD(n_components=min(32, dchar.shape[0]-1, dchar.shape[1]-1), random_state=13, n_iter=3)
    dlsa = normalize(svd.fit_transform(dchar)).astype(np.float32)
    qlsa = normalize(svd.transform(char.transform(queries))).astype(np.float32)
    out['raw_lsa'] = paired_dot(qlsa, dlsa, qi, di)
    qsets = [set(re.findall(r'[A-Za-z0-9]+', q.lower())) for q in queries]
    dsets = [set(re.findall(r'[A-Za-z0-9]+', d.lower())) for d in docs]
    qentities = [{x for x in s if len(x) > 5} for s in qsets]
    dentities = [{x for x in s if len(x) > 5} for s in dsets]
    out['raw_entity'] = np.asarray([len(qentities[q] & dentities[d])/max(len(qentities[q] | dentities[d]), 1) for q, d in zip(qi, di)], dtype=np.float32)
    out['raw_relation'] = np.asarray([len(qsets[q] & dsets[d])/max(len(qsets[q]), 1) for q, d in zip(qi, di)], dtype=np.float32)
    return out


def embeddings(frame, key):
    import torch
    from sentence_transformers import SentenceTransformer
    queries, docs, qi, di = lookups(frame)
    path, rev = snapshot(MODELS[key])
    prefixq, prefixd = ('query: ', 'passage: ') if key == 'e5' else ('', '')
    arrays = []
    model = None
    for role, texts, prefix in [('q', queries, prefixq), ('d', docs, prefixd)]:
        h = hashlib.sha256((MODELS[key]+rev+prefix+'256\0'+'\0'.join(texts)).encode()).hexdigest()
        target = CACHE / f'{key}_{role}_{h}.npy'
        if target.exists():
            arr = np.load(target)
        else:
            if model is None:
                model = SentenceTransformer(str(path), device='cuda' if torch.cuda.is_available() else 'cpu', model_kwargs={'torch_dtype': torch.float16})
                model.max_seq_length = 256
            batch = 48 if key == 'bge' else 256
            while True:
                try:
                    arr = model.encode([prefix+x for x in texts], batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    batch //= 2
                    assert batch >= 4
            np.save(target, arr)
        arrays.append(arr)
    result = paired_dot(arrays[0], arrays[1], qi, di)
    del model, arrays
    gc.collect()
    torch.cuda.empty_cache()
    return result


def neural_pairs(frame, key):
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path, rev = snapshot(MODELS[key])
    h = hashlib.sha256((key+rev+'256'+pair_digest(frame)).encode()).hexdigest()
    target = CACHE / f'{key}_{h}.npz'
    if target.exists():
        return dict(np.load(target))
    torch.set_num_threads(min(os.cpu_count() or 4, 6))
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path), torch_dtype=torch.float16).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    queries, docs, qi, di = lookups(frame)
    # Length estimates from the exact tokenizer support both scheduling and coverage reporting.
    qlens = np.asarray([len(x) for x in tokenizer(queries, add_special_tokens=False, truncation=False)['input_ids']])
    dlens = np.asarray([len(x) for x in tokenizer(docs, add_special_tokens=False, truncation=False)['input_ids']])
    lengths = qlens[qi] + dlens[di] + tokenizer.num_special_tokens_to_add(pair=True)
    order = np.argsort(lengths, kind='stable')
    nlabels = model.config.num_labels
    scores = np.full((len(frame), nlabels), np.nan, dtype=np.float32)
    partial = target.with_suffix('.partial.npy')
    if partial.exists():
        scores = np.load(partial)
    pending = order[~np.isfinite(scores[order, 0])]
    batchsize = 128 if key == 'ce' else 256
    completed, last, started = 0, time.monotonic(), time.monotonic()
    with torch.inference_mode():
        while completed < len(pending):
            indices = pending[completed:completed+batchsize]
            first = [docs[di[i]] if key == 'nli' else queries[qi[i]] for i in indices]
            second = [queries[qi[i]] if key == 'nli' else docs[di[i]] for i in indices]
            encoded = tokenizer(first, second, padding=True, truncation='longest_first', max_length=256, return_tensors='pt')
            try:
                logits = model(**{k: v.to(model.device) for k, v in encoded.items()}).logits.float()
                values = torch.softmax(logits, dim=-1) if key == 'nli' else logits
                scores[indices] = values.cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batchsize //= 2
                assert batchsize >= 8
                continue
            completed += len(indices)
            if time.monotonic()-last >= 30:
                np.save(partial, scores)
                print(key, completed, '/', len(pending), 'pairs;', round(completed/max(time.monotonic()-started, .01)), 'pairs/s; batch', batchsize, flush=True)
                last = time.monotonic()
    assert np.isfinite(scores).all()
    if key == 'ce':
        result = {'raw_ce': scores[:, 0]}
    else:
        mapping = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        assert {'entailment', 'contradiction', 'neutral'} <= set(mapping)
        ent, con = scores[:, mapping['entailment']], scores[:, mapping['contradiction']]
        result = {'raw_nli_relevance': ent+con, 'raw_nli_directional': ent-con, 'raw_nli_entailment': ent, 'nli_contradiction': con, 'nli_neutral': scores[:, mapping['neutral']]}
    np.savez_compressed(target, **result)
    (target.with_suffix('.json')).write_text(json.dumps({'pairs': len(frame), 'truncated_pairs': int((lengths > 256).sum()), 'truncation_fraction': float((lengths > 256).mean()), 'model': MODELS[key], 'revision': rev, 'id2label': model.config.id2label, 'max_length': 256, 'pair_order': 'evidence,claim' if key == 'nli' else 'claim,evidence', 'transformers': transformers.__version__, 'torch': torch.__version__, 'adaptive_batch_final': batchsize, 'input_sha256': pair_digest(frame)}, indent=2), encoding='utf-8')
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def score(path):
    frame = pd.read_parquet(path)
    tag = path.stem
    output = R2 / 'data/scored' / (tag+'.parquet')
    output.parent.mkdir(parents=True, exist_ok=True)
    input_digest = pair_digest(frame)
    metadata = output.with_suffix('.json')
    if output.exists() and metadata.exists() and json.loads(metadata.read_text()).get('input_sha256') == input_digest:
        print('Reusing verified score artifact', tag)
        return output
    lexpath = CACHE / f'lexical_{input_digest}.npz'
    lex = dict(np.load(lexpath)) if lexpath.exists() else lexical(frame)
    if not lexpath.exists():
        np.savez_compressed(lexpath, **lex)
    for col, values in lex.items():
        frame[col] = values
    for key in ['minilm', 'e5', 'bge']:
        print('Scoring', tag, key, flush=True)
        frame['raw_'+key] = embeddings(frame, key)
    for key in ['ce', 'nli']:
        print('Scoring', tag, key, flush=True)
        for col, values in neural_pairs(frame, key).items():
            frame[col] = values
    assert frame.filter(regex='^raw_').notna().all().all()
    frame.to_parquet(output, index=False)
    metadata.write_text(json.dumps({'input_sha256': input_digest, 'input_file_sha256': sha_file(path), 'rows': len(frame), 'query_count': frame.query_id.nunique(), 'raw_score_columns': frame.filter(regex='^raw_').columns.tolist(), 'models': {key: model_manifest(key) for key in MODELS}, 'score_file_sha256': sha_file(output)}, indent=2), encoding='utf-8')
    print('Completed raw scoring', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    for input_path in args.inputs:
        score(input_path)
