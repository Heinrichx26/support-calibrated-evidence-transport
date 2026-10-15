"""Score reserved candidates with the frozen original scorer definitions."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import r2_scoring as scoring

R2 = Path(__file__).resolve().parents[1]
DATA = R2/'data/multivers_confirmation'
OUT = R2/'results/multivers_extension/confirmation_scoring'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lexical_frozen(frame):
    reference = pd.read_parquet(R2/'data/scored/scifact_full_k100.parquet')
    oldqueries, docs, oqi, odi = scoring.lookups(reference)
    cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=150000, dtype=np.float32)
    dtf = cv.fit_transform(docs).tocsr()
    n = dtf.shape[0]
    df = np.bincount(dtf.indices, minlength=dtf.shape[1])
    idf = np.log(1+(n-df+.5)/(df+.5))
    lengths = np.asarray(dtf.sum(axis=1)).ravel()
    average_length = max(lengths.mean(), 1)
    tfidf = TfidfTransformer().fit(dtf)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=100000, dtype=np.float32)
    dchar = char.fit_transform(docs)
    svd = TruncatedSVD(n_components=min(32, dchar.shape[0]-1, dchar.shape[1]-1), random_state=13, n_iter=3).fit(dchar)
    def apply(target):
        qs, ds, qi, di = scoring.lookups(target)
        dt, qt = cv.transform(ds).tocsr(), cv.transform(qs).tocsr()
        lens = np.asarray(dt.sum(axis=1)).ravel()
        rows = np.repeat(np.arange(len(ds)), np.diff(dt.indptr))
        bm = dt.copy()
        denom = bm.data+1.5*(.25+.75*lens[rows]/average_length)
        bm.data = (idf[bm.indices]*bm.data*2.5/denom).astype(np.float32)
        result = {'raw_bm25': scoring.paired_dot(qt, bm, qi, di),
                  'raw_tfidf': scoring.paired_dot(tfidf.transform(qt), tfidf.transform(dt), qi, di),
                  'raw_lsa': scoring.paired_dot(normalize(svd.transform(char.transform(qs))).astype(np.float32),
                      normalize(svd.transform(char.transform(ds))).astype(np.float32), qi, di)}
        import re
        qset = [set(re.findall(r'[A-Za-z0-9]+', q.lower())) for q in qs]
        dset = [set(re.findall(r'[A-Za-z0-9]+', d.lower())) for d in ds]
        qe, de = [{t for t in s if len(t) > 5} for s in qset], [{t for t in s if len(t) > 5} for s in dset]
        result['raw_entity'] = np.array([len(qe[q]&de[d])/max(len(qe[q]|de[d]), 1) for q, d in zip(qi, di)], np.float32)
        result['raw_relation'] = np.array([len(qset[q]&dset[d])/max(len(qset[q]), 1) for q, d in zip(qi, di)], np.float32)
        return result
    # Reconstruct the development features before applying to new candidates.
    old = apply(reference)
    errors = {k: float(np.max(np.abs(reference[k].to_numpy(float)-v))) for k, v in old.items()}
    assert max(errors.values()) < 1e-4, errors
    DATA.mkdir(parents=True, exist_ok=True)
    joblib.dump({'count': cv, 'tfidf': tfidf, 'char': char, 'svd': svd, 'bm25_idf': idf,
                 'bm25_average_length': average_length}, DATA/'frozen_lexical_models.joblib')
    print('development lexical reconstruction errors', errors, flush=True)
    return apply(frame), errors


def run(smoke, lexical_only):
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    source_sha = digest(Path(__file__))
    input_sha = digest(DATA/'candidates.parquet')
    if not smoke and not lexical_only:
        audit = json.loads((OUT/'smoke_audit.json').read_text())
        assert audit['passed'] and audit['source_sha256'] == source_sha and audit['input_sha256'] == input_sha
    frame = pd.read_parquet(DATA/'candidates.parquet')
    if smoke:
        frame = frame[frame.query_id.isin(sorted(frame.query_id.unique())[:12])].reset_index(drop=True)
    lexical_cache = DATA/('lexical_smoke.npz' if smoke else 'lexical_full.npz')
    if lexical_cache.exists():
        lex = dict(np.load(lexical_cache))
        errors = json.loads((OUT/'lexical_reconstruction.json').read_text())
    else:
        lex, errors = lexical_frozen(frame)
        np.savez_compressed(lexical_cache, **lex)
        (OUT/'lexical_reconstruction.json').write_text(json.dumps(errors, indent=2))
    for k, v in lex.items():
        frame[k] = v
    if lexical_only:
        print('lexical check passed', len(frame), 'rows', round(time.monotonic()-start, 1), 's', flush=True)
        return
    for key in ['minilm', 'e5', 'bge']:
        print('embedding', key, flush=True)
        frame['raw_'+key] = scoring.embeddings(frame, key)
    for key in ['ce', 'nli']:
        print('paired scorer', key, flush=True)
        for k, v in scoring.neural_pairs(frame, key).items():
            frame[k] = v
    assert np.isfinite(frame.filter(regex='^raw_').to_numpy(float)).all()
    target = DATA/('scored_smoke.parquet' if smoke else 'scored.parquet')
    frame.to_parquet(target, index=False)
    audit = {'passed': True, 'smoke': smoke, 'source_sha256': source_sha, 'input_sha256': input_sha,
             'output_sha256': digest(target), 'rows': len(frame), 'lexical_transform_fit_on_development_only': True,
             'lexical_reconstruction_max_errors': errors, 'neural_models': scoring.MODEL_REVISIONS,
             'outcome_use': 'Annotation columns are retained for later evaluation; no scoring function receives them as model inputs.',
             'elapsed_seconds': time.monotonic()-start}
    (OUT/('smoke_audit.json' if smoke else 'audit.json')).write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--lexical-only', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=2):
        run(args.smoke, args.lexical_only)
