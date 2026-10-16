"""Label-free confirmation scoring with the original frozen lexical reference."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import pickle
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import r2_scoring as scoring
import r5_context_gain as development

R = Path(__file__).resolve().parents[1]
D = R/'data/independent_confirmation'
O = R/'results/independent_confirmation'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def reference_lexical():
    target=D/'frozen_lexical.pkl'
    if target.exists():
        with target.open('rb') as h:return pickle.load(h)
    original=R/'data/processed/fever_full_k100.parquet'
    frame=pd.read_parquet(original,columns=['query_id','query_text','evidence_text'])
    _,docs,_,_=scoring.lookups(frame)
    print('Reconstructing lexical reference',len(docs),'sentences',flush=True)
    count=CountVectorizer(token_pattern=r'(?u)\b\w+\b',max_features=150000,dtype=np.float32)
    dtf=count.fit_transform(docs).tocsr()
    n=dtf.shape[0]
    df=np.bincount(dtf.indices,minlength=dtf.shape[1])
    idf=np.log(1+(n-df+.5)/(df+.5))
    mean_length=float(np.asarray(dtf.sum(axis=1)).mean())
    tfidf=TfidfTransformer().fit(dtf)
    del dtf
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=100000,dtype=np.float32)
    mat=char.fit_transform(docs)
    svd=TruncatedSVD(n_components=min(32,mat.shape[0]-1,mat.shape[1]-1),random_state=13,n_iter=3)
    svd.fit(mat)
    result={'count':count,'idf':idf,'mean_length':mean_length,'tfidf':tfidf,'char':char,'svd':svd,
            'reference_input_sha256':sha(original),'reference_sentence_count':n}
    with target.open('wb') as h:pickle.dump(result,h)
    check=frame.iloc[:min(1000,len(frame))].reset_index(drop=True)
    actual=lexical(check,result)
    expected=pd.read_parquet(R/'data/scored/fever_full_k100.parquet',columns=list(actual)).iloc[:len(check)]
    differences={c:float(np.max(np.abs(actual[c]-expected[c].to_numpy()))) for c in actual}
    assert all(v<2e-4 for v in differences.values()),differences
    dump(O/'lexical_reference_audit.json',{'passed':True,'maximum_absolute_difference':differences,
         'lexical_state_sha256':sha(target),'reference_input_sha256':sha(original),
         'new_confirmation_candidates_used_for_fitting':False})
    print('Lexical reconstruction verified',differences,flush=True)
    return result

def lexical(frame,state):
    queries,docs,qi,di=scoring.lookups(frame)
    dtf=state['count'].transform(docs).tocsr()
    qtf=state['count'].transform(queries).tocsr()
    lengths=np.asarray(dtf.sum(axis=1)).ravel()
    rows=np.repeat(np.arange(len(docs)),np.diff(dtf.indptr))
    bm=dtf.copy()
    denom=bm.data+1.5*(.25+.75*lengths[rows]/max(state['mean_length'],1))
    bm.data=(state['idf'][bm.indices]*bm.data*2.5/denom).astype(np.float32)
    result={'raw_bm25':scoring.paired_dot(qtf,bm,qi,di)}
    result['raw_tfidf']=scoring.paired_dot(state['tfidf'].transform(qtf),state['tfidf'].transform(dtf),qi,di)
    dlsa=normalize(state['svd'].transform(state['char'].transform(docs))).astype(np.float32)
    qlsa=normalize(state['svd'].transform(state['char'].transform(queries))).astype(np.float32)
    result['raw_lsa']=scoring.paired_dot(qlsa,dlsa,qi,di)
    import re
    qt=[set(re.findall(r'[A-Za-z0-9]+',v.lower())) for v in queries]
    dt=[set(re.findall(r'[A-Za-z0-9]+',v.lower())) for v in docs]
    qe=[{x for x in s if len(x)>5} for s in qt]
    de=[{x for x in s if len(x)>5} for s in dt]
    result['raw_entity']=np.asarray([len(qe[q]&de[d])/max(len(qe[q]|de[d]),1) for q,d in zip(qi,di)],np.float32)
    result['raw_relation']=np.asarray([len(qt[q]&dt[d])/max(len(qt[q]),1) for q,d in zip(qi,di)],np.float32)
    return result

def pairs(frame,key,out):
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    path,revision=scoring.snapshot(scoring.MODELS[key])
    cache=out/f'{key}_pairs.npz'
    digest=scoring.pair_digest(frame)
    if cache.exists():
        record=json.loads(cache.with_suffix('.json').read_text())
        assert record['input_digest']==digest and record['revision']==revision
        return dict(np.load(cache))
    tokenizer=AutoTokenizer.from_pretrained(str(path))
    model=AutoModelForSequenceClassification.from_pretrained(str(path),torch_dtype=torch.float16).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    queries,docs,qi,di=scoring.lookups(frame)
    ql=np.asarray([len(v) for v in tokenizer(queries,add_special_tokens=False,truncation=False)['input_ids']])
    dl=np.asarray([len(v) for v in tokenizer(docs,add_special_tokens=False,truncation=False)['input_ids']])
    lengths=ql[qi]+dl[di]+tokenizer.num_special_tokens_to_add(pair=True)
    order=np.argsort(lengths,kind='stable')
    partial=out/f'{key}_pairs.partial.npy'
    values=np.load(partial) if partial.exists() else np.full((len(frame),model.config.num_labels),np.nan,np.float32)
    pending=order[~np.isfinite(values[order,0])]
    start=time.monotonic();last=start;offset=0;batch=128 if key=='ce' else 256
    with torch.inference_mode():
        while offset<len(pending):
            indices=pending[offset:offset+batch]
            first=[docs[di[i]] if key=='nli' else queries[qi[i]] for i in indices]
            second=[queries[qi[i]] if key=='nli' else docs[di[i]] for i in indices]
            encoded=tokenizer(first,second,padding=True,truncation='longest_first',max_length=256,return_tensors='pt')
            try:
                logits=model(**{k:v.to(model.device) for k,v in encoded.items()}).logits.float()
                values[indices]=(torch.softmax(logits,dim=-1) if key=='nli' else logits).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();batch//=2
                assert batch>=8
                continue
            offset+=len(indices)
            if time.monotonic()-last>=30:
                np.save(partial,values);last=time.monotonic()
                print(key,offset,'/',len(pending),round(offset/(last-start)),'pairs/s',flush=True)
    assert np.isfinite(values).all()
    if key=='ce':result={'raw_ce':values[:,0]}
    else:
        mapping={v.lower():int(k) for k,v in model.config.id2label.items()}
        ent=values[:,mapping['entailment']];con=values[:,mapping['contradiction']]
        result={'raw_nli_relevance':ent+con,'raw_nli_directional':ent-con,'raw_nli_entailment':ent}
    np.savez_compressed(cache,**result)
    dump(cache.with_suffix('.json'),{'model':scoring.MODELS[key],'revision':revision,'input_digest':digest,
         'pairs':len(frame),'truncated_pairs':int((lengths>256).sum()),'max_length':256,
         'pair_order':'evidence,claim' if key=='nli' else 'claim,evidence',
         'id2label':model.config.id2label,'transformers':transformers.__version__,'torch':torch.__version__})
    del model,tokenizer
    gc.collect();torch.cuda.empty_cache()
    return result

def run(stage,input_path):
    import torch
    torch.set_num_threads(2)
    D.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
    out=D/'scores'/stage;out.mkdir(parents=True,exist_ok=True)
    scoring.CACHE=out/'cache';scoring.CACHE.mkdir(exist_ok=True)
    assert input_path.exists(),str(input_path)
    frame=pd.read_parquet(input_path)
    assert not any(c.startswith('gold') or c in ['claim_verdict','label'] for c in frame)
    identity={'scoring_source_sha256':sha(__file__),'input_sha256':sha(input_path),
              'models':{p.name:sha(p) for p in (development.O/'development').glob('model_*.pkl')},
              'model_selection_permitted':False,'evaluation_annotations_read':False}
    freeze=out/'before_scoring.json'
    if freeze.exists():assert json.loads(freeze.read_text(encoding='utf-8'))==identity
    else:dump(freeze,identity)
    if stage!='smoke':
        smoke=json.loads((O/'scoring_smoke_audit.json').read_text())
        assert smoke['passed'] and smoke['scoring_source_sha256']==sha(__file__)
    state=reference_lexical()
    cache=out/'lexical.npz'
    lex=dict(np.load(cache)) if cache.exists() else lexical(frame,state)
    if not cache.exists():np.savez_compressed(cache,**lex)
    for col,v in lex.items():frame[col]=v
    for key in ['minilm','e5','bge']:
        print('Embedding',stage,key,flush=True)
        frame['raw_'+key]=scoring.embeddings(frame,key)
    for key in ['ce','nli']:
        print('Pair scoring',stage,key,flush=True)
        for col,v in pairs(frame,key,out).items():frame[col]=v
    assert np.isfinite(frame[development.a.VIEWS].to_numpy()).all()
    target=out/'scored.parquet';frame.to_parquet(target,index=False)
    x,keep=development.list_inputs(frame)
    screened=frame.loc[keep,['query_id','candidate_id','tie_key']].copy()
    checks=[]
    for seed in [13,17,29,7]:
        for group in ['ce','all']:
            with (development.O/f'development/model_seed{seed}_{group}.pkl').open('rb') as h:model=pickle.load(h)
            score=development.predict(model,x[keep])
            np.testing.assert_array_equal(score,development.predict(model,x[keep][::-1])[::-1])
            assert np.isfinite(score).all()
            screened[f'score_{seed}_{group}']=score
            checks.append(f'{seed}_{group}')
    screened.to_parquet(out/'frozen_predictions.parquet',index=False)
    audit={'passed':True,'stage':stage,**identity,'scored_sha256':sha(target),
           'lexical_reference_sha256':sha(D/'frozen_lexical.pkl'),
           'queries':frame.query_id.nunique(),'candidate_rows':len(frame),'screened_rows':len(screened),
           'finite_and_order_invariant_models':checks,'input_contains_labels':False}
    dump(O/f'scoring_{stage}_audit.json',audit)
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','calibration','evaluation'],required=True)
    p.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=2):run(args.stage,args.input)
