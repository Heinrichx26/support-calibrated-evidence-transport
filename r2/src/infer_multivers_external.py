"""Public-dataset inference with annotations excluded from the model inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import pandas as pd

R2=Path(__file__).resolve().parents[1]
DATA=R2/'data/multivers_external'
OUT=R2/'results/multivers_extension/external_inference'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    rows=[]
    audit={}
    for name in ['healthver','covidfact']:
        root=DATA/'data'/name
        corpus={str(d['doc_id']):d for d in map(json.loads,(root/'corpus.jsonl').read_text().splitlines())}
        for split in ['train','dev','test']:
            path=root/f'claims_{split}.jsonl'
            if not path.exists():
                continue
            # Deliberately retain only the three public inference fields.
            claims=[{k:r[k] for k in ['id','claim','doc_ids']} for r in map(json.loads,path.read_text().splitlines())]
            audit[f'{name}:{split}']={'claims':len(claims),'file_sha256':digest(path)}
            for claim in claims:
                q=f"{name}:{split}:{claim['id']}"
                for docid in claim['doc_ids']:
                    d=corpus[str(docid)]
                    for sid,sentence in enumerate(d['abstract']):
                        cid=f'{docid}::{sid}'
                        rows.append({'dataset':name,'split':split,'query_id':q,'query_text':claim['claim'],
                            'doc_id':str(docid),'sentence_id':str(sid),'candidate_id':cid,
                            'evidence_text':sentence,'tie_key':hashlib.sha256(f'83:{q}:{cid}'.encode()).hexdigest()[:16]})
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(['query_id','candidate_id']).any()
    frame.to_parquet(DATA/'inference_candidates.parquet',index=False)
    audit.update(rows=len(frame),outcome_fields_retained=False,archive_sha256=digest(DATA/'official_data.tar.gz'),
        origin='https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/data.tar.gz',
        code_sha256=digest(Path(__file__)))
    (DATA/'inference_preparation.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


def infer(context,smoke):
    import torch
    from run_multivers_r2 import load_model,MultiVerSDataset,Collator
    from multivers_short_attention import set_window
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    if not smoke:
        audit=json.loads((OUT/f'{context}_smoke_audit.json').read_text())
        assert audit['passed'] and audit['source_sha256']==digest(Path(__file__))
        assert audit['input_sha256']==digest(DATA/'inference_candidates.parquet')
    frame=pd.read_parquet(DATA/'inference_candidates.parquet')
    corpus={}
    for name in ['healthver','covidfact']:
        corpus[name]={str(d['doc_id']):d for d in map(json.loads,(DATA/'data'/name/'corpus.jsonl').read_text().splitlines())}
    path=OUT/(context+('_smoke' if smoke else '')+'_scores.jsonl')
    keyname='doc_id' if context=='full' else 'candidate_id'
    done=set()
    if path.exists() and not smoke:
        done={(r['query_id'],r[keyname]) for r in map(json.loads,path.read_text().splitlines())}
    entries=[]
    if context=='full':
        grouped=frame.groupby(['dataset','query_id','doc_id'],sort=True)
        for (name,q,did),part in grouped:
            if (q,did) in done:
                continue
            doc=corpus[name][did]
            entries.append({'claim_id':len(entries),'abstract_id':len(entries),'query_id':q,'output_key':did,
                'to_tensorize':{'claim':part.query_text.iloc[0],'title':doc.get('title',''),'sentences':doc['abstract']}})
    else:
        for r in frame.itertuples(index=False):
            if (r.query_id,r.candidate_id) in done:
                continue
            doc=corpus[r.dataset][r.doc_id]
            entries.append({'claim_id':len(entries),'abstract_id':len(entries),'query_id':r.query_id,'output_key':r.candidate_id,
                'to_tensorize':{'claim':r.query_text,'title':doc.get('title',''),'sentences':[r.evidence_text]}})
    if smoke:
        # Spread checks over both datasets and splits without accessing outcomes.
        entries=entries[::max(1,len(entries)//(18 if context=='full' else 64))][:18 if context=='full' else 64]
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.65)
    model,tokenizer=load_model(OUT/f'{context}_{"smoke_" if smoke else ""}model_manifest.json')
    dataset,collator=MultiVerSDataset(entries,tokenizer),Collator(tokenizer)
    items=[]
    truncated=0
    for i,entry in enumerate(entries):
        item=dataset[i]
        if len(item['tokenized'].input_ids)>4096:
            item['tokenized'],item['abstract_sent_idx']=dataset._tokenize_truncated(**entry['to_tensorize'])
            truncated+=1
        item.update(query_id=entry['query_id'],output_key=entry['output_key'])
        items.append(item)
    items.sort(key=lambda x:len(x['tokenized'].input_ids))
    processed,limit,last=0,96,time.monotonic()
    with path.open('w' if smoke else 'a',encoding='utf-8') as stream,torch.inference_mode():
        while processed<len(items):
            length=len(items[processed]['tokenized'].input_ids)
            padded=2*max(1,length-1) if length<=256 else max(512,((length+511)//512)*512)
            size=min(limit,max(1,(8192 if context=='full' else 12288)//padded))
            batch=items[processed:processed+size]
            tensor=collator([{k:v for k,v in x.items() if k not in ['query_id','output_key']} for x in batch])
            tokens={k:v.to('cuda') for k,v in tensor['tokenized'].items()}
            window=set_window(model,tokens)
            try:
                with torch.autocast('cuda',dtype=torch.float16):
                    result=model(tokens,tensor['abstract_sent_idx'].to('cuda'))
            except torch.cuda.OutOfMemoryError:
                assert size>1
                limit=size//2
                torch.cuda.empty_cache()
                continue
            probs=result['rationale_probs'].float().cpu().numpy()
            labels=result['label_probs'].float().cpu().numpy()
            assert np.isfinite(probs).all() and np.isfinite(labels).all()
            assert np.allclose(labels.sum(axis=1),1,atol=.002)
            for j,item in enumerate(batch):
                n=len(item['abstract_sent_idx'])
                row={'query_id':item['query_id'],keyname:item['output_key'],
                     'label_probabilities':labels[j].tolist(),'attention_window':window}
                if context=='full':
                    row['rationale_probabilities']=probs[j,:n].tolist()
                else:
                    row['single_rationale']=float(probs[j,0])
                stream.write(json.dumps(row)+'\n')
            stream.flush()
            processed+=len(batch)
            if time.monotonic()-last>25 or processed==len(items):
                print(context,processed,'/',len(items),'elapsed',round(time.monotonic()-start,1),'s',flush=True)
                last=time.monotonic()
    records=[json.loads(l) for l in path.read_text().splitlines()]
    expected=len(frame[['query_id',keyname]].drop_duplicates()) if not smoke else len(entries)
    assert len(records)==expected and len({(r['query_id'],r[keyname]) for r in records})==expected
    audit={'passed':True,'smoke':smoke,'context':context,'source_sha256':digest(Path(__file__)),
        'input_sha256':digest(DATA/'inference_candidates.parquet'),'scores_sha256':digest(path),
        'unique_records':expected,'truncated':truncated,'labels_accessed_for_performance':False,
        'elapsed_seconds':time.monotonic()-start}
    (OUT/f'{context}_{"smoke_" if smoke else ""}audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--context',choices=['full','single'],default='full')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.prepare:
        prepare()
    else:
        infer(a.context,a.smoke)
