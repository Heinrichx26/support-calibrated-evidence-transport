"""Same-checkpoint MultiVerS inference with a single candidate sentence."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from run_multivers_r2 import load_model, MultiVerSDataset, Collator

R2=Path(__file__).resolve().parents[1]
OUT=R2/'results/multivers_extension/inference_single'


def run(smoke):
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    model,tokenizer=load_model(OUT/'model_manifest.json')
    frame=pd.read_parquet(R2/'data/scored/scifact_full_k100.parquet')
    corpus={str(x['doc_id']):x for x in map(json.loads,(R2/'data/raw/scifact/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    if smoke:
        frame=frame.iloc[:64].copy()
    destination=OUT/('scores_smoke.jsonl' if smoke else 'scores.jsonl')
    done=set()
    if destination.exists() and not smoke:
        done={(x['query_id'],x['candidate_id']) for x in map(json.loads,destination.read_text().splitlines())}
    entries=[]
    for qid,cid,claim in frame[['query_id','candidate_id','query_text']].itertuples(index=False,name=None):
        if (qid,cid) in done:
            continue
        did,sid=cid.split('::');document=corpus[did]
        entries.append({'claim_id':int(qid),'abstract_id':len(entries), 'candidate_id':cid,
            'to_tensorize':{'claim':claim,'title':document['title'],'sentences':[document['abstract'][int(sid)]]}})
    dataset=MultiVerSDataset(entries,tokenizer);collator=Collator(tokenizer)
    items=[]
    for i,entry in enumerate(entries):
        item=dataset[i];item['candidate_id']=entry['candidate_id']
        assert len(item['abstract_sent_idx'])==1
        assert len(item['tokenized'].input_ids)<=4096
        items.append(item)
    items.sort(key=lambda x:len(x['tokenized'].input_ids))
    processed=0;batch_limit=32
    with destination.open('w' if smoke else 'a',encoding='utf-8') as stream,torch.inference_mode():
        while processed<len(items):
            window=max(512,((len(items[processed]['tokenized'].input_ids)+511)//512)*512)
            batch_size=min(batch_limit,max(1,16384//window))
            batch=items[processed:processed+batch_size]
            tensor=collator([{k:v for k,v in item.items() if k!='candidate_id'} for item in batch])
            inputs={k:v.to('cuda') for k,v in tensor['tokenized'].items()}
            try:
                with torch.autocast('cuda',dtype=torch.float16):
                    result=model(inputs,tensor['abstract_sent_idx'].to('cuda'))
            except torch.cuda.OutOfMemoryError:
                assert batch_size>1
                batch_limit=batch_size//2;torch.cuda.empty_cache();continue
            label=result['label_probs'].float().cpu().numpy()
            rationale=result['rationale_probs'].float().cpu().numpy()
            assert np.isfinite(label).all() and np.isfinite(rationale).all()
            for j,item in enumerate(batch):
                score=float(rationale[j,0]*(1-label[j,1]))
                stream.write(json.dumps({'query_id':str(item['claim_id']),'candidate_id':item['candidate_id'],
                    'raw_multivers':score,'label_probabilities':label[j].tolist(),'rationale_probability':float(rationale[j,0])})+'\n')
            stream.flush();processed+=len(batch)
            if processed%500<batch_size or smoke:
                print(processed,'/',len(items),'sentences',round(time.monotonic()-start,1),'seconds',flush=True)
    rows=[json.loads(line) for line in destination.read_text().splitlines()]
    scores=pd.DataFrame(rows)
    assert not scores.duplicated(['query_id','candidate_id']).any()
    assert len(scores)==len(frame)
    assert scores.raw_multivers.between(0,1).all()
    scores[['query_id','candidate_id','raw_multivers']].to_parquet(OUT/('candidate_scores_smoke.parquet' if smoke else 'candidate_scores.parquet'),index=False)
    manifest=json.loads((OUT/'model_manifest.json').read_text())
    manifest['input']='claim, title, a single candidate sentence; official tokenizer, collator and forward'
    (OUT/'model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    audit={'smoke':smoke,'pairs':len(frame),'scores_finite':True,'scores_in_unit_interval':True,
        'duplicate_pairs':False,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seconds':time.monotonic()-start}
    (OUT/('smoke_audit.json' if smoke else 'audit.json')).write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(audit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not a.smoke:assert json.loads((OUT/'smoke_audit.json').read_text())['scores_finite']
    run(a.smoke)
