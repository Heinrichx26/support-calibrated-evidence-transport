"""Label-independent inference of a target sentence in adjacent-sentence context."""
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
import r2_admission as admission
from strengthen_r2_analysis import budget_rule
import run_multivers_adaptive_calibration as adaptive

R2 = Path(__file__).resolve().parents[1]
DATA = R2/'data/multivers_local'
OUT = R2/'results/multivers_extension/local_context'
PROTOCOL = R2/'data/multivers_local_protocol.json'
SPEC = json.loads(PROTOCOL.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screens(frame):
    keep = np.zeros(len(frame),bool)
    for _,ix in frame.groupby('pool',sort=False).indices.items():
        part = frame.iloc[ix].reset_index(drop=True)
        for name in ['full_raw','full_product','single_raw','mean:0.25:1','mean:0.5:1','mean:0.75:1']:
            s = adaptive.score({'name':name},part)
            keep[ix[admission.ranks(part,s)<=5]] = True
    return keep


def inputs(smoke):
    frame = adaptive.prepare()
    if smoke:
        groups = sorted(frame.family.unique(),key=lambda g:hashlib.sha256(f"{SPEC['seed']}:{g}".encode()).hexdigest())[:32]
        frame = frame[frame.family.isin(groups)].reset_index(drop=True)
    keep = screens(frame)
    # Read only model inputs and precomputed scores for screening. Outcomes are
    # excluded from the serialized inference input and the transformer inputs.
    selected = frame[keep][['query_id','candidate_id','query_text','doc_id','sentence_id']].drop_duplicates(['query_id','candidate_id'])
    return frame,keep,selected


def infer(smoke):
    import torch
    from run_multivers_r2 import load_model,MultiVerSDataset,Collator
    from multivers_short_attention import set_window
    start = time.monotonic()
    DATA.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if not smoke:
        gate = json.loads((OUT/'smoke_assessment.json').read_text())
        assert gate['passed'] and gate['useful_precision_signal']
        assert gate['inference_source_sha256'] == digest(Path(__file__))
        assert gate['protocol_sha256'] == digest(PROTOCOL)
    frame,keep,selected = inputs(smoke)
    selected.to_parquet(DATA/('smoke_inputs.parquet' if smoke else 'inputs.parquet'),index=False)
    corpus = {str(d['doc_id']):d for d in map(json.loads,(R2/'data/raw/scifact/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    output = OUT/('scores_smoke.jsonl' if smoke else 'scores.jsonl')
    done = set()
    if output.exists() and not smoke:
        done = {(r['query_id'],r['candidate_id']) for r in map(json.loads,output.read_text().splitlines())}
    entries = []
    for r in selected.itertuples(index=False):
        if (r.query_id,r.candidate_id) in done:
            continue
        doc,sid = corpus[r.doc_id],int(r.sentence_id)
        begin,end = max(0,sid-1),min(len(doc['abstract']),sid+2)
        entries.append({'claim_id':int(r.query_id),'abstract_id':len(entries),
            'candidate_id':r.candidate_id,'target_index':sid-begin,
            'to_tensorize':{'claim':r.query_text,'title':doc['title'],'sentences':doc['abstract'][begin:end]}})
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.65)
    model,tokenizer = load_model(OUT/('smoke_model_manifest.json' if smoke else 'model_manifest.json'))
    dataset,collator = MultiVerSDataset(entries,tokenizer),Collator(tokenizer)
    items=[]
    for i,entry in enumerate(entries):
        item=dataset[i]
        assert len(item['tokenized'].input_ids)<4096
        assert entry['target_index']<len(item['abstract_sent_idx'])
        item.update(candidate_id=entry['candidate_id'],target_index=entry['target_index'])
        items.append(item)
    items.sort(key=lambda x:len(x['tokenized'].input_ids))
    processed,limit,last=0,64,time.monotonic()
    with output.open('w' if smoke else 'a',encoding='utf-8') as stream,torch.inference_mode():
        while processed<len(items):
            length=len(items[processed]['tokenized'].input_ids)
            padded=2*max(1,length-1) if length<=256 else max(512,((length+511)//512)*512)
            size=min(limit,max(1,8192//padded))
            batch=items[processed:processed+size]
            tensor=collator([{k:v for k,v in x.items() if k not in ['candidate_id','target_index']} for x in batch])
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
            p=result['rationale_probs'].float().cpu().numpy()
            d=result['label_probs'].float().cpu().numpy()
            assert np.isfinite(p).all() and np.isfinite(d).all()
            for j,item in enumerate(batch):
                stream.write(json.dumps({'query_id':str(item['claim_id']),'candidate_id':item['candidate_id'],
                    'local_rationale':float(p[j,item['target_index']]),'local_evidence':float(1-d[j,1]),
                    'local_support':float(d[j,2]),'local_refute':float(d[j,0]),
                    'target_index':item['target_index'],'attention_window':window})+'\n')
            stream.flush()
            processed+=len(batch)
            if time.monotonic()-last>25 or processed==len(items):
                print(processed,'/',len(items),'local contexts; elapsed',round(time.monotonic()-start,1),'s',flush=True)
                last=time.monotonic()
    rows=pd.read_json(output,lines=True,dtype={'query_id':str,'candidate_id':str})
    assert len(rows)==len(selected) and not rows.duplicated(['query_id','candidate_id']).any()
    audit={'passed':True,'smoke':smoke,'claims':frame.query_id.nunique(),'pairs':len(selected),
        'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),
        'scores_sha256':digest(output),'outcomes_enter_network':False,'elapsed_seconds':time.monotonic()-start}
    (OUT/('smoke_inference_audit.json' if smoke else 'inference_audit.json')).write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


def assess_smoke():
    frame,keep,selected=inputs(True)
    rows=pd.read_json(OUT/'scores_smoke.jsonl',lines=True,dtype={'query_id':str,'candidate_id':str})
    fullframe=frame.copy()
    frame=frame[keep].merge(rows,on=['query_id','candidate_id'],validate='many_to_one',sort=False).reset_index(drop=True)
    results=[]
    for pool in frame.pool.unique():
        p=frame[frame.pool==pool].reset_index(drop=True)
        original=fullframe[fullframe.pool==pool].reset_index(drop=True)
        n=p.query_id.nunique()
        for name in ['full_raw','full_product','single_raw','mean:0.25:1','mean:0.5:1','mean:0.75:1']:
            score=adaptive.score({'name':name},p)
            original_score=adaptive.score({'name':name},original)
            mask,_=admission.apply_threshold(p,score,budget_rule(p,score,n))
            omask,_=admission.apply_threshold(original,original_score,budget_rule(original,original_score,n))
            assert set(zip(p.query_id[mask],p.candidate_id[mask]))==set(zip(original.query_id[omask],original.candidate_id[omask]))
            results.append({'pool':pool,'method':name,'precision':float(p.gold_evidence[mask].mean()),'claims':n})
        local=p.local_rationale.to_numpy(float)
        full=p.raw_multivers_rationale.to_numpy(float)
        single=p.single_rationale.to_numpy(float)
        values={'local_raw':local,'local_product':local*p.local_evidence.to_numpy(float)}
        for w in [.25,.5,.75]:
            values[f'local_full:{w:g}']=w*local+(1-w)*full
            values[f'local_dual:{w:g}']=w*local+(1-w)*(.5*full+.5*single)
        for name,s in values.items():
            mask,_=admission.apply_threshold(p,s,budget_rule(p,s,n))
            assert mask.sum()==n
            results.append({'pool':pool,'method':name,'precision':float(p.gold_evidence[mask].mean()),'claims':n})
    data=pd.DataFrame(results)
    data.to_csv(OUT/'smoke_precision.csv',index=False)
    means=data.groupby('method').precision.mean()
    useful=means[[n for n in means.index if n.startswith('local')]].max()>means['mean:0.5:1']
    audit={'passed':True,'useful_precision_signal':bool(useful),
        'controls_admissions_unchanged':True,'claims':frame.query_id.nunique(),
        'inference_source_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),
        'interpretation':'Exploratory smoke, used only for deciding whether to expand inference.'}
    (OUT/'smoke_assessment.json').write_text(json.dumps(audit,indent=2))
    print(data.pivot(index='method',columns='pool',values='precision').to_string(),flush=True)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--assess-smoke',action='store_true')
    a=p.parse_args()
    if a.assess_smoke:
        assess_smoke()
    else:
        infer(a.smoke)
