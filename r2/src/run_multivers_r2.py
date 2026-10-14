"""Official MultiVerS inference and calibrated SciFact sentence admission."""
from pathlib import Path
import argparse, ast, hashlib, json, os, sys, time, types
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LongformerConfig, LongformerModel
from huggingface_hub import snapshot_download

R2=Path(__file__).resolve().parents[1]
VENDOR=R2/'src/vendor/mv/multivers';sys.path.insert(0,str(VENDOR))
from allennlp_feedforward import FeedForward
from allennlp_nn_util import batched_index_select
from data import MultiVerSDataset, Collator, get_tokenizer
from r2_admission import PROTOCOL, fit_threshold, summarize, score_pipeline
from r2_statistics import query_statistics
from strengthen_r2_analysis import paired_interval, budget_rule

MODEL=R2/'data/models/multivers'
OUT=R2/'results/strengthening/multivers'
OUT.mkdir(exist_ok=True)


def compatibility_modules():
    # Only checkpoint metadata refers to historical Lightning symbols.
    # Inference invokes the official forward method without a training engine.
    names=['pytorch_lightning','pytorch_lightning.callbacks',
           'pytorch_lightning.callbacks.model_checkpoint','pytorch_lightning.utilities',
           'pytorch_lightning.utilities.argparse']
    for name in names:sys.modules[name]=types.ModuleType(name)
    sys.modules[names[2]].ModelCheckpoint=type('ModelCheckpoint',(),{})
    sys.modules[names[4]]._gpus_arg_default=lambda:None


def original_forward():
    tree=ast.parse((VENDOR/'model.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiVerSModel')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    scope={'torch':torch,'F':F,'batched_index_select':batched_index_select}
    exec(compile(ast.Module(body=[forward],type_ignores=[]),str(VENDOR/'model.py'),'exec'),scope)
    return scope['forward']


class Native(nn.Module):
    forward=original_forward()
    def __init__(self,config):
        super().__init__();self.encoder=LongformerModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        h=config.hidden_size
        self.label_classifier=FeedForward(h,2,[h,3],[nn.GELU(),nn.Identity()],[self.dropout.p,0])
        self.rationale_classifier=FeedForward(2*h,2,[h,1],[nn.GELU(),nn.Identity()],[self.dropout.p,0])
        self.nei_label=1;self.label_threshold=None;self.rationale_threshold=.5


def load_model():
    compatibility_modules()
    ckpt=torch.load(MODEL/'fever_sci.ckpt',map_location='cpu',mmap=True,weights_only=False)
    state=ckpt['state_dict']
    folder=Path(snapshot_download('allenai/longformer-large-4096',revision='b190dd42e462d2dee634d1162c839710079f97ab',
        allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json']))
    config=LongformerConfig.from_pretrained(folder)
    config.vocab_size=state['encoder.embeddings.word_embeddings.weight'].shape[0]
    model=Native(config)
    ignored=[k for k in state if k.endswith('position_ids')]
    result=model.load_state_dict({k:v for k,v in state.items() if k not in ignored},strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model=model.to('cuda').eval()
    # The official function adds the same ten tokens to this pinned tokenizer.
    import data as official_data
    official_auto=official_data.AutoTokenizer
    class PinnedAuto:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            return official_auto.from_pretrained(folder,**kwargs)
    official_data.AutoTokenizer=PinnedAuto
    try:tokenizer=get_tokenizer()
    finally:official_data.AutoTokenizer=official_auto
    assert len(tokenizer)==config.vocab_size
    record=json.loads((MODEL/'checkpoint_manifest.json').read_text())
    record.update(encoder_revision=folder.name,ignored_nonpersistent_buffers=ignored,
        tokenizer_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()},
        strict_parameter_load=True,official_forward_sha256=hashlib.sha256((VENDOR/'model.py').read_bytes()).hexdigest(),
        input='claim, title, all abstract sentences; official EOS and global attention',
        score='rationale probability times (1 - document NOT ENOUGH INFO probability)',
        torch=torch.__version__,precision='float16 autocast',batching='Length ordering; at most 8192 padded tokens; reduce batch after memory pressure')
    (MODEL/'inference_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return model,tokenizer


def inference(smoke):
    model,tokenizer=load_model();start=time.monotonic()
    frame=pd.read_parquet(R2/'data/processed/scifact_full_k100.parquet')
    frame['doc_id']=frame.candidate_id.str.split('::').str[0]
    corpus_path=R2/'data/raw/scifact/corpus.jsonl'
    if not corpus_path.exists():corpus_path=R2.parent/'data/raw_open/evidence_retrieval/scifact/corpus.jsonl'
    corpus={str(x['doc_id']):x for x in [json.loads(l) for l in corpus_path.read_text(encoding='utf-8').splitlines()]}
    entries=[]
    for (qid,did),part in frame.groupby(['query_id','doc_id'],sort=True):
        document=corpus[did]
        entries.append({'claim_id':int(qid),'abstract_id':int(did),'to_tensorize':{
            'claim':part.query_text.iloc[0],'title':document['title'],'sentences':document['abstract']}})
    if smoke:entries=entries[:8]
    path=OUT/('document_scores_smoke_adaptive.jsonl' if smoke else 'document_scores.jsonl')
    done={}
    if path.exists() and not smoke:
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);done[(row['query_id'],row['doc_id'])]=row
    dataset=MultiVerSDataset(entries,tokenizer);collator=Collator(tokenizer)
    items=[];truncated=0
    for i,entry in enumerate(entries):
        if (str(entry['claim_id']),str(entry['abstract_id'])) in done:continue
        item=dataset[i]
        if len(item['tokenized'].input_ids)>4096:
            item['tokenized'],item['abstract_sent_idx']=dataset._tokenize_truncated(**entry['to_tensorize']);truncated+=1
        items.append(item)
    items.sort(key=lambda x:len(x['tokenized'].input_ids))
    batch_limit=16;processed=0
    with path.open('w' if smoke else 'a',encoding='utf-8') as stream,torch.inference_mode():
        while processed<len(items):
            window=max(512,((len(items[processed]['tokenized'].input_ids)+511)//512)*512)
            batch_size=min(batch_limit,max(1,8192//window))
            batch=items[processed:processed+batch_size]
            tensor=collator(batch)
            inputs={k:v.to('cuda') for k,v in tensor['tokenized'].items()}
            try:
                with torch.autocast('cuda',dtype=torch.float16):
                    res=model(inputs,tensor['abstract_sent_idx'].to('cuda'))
            except torch.cuda.OutOfMemoryError:
                assert batch_size>1
                batch_limit=batch_size//2;torch.cuda.empty_cache();continue
            labels=res['label_probs'].float().cpu().numpy()
            rationale=res['rationale_probs'].float().cpu().numpy()
            assert np.isfinite(labels).all() and np.isfinite(rationale).all()
            for j,item in enumerate(batch):
                n=len(item['abstract_sent_idx'])
                row={'query_id':str(item['claim_id']),'doc_id':str(item['abstract_id']),
                     'label_probabilities':labels[j].tolist(),'rationale_probabilities':rationale[j,:n].tolist()}
                stream.write(json.dumps(row)+'\n')
            stream.flush();processed+=len(batch)
            if processed%50<batch_size or smoke:
                print(processed,'/',len(items),'documents',round(time.monotonic()-start,1),'s',flush=True)
    record={'smoke':smoke,'documents':len(entries),'new_documents':len(items),'truncated_documents':truncated,
            'all_scores_finite':True,'elapsed_seconds':time.monotonic()-start}
    (OUT/('smoke_audit.json' if smoke else 'inference_audit.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')


def evaluate():
    frame=pd.read_parquet(R2/'data/scored/scifact_full_k100.parquet')
    queries=json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    docs={(x['query_id'],x['doc_id']):x for x in [json.loads(l) for l in (OUT/'document_scores.jsonl').read_text().splitlines()]}
    values=[]
    for q,c in zip(frame.query_id,frame.candidate_id):
        did,sid=c.split('::');r=docs[(str(q),did)]
        values.append(r['rationale_probabilities'][int(sid)]*(1-r['label_probabilities'][1]))
    frame['raw_multivers']=values
    frame[['query_id','candidate_id','raw_multivers']].to_parquet(OUT/'candidate_scores.parquet',index=False)
    results=[];stats={};batch_results=[];batch_stats={}
    for seed in PROTOCOL['seeds']:
        root=R2/'results/experiments/scifact_full_k100'
        model=json.loads((root/f'model_seed{seed}.json').read_text())
        test=frame[frame.query_id.isin(model['evaluation_query_ids'])].copy().reset_index(drop=True)
        cal=frame[frame.query_id.isin(model['threshold_query_ids'])].copy().reset_index(drop=True)
        qtest={q:queries[q] for q in model['evaluation_query_ids']}
        _,stress,_=score_pipeline(test,model)
        rule=fit_threshold(cal,cal.raw_multivers.to_numpy(float),100)
        s=query_statistics(test,test.raw_multivers.to_numpy(float),rule,stress,qtest)
        s.to_csv(OUT/f'query_seed{seed}_multivers.csv',index=False)
        results.append({'dataset':'scifact','seed':seed,'method':'multivers',**summarize(s)})
        a=pd.read_csv(root/f'query_seed{seed}_scet.csv',dtype={'query_id':str})
        stats[seed]={'scet':a,'multivers':s}
        batch=query_statistics(test,test.raw_multivers.to_numpy(float),budget_rule(test,test.raw_multivers.to_numpy(float),len(qtest)),stress,qtest)
        assert batch.admitted.sum()==len(qtest)
        batch.to_csv(OUT/f'query_seed{seed}_multivers_exact_budget.csv',index=False)
        batch_results.append({'dataset':'scifact','seed':seed,'method':'multivers',**summarize(batch)})
        batch_scet=pd.read_csv(R2/f'results/strengthening/scifact_full_k100/query_seed{seed}_scet.csv',dtype={'query_id':str})
        batch_stats[seed]={'scet':batch_scet,'multivers':batch}
        (OUT/f'rule_seed{seed}.json').write_text(json.dumps(rule,indent=2),encoding='utf-8')
    pd.DataFrame(results).to_csv(OUT/'primary.csv',index=False)
    (OUT/'paired_interval.json').write_text(json.dumps(paired_interval(stats,'multivers'),indent=2),encoding='utf-8')
    pd.DataFrame(batch_results).to_csv(OUT/'exact_budget.csv',index=False)
    (OUT/'exact_budget_interval.json').write_text(json.dumps(paired_interval(batch_stats,'multivers'),indent=2),encoding='utf-8')
    print(pd.DataFrame(results)[['seed','evidence_precision','volume_per_100','evidence_recall']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--evaluate-only',action='store_true');a=p.parse_args()
    if a.evaluate_only:evaluate()
    else:
        if not a.smoke:assert json.loads((OUT/'smoke_audit.json').read_text())['all_scores_finite']
        inference(a.smoke)
        if not a.smoke:evaluate()
