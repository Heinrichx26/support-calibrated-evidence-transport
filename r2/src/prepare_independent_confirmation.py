"""Prepare source-separated, new-claim FEVER confirmation inputs without scores."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, hashlib, json, re, sys, time
from zipfile import ZipFile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT
OUT = R2/'data/independent_confirmation'
RESULT = R2/'results/independent_confirmation'
sys.path.insert(0,str(ROOT/'src'))
import ipm_evidence_dataset_adapters as old

SALT = '20261016:independent-source-confirmation:v1'
LABELS = ['SUPPORTS','REFUTES','NOT ENOUGH INFO']
TARGETS = {'smoke':20,'calibration':200,'evaluation':1000}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def h(text): return hashlib.sha256((SALT+':'+str(text)).encode()).hexdigest()
def normal(text): return ' '.join(re.findall(r'[a-z0-9]+',text.lower()))
def pageids(row): return {c.rsplit('::',1)[0] for c in old.fever_evidence_pairs(row)}
def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def freeze():
    OUT.mkdir(exist_ok=True,parents=True); RESULT.mkdir(exist_ok=True,parents=True)
    path=OUT/'frozen_sample_plan.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    raw=ROOT/'data/raw_open/evidence_retrieval/fever'
    train,dev=old.read_jsonl(raw/'train.jsonl'),old.read_jsonl(raw/'shared_task_dev.jsonl')
    oldids={str(r['id']) for r in dev}; oldtexts={normal(r['claim']) for r in dev}
    oldframe=pd.read_parquet(R2/'data/processed/fever_full_k100.parquet',columns=['candidate_id'])
    oldpages={c.rsplit('::',1)[0] for c in oldframe.candidate_id}
    unique={}
    for row in sorted(train,key=lambda r:h(r['id'])):
        key=normal(row['claim'])
        if str(row['id']) in oldids or key in oldtexts or pageids(row)&oldpages: continue
        unique.setdefault(key,row)
    eligible=list(unique.values())
    # Connected annotation-page groups are assigned as a whole before stratified sampling.
    parent={str(r['id']):str(r['id']) for r in eligible}
    def find(q):
        while parent[q]!=q: parent[q]=parent[parent[q]];q=parent[q]
        return q
    owner={}
    for row in eligible:
        q=str(row['id'])
        for page in sorted(pageids(row)):
            if page in owner:
                a,b=find(q),find(owner[page]);parent[max(a,b)]=min(a,b)
            else: owner[page]=q
    groups={q:find(q) for q in parent}
    assignment={}
    for g in set(groups.values()):
        unit=int(h('group:'+g)[:12],16)%100
        assignment[g]='smoke' if unit<10 else ('calibration' if unit<30 else 'evaluation')
    selected={}; availability={}
    for split,target in TARGETS.items():
        strata={label:sorted([r for r in eligible if r['label']==label and assignment[groups[str(r['id'])]]==split],key=lambda r:h('sample:'+str(r['id']))) for label in LABELS}
        take=min(target,min(len(v) for v in strata.values()))
        availability[split]={label:len(rows) for label,rows in strata.items()}
        selected[split]=[{'query_id':str(r['id']),'claim':r['claim'],'label':r['label'],
                         'group':groups[str(r['id'])],'evidence':r['evidence']} for label in LABELS for r in strata[label][:take]]
    sets={s:{p for r in rows for p in pageids(r)} for s,rows in selected.items()}
    for a in sets:
        assert not sets[a]&oldpages
        for b in sets:
            if a!=b: assert not sets[a]&sets[b]
    texts=[normal(r['claim']) for rows in selected.values() for r in rows]
    assert len(texts)==len(set(texts))
    plan={'created':'2026-10-16','selection_salt':SALT,
        'scope':'Previously unevaluated FEVER training claims; every annotated evidence page is absent from every prior FEVER k100 candidate. The three verdicts have equal counts.',
        'eligibility':'Exclude development query IDs, normalized development claim texts, and any train claim with an annotated page in the original development k100 candidate pool. Deduplicate normalized train claim texts.',
        'grouping':'Connected annotated-page components assigned by SHA256 into 10% smoke, 20% calibration, 70% evaluation, then deterministic per-verdict ID sampling. Evidence-free queries form single-query components.',
        'candidate_protocol':'Reuse original train-plus-development evidence-sentence pool and original lexical candidate selection, including annotated evidence augmentation. Then restore original June 2017 sentence text and exclude unavailable text rows. No scoring outcomes select queries.',
        'scoring_input':'Identifiers, tie key, claim text, restored evidence text, source and provenance only; labels and verdicts stored separately.',
        'targets_per_verdict':TARGETS,'eligible_counts':dict(Counter(r['label'] for r in eligible)),
        'available_per_split':availability,'selected_counts':{s:dict(Counter(r['label'] for r in rows)) for s,rows in selected.items()},
        'old_candidate_page_count':len(oldpages),'new_annotated_page_count':{s:len(p) for s,p in sets.items()},
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [raw/'train.jsonl',raw/'shared_task_dev.jsonl',raw/'wikipedia_sentence_cache.json',R2/'data/processed/fever_full_k100.parquet']},
        'preparation_code_sha256':sha(__file__),'selected':selected}
    write_json(path,plan)
    write_json(RESULT/'sample_plan_summary.json',{k:v for k,v in plan.items() if k!='selected'})
    return plan

def build_pool():
    raw=ROOT/'data/raw_open/evidence_retrieval/fever'
    pool=old.collect_fever_candidate_pool([raw/'train.jsonl',raw/'shared_task_dev.jsonl'],sentence_cache=old.load_fever_sentence_cache(raw))
    return pool,old.build_candidate_token_index(pool),{r['candidate_id']:r for r in pool}

def restore(required):
    target=OUT/'source_sentences.parquet'
    cache=pd.read_parquet(target) if target.exists() else pd.read_parquet(R2/'data/raw/fever/verified_sentences.parquet')
    found=cache[cache.candidate_id.isin(required)].copy()
    missing=required-set(found.candidate_id)
    wanted=defaultdict(set)
    for cid in missing:
        title,sid=cid.rsplit('::',1); wanted[title].add(sid)
    new=[];start=time.monotonic()
    if wanted:
        with ZipFile(R2/'data/raw/fever/wiki-pages.zip') as archive:
            members=[n for n in archive.namelist() if n.startswith('wiki-pages/') and n.endswith('.jsonl')]
            for i,member in enumerate(members):
                with archive.open(member) as handle:
                    for line in handle:
                        page=json.loads(line); title=page['id']
                        if title not in wanted: continue
                        for sentline in page['lines'].split('\n'):
                            fields=sentline.split('\t')
                            if len(fields)<2 or fields[0] not in wanted[title] or not fields[1].strip(): continue
                            text=fields[1].strip()
                            new.append({'candidate_id':f'{title}::{fields[0]}','sentence_text':text,
                                'evidence_text':f'{title.replace("_"," ")}. {text}','evidence_source':title,
                                'text_provenance':'FEVER_JUNE_2017','member':member,
                                'sentence_sha256':hashlib.sha256(text.encode()).hexdigest()})
                if i%15==0: print('snapshot',i+1,'/',len(members),'new',len(new),'seconds',round(time.monotonic()-start,1),flush=True)
    merged=pd.concat([found,pd.DataFrame(new)],ignore_index=True).drop_duplicates('candidate_id',keep='last')
    # Retain prior new-cache entries so the full step can reuse the smoke extraction.
    if target.exists(): merged=pd.concat([pd.read_parquet(target),merged],ignore_index=True).drop_duplicates('candidate_id',keep='last')
    merged.to_parquet(target,index=False)
    return merged.set_index('candidate_id').to_dict('index')

def prepare(split,plan,pool,index,lookup):
    selected=plan['selected'][split]; rows=[];truth={}; pool_order={cid:i for i,cid in enumerate(lookup)}
    for i,item in enumerate(selected):
        q=item['query_id']; claim=item['claim'];gold=old.fever_evidence_pairs(item)
        chosen=old.top_candidates_by_index(claim,pool,index,100)
        merged={r['candidate_id']:r for r in chosen}
        # Pool order preserves original tie handling before the independent final tie key.
        for cid in sorted(gold,key=lambda c:pool_order.get(c,len(pool))):
            if cid in lookup: merged[cid]=lookup[cid]
        chosen=sorted(merged.values(),key=lambda r:(r['candidate_id'] not in gold,-old.lexical_overlap(claim,r['evidence_text'])))[:max(100,len(gold))]
        assert len(chosen)==100
        for c in chosen: rows.append({'query_id':q,'query_text':claim,'candidate_id':c['candidate_id']})
        groups=[sorted({f'{p[2]}::{p[3]}' for p in group if len(p)>=4 and p[2] is not None and p[3] is not None}) for group in item['evidence']]
        truth[q]={'query_text':claim,'claim_verdict':item['label'].replace(' ','_'),'gold_ids':sorted(gold),
            'gold_sets':[g for g in groups if g],'gold_count':len(gold),'annotation_page_group':item['group'],'num_hops':0}
        if i%100==0: print(split,'candidates',i+1,'/',len(selected),flush=True)
    frame=pd.DataFrame(rows);needed=set(frame.candidate_id)|{c for info in truth.values() for c in info['gold_ids']}
    sources=restore(needed); before=len(frame); available=frame.candidate_id.isin(sources)
    excluded=frame[~available].copy(); frame=frame[available].copy()
    for col in ['evidence_text','evidence_source','text_provenance','sentence_sha256']:
        frame[col]=frame.candidate_id.map(lambda cid:sources[cid][col])
    frame['tie_key']=[hashlib.sha256(f'73:fever:{q}:{c}'.encode()).hexdigest()[:16] for q,c in zip(frame.query_id,frame.candidate_id)]
    frame=frame.sort_values(['query_id','tie_key']).reset_index(drop=True)
    labels=frame[['query_id','candidate_id']].copy()
    labels['gold_evidence']=[int(c in set(truth[q]['gold_ids'])) for q,c in zip(frame.query_id,frame.candidate_id)]
    present_counts=frame.query_id.value_counts().to_dict()
    present_pairs=set(zip(frame.query_id,frame.candidate_id))
    for q,info in truth.items():
        info['verified_gold_count']=sum(c in sources for c in info['gold_ids'])
        info['fully_verified']=all(c in sources for c in info['gold_ids'])
        info['present_candidates']=int(present_counts.get(q,0))
    assert frame.query_id.nunique()==len(selected)
    assert frame.evidence_text.str.len().min()>4
    assert not frame.evidence_text.str.match(r'^.*\. Sentence \d+\.$').any()
    assert not frame.duplicated(['query_id','candidate_id']).any()
    assert frame.tie_key.str.len().eq(16).all()
    assert not any(c in frame.columns for c in ['gold_evidence','claim_verdict','gold_ids','evidence'])
    assert labels.gold_evidence.sum()==sum((q,c) in present_pairs for q,info in truth.items() for c in info['gold_ids'])
    frame.to_parquet(OUT/f'{split}_inputs.parquet',index=False)
    labels.to_parquet(OUT/f'{split}_labels.parquet',index=False)
    write_json(OUT/f'{split}_queries.json',truth)
    excluded.to_parquet(OUT/f'{split}_missing_text.parquet',index=False)
    oldpages={c.rsplit('::',1)[0] for c in pd.read_parquet(R2/'data/processed/fever_full_k100.parquet',columns=['candidate_id']).candidate_id}
    candpages={c.rsplit('::',1)[0] for c in frame.candidate_id}
    audit={'passed':True,'split':split,'queries':len(truth),'rows_before_text_exclusion':before,'rows':len(frame),
        'missing_text_rows':len(excluded),'gold_candidate_rows':int(labels.gold_evidence.sum()),
        'gold_pairs_missing_snapshot':sum(i['gold_count']-i['verified_gold_count'] for i in truth.values()),
        'labels':dict(Counter(i['claim_verdict'] for i in truth.values())),
        'candidate_pages':len(candpages),'candidate_pages_shared_with_old_k100':len(candpages&oldpages),
        'annotated_pages_shared_with_old_k100':0,'unlabelled_input':True,
        'source_page_exclusion_asserted_before_selection':True,'sample_plan_sha256':sha(OUT/'frozen_sample_plan.json'),
        'files_sha256':{p.name:sha(p) for p in [OUT/f'{split}_inputs.parquet',OUT/f'{split}_labels.parquet',OUT/f'{split}_queries.json']}}
    write_json(RESULT/f'{split}_data_audit.json',audit);print(json.dumps(audit,ensure_ascii=False),flush=True)
    return audit

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['plan','smoke','full'],default='plan');args=parser.parse_args()
    plan=freeze();print(json.dumps({k:v for k,v in plan.items() if k in ['eligible_counts','selected_counts','new_annotated_page_count']},ensure_ascii=False),flush=True)
    if args.mode=='plan': return
    if args.mode=='full': assert json.loads((RESULT/'smoke_data_audit.json').read_text())['passed']
    pool,index,lookup=build_pool();print('fixed original pool',len(pool),'sentences',flush=True)
    for split in (['smoke'] if args.mode=='smoke' else ['calibration','evaluation']): prepare(split,plan,pool,index,lookup)

if __name__=='__main__': main()
