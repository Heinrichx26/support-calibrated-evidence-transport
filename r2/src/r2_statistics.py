"""Vectorized query summaries with the same admission semantics as r2_admission."""
import numpy as np
import pandas as pd
from r2_admission import PROTOCOL

_CACHE={}

def context(frame,queries,label):
    key=(id(frame),id(queries),label)
    if key in _CACHE:return _CACHE[key]
    _CACHE.clear()
    ids=list(queries)
    codes=pd.Categorical(frame.query_id,categories=ids).codes.astype(np.int32)
    y=frame[label].to_numpy(np.int8)
    tie=np.fromiter((int(k,16) for k in frame.tie_key),dtype=np.uint64,count=len(frame))
    counts=np.asarray([len(q['support_ids']) if label=='gold_directional' else q['gold_count'] for q in queries.values()])
    ideal=np.array([(1/np.log2(np.arange(1,min(n,10)+1)+1)).sum() for n in counts])
    positions={(q,c):j for j,(q,c) in enumerate(zip(frame.query_id,frame.candidate_id)) if y[j]}
    member_positions=[];set_codes=[];set_sizes=[];set_owners=[]
    for owner,(qid,q) in enumerate(queries.items()):
        groups=[q['support_ids']] if label=='gold_directional' else q['gold_sets']
        for group in groups:
            if not group:continue
            sid=len(set_sizes)
            set_sizes.append(len(set(group)));set_owners.append(owner)
            for candidate in set(group):
                if (qid,candidate) in positions:
                    member_positions.append(positions[qid,candidate]);set_codes.append(sid)
    c={'ids':ids,'codes':codes,'y':y,'tie':tie,'counts':counts,'ideal':ideal,
       'member_positions':np.asarray(member_positions,dtype=np.int64),'set_codes':np.asarray(set_codes,dtype=np.int32),
       'set_sizes':np.asarray(set_sizes,dtype=np.int32),'set_owners':np.asarray(set_owners,dtype=np.int32),
       'queries':queries,'frame':frame}
    _CACHE[key]=c
    return c

def query_statistics(frame,scores,rule,stress,queries,label='gold_evidence'):
    c=context(frame,queries,label);codes=c['codes'];n=len(c['ids']);y=c['y'];score=np.asarray(scores)
    order=np.lexsort((c['tie'],-score,codes))
    ordered=codes[order]
    starts=np.maximum.accumulate(np.where(np.r_[True,ordered[1:]!=ordered[:-1]],np.arange(len(order)),0))
    rank=np.empty(len(order),dtype=np.int16);rank[order]=np.arange(len(order))-starts+1
    cutoff=int(rule['tie_cutoff'],16) if rule['tie_cutoff'] else 0
    admitted=(rank<=rule.get('eligible_rank',PROTOCOL['eligible_rank']))&((score>rule['threshold'])|((score==rule['threshold'])&(c['tie']<=cutoff)))
    def total(values):return np.bincount(codes,weights=values,minlength=n).astype(np.int64)
    issued=total(admitted);correct=total(admitted*y)
    first=np.full(n,32767,dtype=np.int32)
    positive=y.astype(bool)
    np.minimum.at(first,codes[positive],rank[positive])
    reciprocal=np.where(first==32767,0,1/first)
    top=positive&(rank<=10)
    gains=np.bincount(codes[top],weights=1/np.log2(rank[top].astype(np.float64)+1),minlength=n)
    ndcg=np.divide(gains,c['ideal'],out=np.zeros(n),where=c['ideal']>0)
    set_hits=np.bincount(c['set_codes'],weights=admitted[c['member_positions']],minlength=len(c['set_sizes']))
    complete=set_hits==c['set_sizes']
    all_facts=np.bincount(c['set_owners'],weights=complete,minlength=n)>0
    return pd.DataFrame({'query_id':c['ids'],
        'claim_verdict':[q['claim_verdict'] for q in queries.values()],
        'gold_count':c['counts'],'verified_gold_count':[q['verified_gold_count'] for q in queries.values()],
        'fully_verified':[q['fully_verified'] for q in queries.values()],
        'admitted':issued,'correct':correct,'covered':(issued>0).astype(int),'hit':(correct>0).astype(int),
        'ranking_gold_at_5':total(positive&(rank<=5)),'mrr':reciprocal,'ndcg_at_10':ndcg,
        'all_facts':all_facts.astype(int),'stress_admitted':total(admitted*stress),
        'stress_correct':total(y*admitted*stress),'has_error':(total(admitted*(1-y))>0).astype(int)})
