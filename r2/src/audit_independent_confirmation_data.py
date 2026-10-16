"""Describe new-claim confirmation availability without fitting or evaluating models."""
from pathlib import Path
import hashlib, json, re, collections
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT
OUT = R2 / 'results/independent_confirmation'

def records(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def ids(row):
    return {f'{p[2]}::{p[3]}' for group in row.get('evidence', []) for p in group
            if len(p) >= 4 and p[2] is not None and p[3] is not None}

def normal(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))

def run():
    trainpath = ROOT/'data/raw_open/evidence_retrieval/fever/train.jsonl'
    devpath = ROOT/'data/raw_open/evidence_retrieval/fever/shared_task_dev.jsonl'
    train, dev = records(trainpath), records(devpath)
    oldids = {str(r['id']) for r in dev}
    oldtexts = {normal(r['claim']) for r in dev}
    devpages = {c.rsplit('::',1)[0] for r in dev for c in ids(r)}
    oldcands = set(pd.read_parquet(R2/'data/processed/fever_full_k100.parquet',
                                 columns=['candidate_id']).candidate_id)
    oldpages = {c.rsplit('::',1)[0] for c in oldcands}
    def count(rows):
        return {'queries':len(rows),'labels':dict(collections.Counter(r['label'] for r in rows)),
                'annotated_pairs':sum(len(ids(r)) for r in rows),
                'unique_annotated_pages':len({c.rsplit('::',1)[0] for r in rows for c in ids(r)})}
    unique = [r for r in train if str(r['id']) not in oldids and normal(r['claim']) not in oldtexts]
    devdisjoint = [r for r in unique if not {c.rsplit('::',1)[0] for c in ids(r)} & devpages]
    canddisjoint = [r for r in unique if not {c.rsplit('::',1)[0] for c in ids(r)} & oldpages]
    cache = set(pd.read_parquet(R2/'data/raw/fever/verified_sentences.parquet',columns=['candidate_id']).candidate_id)
    available = [r for r in unique if ids(r) and ids(r) <= cache]
    result = {
        'purpose':'Data availability and overlap audit; no prediction outcomes or fitted models were computed.',
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [trainpath,devpath]},
        'official_train':count(train),'official_dev':count(dev),
        'new_id_and_normalized_claim':count(unique),
        'new_claim_and_no_annotated_page_shared_with_dev':count(devdisjoint),
        'new_claim_and_no_annotated_page_shared_with_any_previous_k100_candidate':count(canddisjoint),
        'new_claim_all_gold_pairs_already_text_verified':count(available),
        'old_dev_annotated_pages':len(devpages),'old_k100_candidate_pages':len(oldpages),
        'old_k100_candidate_sentence_ids':len(oldcands),'verified_cache_sentence_ids':len(cache),
        'train_gold_pair_overlap_with_old_candidate_sentence_ids':len({c for r in train for c in ids(r)} & oldcands),
        'note':'Page exclusion uses evidence annotation only for the overlap audit. A new experiment should select claims by identifier and freeze predictions before joining evaluation labels.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'data_availability_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': run()
