"""Reconstruct released confirmation predictions, decisions and query outcomes."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import independent_confirmation_evaluate as experiment
import export_independent_confirmation_tables as tables

R=Path(__file__).resolve().parents[1]
O=R/'results/independent_confirmation/release_verification'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def same(a,b,name):
    np.testing.assert_allclose(a,b,atol=2e-12,rtol=2e-12,equal_nan=True,err_msg=name)


def run(smoke):
    started=time.monotonic(); O.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((R/'INDEPENDENT_CONFIRMATION_MANIFEST.json').read_text(encoding='utf-8'))
    files=manifest['files']
    if smoke:files=[row for row in files if row['path'].endswith('.py')][:5]
    for row in files:
        path=R/row['path'];assert path.stat().st_size==row['bytes']
        assert sha(path)==row['sha256'],row['path']
    if not smoke:
        gate=json.loads((O/'smoke.json').read_text(encoding='utf-8'))
        assert gate['passed'] and gate['verifier_sha256']==sha(__file__)
    stage='smoke' if smoke else 'evaluation'
    reference=experiment.O/('evaluation_smoke' if smoke else 'evaluation_full')
    calibration,_=experiment.unlabelled_predictions('calibration')
    predictions,_=experiment.unlabelled_predictions(stage)
    labelsframe=pd.read_parquet(experiment.D/f'{stage}_labels.parquet')
    labels={(str(q),str(c)):int(y) for q,c,y in labelsframe.itertuples(index=False,name=None)}
    queries=json.loads((experiment.D/f'{stage}_queries.json').read_text(encoding='utf-8'))
    rules=pd.read_csv(reference/'thresholds.csv',dtype={'tie_cutoff':str},float_precision='round_trip')
    saved_predictions=pd.read_parquet(reference/'candidate_predictions.parquet')
    table=pd.read_csv(reference/'by_model.csv',float_precision='round_trip')
    collection={}; decisions=values=0
    for (seed,name),(data,score) in predictions.items():
        old=saved_predictions[(saved_predictions.seed==seed)&(saved_predictions.method==name)]
        assert np.array_equal(old.query_id.to_numpy(str),data['qids'][data['codes']])
        assert np.array_equal(old.candidate_id.to_numpy(str),data['cids'])
        np.testing.assert_array_equal(old.score.to_numpy(float),score)
        labelled=experiment.attach(data,labels,queries)
        for protocol in ['prospective','fixed_batch']:
            source,source_score=calibration[(seed,name)] if protocol=='prospective' else (data,score)
            rule=experiment.simple.threshold(source,source_score,100)
            saved=rules[(rules.seed==seed)&(rules.method==name)&(rules.protocol==protocol)]
            assert len(saved)==1
            same(rule['threshold'],saved.iloc[0].threshold,'threshold')
            assert rule['tie_cutoff']==saved.iloc[0].tie_cutoff
            mask=experiment.simple.admitted(data,score,rule)
            if protocol=='fixed_batch':assert mask.sum()==len(data['qids'])
            stat=experiment.simple.statistics(labelled,mask)
            oldstat=pd.read_csv(reference/f'query_seed{seed}_{name}_{protocol}.csv',dtype={'query_id':str})
            assert np.array_equal(stat.query_id.to_numpy(str),oldstat.query_id.to_numpy(str))
            for column in stat:
                if column!='query_id':
                    np.testing.assert_array_equal(stat[column],oldstat[column]);values+=len(stat)
            row=table[(table.seed==seed)&(table.method==name)&(table.protocol==protocol)].iloc[0]
            for key,value in experiment.describe(stat).items():same(value,row[key],key);values+=1
            collection[(seed,name,protocol)]=stat
            decisions+=len(data['qids'])
    interval_dir=O/('smoke_intervals' if smoke else 'full_intervals')
    interval_dir.mkdir(exist_ok=True)
    rebuilt=experiment.intervals(collection,queries,interval_dir,200 if smoke else 2000)
    original=pd.read_csv(reference/'paired_intervals.csv',float_precision='round_trip')
    for column in rebuilt:
        if pd.api.types.is_numeric_dtype(rebuilt[column]):same(rebuilt[column],original[column],column)
        else:assert rebuilt[column].astype(str).equals(original[column].astype(str)),column
    index=json.loads((R/'results/tables/index.json').read_text(encoding='utf-8'))
    for name,spec in tables.specs().items():
        stream=io.StringIO(newline='');csv.writer(stream).writerows([spec['headers']]+spec['rows'])
        path=R/f'results/tables/independent_confirmation/{name}.csv'
        assert stream.getvalue().encode('utf-8')==path.read_bytes(),name
        row=next(row for row in index if row['table']==name)
        assert row['sha256']==sha(path),name
    result={'passed':True,'smoke':smoke,'verifier_sha256':sha(__file__),
        'manifest_members_verified':len(files),'model_candidate_predictions_reconstructed':len(saved_predictions),
        'query_decisions_reconstructed':decisions,'numeric_values_verified':values,
        'all_paired_intervals_reconstructed':len(rebuilt),'primary_intervals':int(rebuilt.primary.sum()),
        'tables_reconstructed_byte_exact':['S36','S37'],'complete_table_index_entries':len(index),
        'pretrained_neural_inference_performed':False,'models_refitted':False,
        'elapsed_seconds':time.monotonic()-started}
    (O/('smoke.json' if smoke else 'full.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true');mode.add_argument('--full',action='store_true')
    with threadpool_limits(limits=2):run(parser.parse_args().smoke)
