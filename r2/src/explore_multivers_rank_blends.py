"""Fixed-family rank blends for a leakage-controlled MultiVerS ablation."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import r2_admission as admission
from strengthen_r2_analysis import budget_rule
from explore_multivers_ensemble import load_frame

R2 = Path(__file__).resolve().parents[1]
OUT = R2 / 'results/multivers_extension/rank_blends'
SEEDS = [13, 17, 29, 7]
ALPHAS = [0.25, 0.5, 0.75]


def rank_from_fit(train, values):
    grid = np.linspace(0, 1, 1001)
    quantiles = np.quantile(np.asarray(train, float), grid)
    unique, first = np.unique(quantiles, return_index=True)
    last = np.searchsorted(quantiles, unique, side='right') - 1
    mids = (first + last) / (2 * (len(quantiles) - 1))
    return np.interp(np.asarray(values, float), unique, mids, left=0, right=1)


def score_family(train, apply):
    cols = ['raw_multivers', 'raw_multivers_rationale', 'raw_ce', 'raw_e5',
            'raw_bge', 'raw_minilm', 'raw_nli_entailment', 'raw_nli_relevance']
    r = {name: rank_from_fit(train[name].to_numpy(float), apply[name].to_numpy(float)) for name in cols}
    rationale = r['raw_multivers_rationale']
    mv = r['raw_multivers']
    semantic = np.mean([r['raw_ce'], r['raw_e5'], r['raw_bge']], axis=0)
    nli = r['raw_nli_entailment']
    out = {
        'multivers_rank': mv,
        'rationale_rank': rationale,
        'semantic_rank': semantic,
        'nli_rank': nli,
        'rank_geometric_mv_rationale': np.sqrt(np.clip(mv*rationale, 0, 1)),
        'rank_consensus_mv_rationale': (mv + rationale + np.minimum(mv, rationale)) / 3,
        'rank_consensus_rationale_semantic': (rationale + semantic + np.minimum(rationale, semantic)) / 3,
        'rank_rationale_semantic_gap': rationale * (1 - np.abs(rationale-semantic)),
        'rank_rationale_nli_gap': rationale * (1 - np.abs(rationale-nli)),
    }
    for alpha in ALPHAS:
        out[f'blend_rationale_mv_{alpha:g}'] = alpha*rationale + (1-alpha)*mv
        out[f'blend_rationale_semantic_{alpha:g}'] = alpha*rationale + (1-alpha)*semantic
        out[f'blend_rationale_nli_{alpha:g}'] = alpha*rationale + (1-alpha)*nli
    return out


def precision(frame, score):
    rule = budget_rule(frame, score, frame.query_id.nunique())
    admitted, _ = admission.apply_threshold(frame, score, rule)
    y = frame.gold_evidence.to_numpy(int)
    return float((y*admitted).sum()/max(admitted.sum(), 1)), rule


def inner_select(fitting, seed):
    ids = sorted(fitting.query_id.unique(), key=lambda q: hashlib.sha256(f'{seed}:rank-blend:{q}'.encode()).hexdigest())
    fold_map = {q: i % 4 for i, q in enumerate(ids)}
    folds = fitting.query_id.map(fold_map).to_numpy()
    rows = []
    for fold in range(4):
        train = fitting.loc[folds != fold].reset_index(drop=True)
        valid = fitting.loc[folds == fold].reset_index(drop=True)
        for name, score in score_family(train, valid).items():
            value, _ = precision(valid, score)
            rows.append({'seed': seed, 'fold': fold, 'method': name, 'precision': value})
    cv = pd.DataFrame(rows)
    mean = cv.groupby('method').precision.mean().to_dict()
    selected = max(score_family(fitting.iloc[:1], fitting.iloc[:1]), key=lambda name: (round(mean[name], 12), -list(mean).index(name)))
    return selected, cv


def run(smoke=False):
    frame = load_frame().rename(columns={'mv_rationale_raw': 'raw_multivers_rationale'})
    out = OUT/('smoke' if smoke else 'full'); out.mkdir(parents=True, exist_ok=True)
    rows=[]; cv_rows=[]; selected_rows=[]
    for seed in [13] if smoke else SEEDS:
        model=json.loads((R2/f'results/experiments/scifact_full_k100/model_seed{seed}.json').read_text())
        fitids, testids = model['fitting_query_ids'], model['evaluation_query_ids']
        evalids=sorted(testids)[:12] if smoke else testids
        fitting=frame[frame.query_id.isin(fitids)].reset_index(drop=True)
        test=frame[frame.query_id.isin(evalids)].reset_index(drop=True)
        selected, cv=inner_select(fitting, seed); cv_rows.append(cv)
        selected_rows.append({'seed':seed,'selected':selected})
        fit_scores=score_family(fitting,fitting); test_scores=score_family(fitting,test)
        for name,score in test_scores.items():
            value,rule=precision(test,score)
            rows.append({'seed':seed,'method':name,'selected':int(name==selected),'test_precision':value,'admitted':int(len(evalids)),'threshold':rule['threshold']})
        print('seed',seed,'selected',selected,'cv',float(cv[cv.method==selected].precision.mean()),flush=True)
    data=pd.DataFrame(rows);data.to_csv(out/'candidate_results.csv',index=False)
    data[data.selected==1].to_csv(out/'selected_results.csv',index=False)
    pd.concat(cv_rows).to_csv(out/'inner_cv.csv',index=False);pd.DataFrame(selected_rows).to_csv(out/'selected.csv',index=False)
    audit={'smoke':smoke,'candidate_family':list(score_family(frame.iloc[:1],frame.iloc[:1])),'test_labels_used_for_selection':False,'selection':'four query-disjoint folds inside fitting queries','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(data[data.selected==1][['seed','method','test_precision']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.smoke)
