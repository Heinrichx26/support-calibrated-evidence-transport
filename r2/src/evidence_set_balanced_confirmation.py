"""One finite fitting-only follow-up with equal evidence-set-size stratum weight."""
import argparse
from pathlib import Path
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import evidence_set_confirmation as base

base.O = base.R/'results/evidence_set_confirmation/balanced'
base.PLAN = dict(base.PLAN)
base.PLAN.update({
    'query_model': '100-iteration histogram gradient boosting, depth two, seven leaves, min leaf 30, learning rate .05, L2 10; equal total fitting weight for queries with no evidence, minimum singleton set, and minimum multi-sentence set.',
    'utility': 'Sum of calibrated expected correct sentence probabilities plus beta times the stratum-weighted classifier cumulative output for a complete prefix. The latter is a ranking utility under the fitting weights. A least concave majorant makes marginal utilities nonincreasing.',
    'betas': [0.0, 1.0, 4.0, 16.0],
    'relation_to_previous_attempt': 'A single finite follow-up motivated only by the saved fitting-only failure of unweighted prefix utility; the original failed attempt is retained.',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
})


def fit(data, features, target):
    sizes = np.asarray([min([len(set(s)) for s in data['queries'][qid]['gold_sets'] if s], default=0)
                        for qid in data['qids']])
    strata = np.minimum(sizes, 2)
    count = np.bincount(strata, minlength=3)
    w = 1/count[strata]
    w /= w.mean()
    model = HistGradientBoostingClassifier(learning_rate=.05,max_iter=100,max_depth=2,
        max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,max_bins=63,
        early_stopping=False,random_state=20261016)
    model.fit(features,target,sample_weight=w)
    cal = LogisticRegression(C=1,max_iter=300,solver='lbfgs')
    n = np.bincount(data['codes'],minlength=len(data['qids']))
    cw = 1/n[data['codes']]
    p = np.clip(data['raw'],1e-6,1-1e-6)
    cal.fit(np.log(p/(1-p))[:,None],data['y'],sample_weight=cw/cw.mean())
    return {'model':model,'calibrator':cal}


base.fit_utility = fit

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--development',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        base.run(args.smoke)
