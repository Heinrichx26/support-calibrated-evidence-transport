from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R2 = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((R2 / 'data/r2_protocol.json').read_text())
VIEWS = ['raw_bm25', 'raw_tfidf', 'raw_lsa', 'raw_entity', 'raw_relation', 'raw_minilm', 'raw_e5', 'raw_bge', 'raw_ce', 'raw_nli_relevance', 'raw_nli_directional', 'raw_nli_entailment']
NEGATION = set('no not without never none lack lacks lacking failed fails failure inhibit inhibits inhibited decrease decreases decreased reduced reduces absence'.split())
CONFLICT = set('contradict contradicts contradicted opposite refute refutes refuted inconsistent unrelated'.split())


def split_ids(ids, fraction, seed):
    ordered = sorted(ids, key=lambda q: int(hashlib.sha1(f'{seed}:{q}'.encode()).hexdigest()[:12], 16))
    n = max(1, min(len(ids)-1, round(len(ids)*fraction)))
    return ordered[:n], ordered[n:]


def fit_scale(frame):
    probs = np.linspace(0, 1, 1001)
    return {col: np.quantile(frame[col].to_numpy(float), probs).tolist() for col in VIEWS}


def transform(frame, scale):
    result = []
    for col in VIEWS:
        values = np.asarray(scale[col])
        unique, first = np.unique(values, return_index=True)
        # Use midranks for exact constant/duplicate values, on one frozen global distribution.
        last = np.searchsorted(values, unique, side='right') - 1
        probs = (first + last)/(2*(len(values)-1))
        result.append(np.interp(frame[col].to_numpy(float), unique, probs, left=0, right=1))
    return np.column_stack(result)


def conflict(frame, x):
    disagreement = np.std(x[:, :9], axis=1, ddof=1)
    semantic = x[:, [2, 5, 6, 7, 8]].max(axis=1)
    structural = x[:, [3, 4]].max(axis=1)
    gap = np.clip(semantic - structural, 0, 1)
    def tokens(text):
        return set(re.findall(r'[A-Za-z0-9]+', str(text).lower()))
    queries = {q: tokens(t) for q, t in frame[['query_id', 'query_text']].drop_duplicates().itertuples(index=False)}
    docs = {text: tokens(text) for text in frame.evidence_text.unique()}
    mismatch, terms = [], []
    for q, doc in zip(frame.query_id, frame.evidence_text):
        qt, dt = queries[q], docs[doc]
        mismatch.append(float(bool(qt & NEGATION) != bool(dt & NEGATION)))
        terms.append(float(bool((qt | dt) & CONFLICT)))
    signal = .35*disagreement + .35*gap + .15*np.asarray(mismatch) + .15*np.asarray(terms)
    return signal, np.column_stack([disagreement, gap, mismatch, terms])


def features(x, signal):
    semantic = x[:, [2, 5, 6, 7, 8, 9]].mean(axis=1)
    anchor = x[:, [3, 4]].mean(axis=1)
    return np.column_stack([x, x[:, :9].mean(axis=1), x[:, :9].max(axis=1), x[:, :9].std(axis=1), semantic*anchor, semantic-anchor, signal])


def fit_portfolio(frame, z, c, label):
    y = frame[label].to_numpy(int)
    if len(np.unique(y)) < 2:
        return {'constant': float(y.mean())}
    counts = frame.groupby('query_id').size()
    weights = frame.query_id.map(1/counts).to_numpy(float)
    weights *= len(weights)/weights.sum()
    clf = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs', max_iter=500, random_state=13)
    clf.fit(z, y, sample_weight=weights)
    return {'weights': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]), 'C': c}


def portfolio(z, model):
    if 'constant' in model:
        return np.full(len(z), model['constant'])
    return expit(z @ np.asarray(model['weights']) + model['intercept'])


def ranks(frame, scores):
    # Fixed hash keys carry no label, verdict, candidate order, or fitted outcome information.
    order = np.lexsort((frame.tie_key.to_numpy(str), -np.asarray(scores), frame.query_id.to_numpy(str)))
    qs = frame.query_id.to_numpy(str)[order]
    starts = np.maximum.accumulate(np.where(np.r_[True, qs[1:] != qs[:-1]], np.arange(len(qs)), 0))
    result = np.empty(len(qs), dtype=np.int16)
    result[order] = np.arange(len(qs))-starts+1
    return result


def fit_threshold(frame, scores, target):
    rank = ranks(frame, scores)
    eligible = np.flatnonzero(rank <= PROTOCOL['eligible_rank'])
    order = eligible[np.lexsort((frame.tie_key.to_numpy(str)[eligible], -np.asarray(scores)[eligible]))]
    desired = min(len(order), round(frame.query_id.nunique()*target/100))
    if desired == 0:
        return {'threshold': float('inf'), 'tie_cutoff': '', 'target': target, 'calibration_volume': 0}
    boundary = order[desired-1]
    return {'threshold': float(scores[boundary]), 'tie_cutoff': str(frame.iloc[boundary].tie_key), 'target': target, 'calibration_volume': 100*desired/frame.query_id.nunique()}


def apply_threshold(frame, scores, rule):
    rank = ranks(frame, scores)
    score = np.asarray(scores)
    above = (score > rule['threshold']) | ((score == rule['threshold']) & (frame.tie_key.to_numpy(str) <= rule['tie_cutoff']))
    return (rank <= rule.get('eligible_rank', PROTOCOL['eligible_rank'])) & above, rank


def selection_quality(frame, scores, target, label):
    rule = fit_threshold(frame, scores, target)
    admit, rank = apply_threshold(frame, scores, rule)
    gold = frame[label].to_numpy(int)
    issued = int(admit.sum())
    correct = int((gold*admit).sum())
    nq = frame.query_id.nunique()
    hit = frame.loc[admit & (gold == 1), 'query_id'].nunique()
    return (correct/max(issued, 1), hit/max(nq, 1), correct/max(nq, 1))


def family(frame, x, signal, z, portfolios, reliability):
    scores = {col[4:]: x[:, j] for j, col in enumerate(VIEWS)}
    scores['reliability_mean'] = x[:, :9] @ np.asarray(reliability)
    scores['rag_utility'] = .62*x[:, [6, 7]].max(axis=1) + .28*(.6*x[:, 4]+.4*x[:, 3]) + .10 - .20*signal
    scores['retrieval_verification'] = .7*x[:, 8] + .2*x[:, 4] + .1*x[:, 3]
    for c, fitted in portfolios.items():
        scores['portfolio_'+format(float(c), 'g')] = portfolio(z, fitted)
    return {name: np.round(score, 12) for name, score in scores.items()}


def fit_pipeline(fitting, threshold_calibration, seed, target=100, label='gold_evidence'):
    assert set(fitting.query_id).isdisjoint(set(threshold_calibration.query_id))
    fitting = fitting.reset_index(drop=True)
    scale = fit_scale(fitting)
    x = transform(fitting, scale)
    signal, components = conflict(fitting, x)
    z = features(x, signal)
    quality = []
    for j in range(9):
        y = fitting[label].to_numpy(int)
        auc = roc_auc_score(y, x[:, j]) if len(np.unique(y)) > 1 else .5
        quality.append(max(auc-.5, .01))
    reliability = (np.asarray(quality)/sum(quality)).tolist()
    query_ids = sorted(fitting.query_id.unique(), key=lambda q: hashlib.sha256(f'{seed}:inner:{q}'.encode()).hexdigest())
    fold = {q: i%PROTOCOL['inner_folds'] for i, q in enumerate(query_ids)}
    folds = fitting.query_id.map(fold).to_numpy()
    oof = {c: np.zeros(len(fitting)) for c in PROTOCOL['portfolio_regularization_C']}
    for f in range(PROTOCOL['inner_folds']):
        train, valid = folds != f, folds == f
        for c in oof:
            model = fit_portfolio(fitting.loc[train], z[train], c, label)
            oof[c][valid] = portfolio(z[valid], model)
    fitted = {c: fit_portfolio(fitting, z, c, label) for c in oof}
    base = family(fitting, x, signal, z, fitted, reliability)
    for c, values in oof.items():
        base['portfolio_'+format(float(c), 'g')] = values
    single_names = [v[4:] for v in VIEWS]
    singles = [(selection_quality(fitting, base[n], target, label), n) for n in single_names]
    primary_single = max(singles, key=lambda x: (x[0], x[1]))[1]
    all_forms = []
    for name, score in base.items():
        for penalty in PROTOCOL['conflict_penalties']:
            scored = np.round(score - penalty*signal, 12)
            all_forms.append((selection_quality(fitting, scored, target, label), name, penalty))
    best = max(all_forms, key=lambda x: (x[0], -x[2], x[1]))
    best_penalty = max((x for x in all_forms if x[2] > 0), key=lambda x: (x[0], -x[2], x[1]))
    # A fixed reference conflict quartile, independent of every evaluated admission rule and label.
    stress_cut = float(np.quantile(signal, .75))
    model = {'version': PROTOCOL['version'], 'seed': seed, 'target': target, 'label': label, 'scale': scale, 'reliability': reliability, 'portfolios': fitted, 'selected_form': best[1], 'penalty': best[2], 'selection_quality': best[0], 'conflict_form': best_penalty[1], 'conflict_penalty': best_penalty[2], 'primary_single': primary_single, 'stress_cutoff': stress_cut, 'fitting_query_ids': sorted(fitting.query_id.unique()), 'threshold_query_ids': sorted(threshold_calibration.query_id.unique())}
    scores, stress, _ = score_pipeline(threshold_calibration, model)
    model['thresholds'] = {name: fit_threshold(threshold_calibration, score, target) for name, score in scores.items()}
    # Learn-then-test uses the selected, fitted scorer and an independent calibration sample.
    grid = PROTOCOL['risk_control']['threshold_grid']
    alpha = PROTOCOL['risk_control']['query_error_target']
    delta = PROTOCOL['risk_control']['failure_probability']
    ltt = []
    y = threshold_calibration[label].to_numpy(int)
    for value in grid:
        rule = {'threshold': value, 'tie_cutoff': 'ffffffffffffffff', 'target': None}
        admitted, _ = apply_threshold(threshold_calibration, scores['scet'], rule)
        errors = threshold_calibration.loc[admitted & (y == 0), 'query_id'].nunique()
        n = threshold_calibration.query_id.nunique()
        p = float(binom.cdf(errors, n, alpha))
        ltt.append({'threshold': value, 'error_queries': int(errors), 'queries': int(n), 'p_value': p, 'passes': p <= delta/len(grid), 'admitted': int(admitted.sum())})
    acceptable = [r for r in ltt if r['passes'] and r['admitted'] > 0]
    risk_rule = min(acceptable, key=lambda r: r['threshold']) if acceptable else {'threshold': float('inf'), 'admitted': 0}
    model['risk_control'] = {'alpha': alpha, 'delta': delta, 'tests': ltt, 'threshold': risk_rule['threshold'], 'nonempty_feasible': bool(acceptable)}
    model['thresholds']['scet_risk_control'] = {'threshold': risk_rule['threshold'], 'tie_cutoff': 'ffffffffffffffff', 'target': None}
    return model


def score_pipeline(frame, model):
    if frame.empty:
        return {}, np.array([], dtype=bool), np.array([])
    x = transform(frame, model['scale'])
    signal, components = conflict(frame, x)
    z = features(x, signal)
    fitted = {float(c): v for c, v in model['portfolios'].items()}
    scores = family(frame, x, signal, z, fitted, model['reliability'])
    scores['scet'] = scores[model['selected_form']] - model['penalty']*signal
    scores['conflict_pressure'] = scores[model['conflict_form']] - model['conflict_penalty']*signal
    scores['scet_zero_penalty'] = scores[model['selected_form']]
    scores['scet_risk_control'] = scores['scet']
    scores = {name: np.round(score, 12) for name, score in scores.items()}
    return scores, signal >= model['stress_cutoff'], signal


def query_statistics(frame, scores, rule, stress, queries, label='gold_evidence'):
    admitted, rank = apply_threshold(frame, scores, rule)
    y = frame[label].to_numpy(int)
    rows = []
    byq = frame.groupby('query_id', sort=False).indices
    for qid, q in queries.items():
        ix = np.asarray(byq.get(qid, []), dtype=int)
        a, gold = admitted[ix], y[ix]
        original_gold = len(q['support_ids']) if label == 'gold_directional' else q['gold_count']
        sets = [q['support_ids']] if label == 'gold_directional' else q['gold_sets']
        admitted_ids = set(frame.iloc[ix[a]].candidate_id)
        all_facts = int(any(set(s) <= admitted_ids for s in sets if s))
        gold_ranks = np.sort(rank[ix][gold == 1])
        reciprocal = 1/float(gold_ranks[0]) if len(gold_ranks) else 0
        discount = 1/np.log2(gold_ranks[gold_ranks <= 10]+1)
        ideal = (1/np.log2(np.arange(1, min(original_gold, 10)+1)+1)).sum()
        ndcg = float(discount.sum()/ideal) if ideal else 0
        s = stress[ix]
        rows.append({'query_id': qid, 'claim_verdict': q['claim_verdict'], 'gold_count': original_gold, 'verified_gold_count': q['verified_gold_count'], 'fully_verified': q['fully_verified'], 'admitted': int(a.sum()), 'correct': int((gold*a).sum()), 'covered': int(a.any()), 'hit': int((gold*a).sum() > 0), 'ranking_gold_at_5': int((gold*(rank[ix] <= 5)).sum()), 'mrr': reciprocal, 'ndcg_at_10': ndcg, 'all_facts': all_facts, 'stress_admitted': int((a*s).sum()), 'stress_correct': int((gold*a*s).sum()), 'has_error': int((a*(1-gold)).any())})
    return pd.DataFrame(rows)


def summarize(stats):
    n = len(stats)
    issued, correct = stats.admitted.sum(), stats.correct.sum()
    stress_n = stats.stress_admitted.sum()
    gold_q = stats.gold_count > 0
    nogold = ~gold_q
    recall = stats.loc[gold_q, 'correct']/stats.loc[gold_q, 'gold_count']
    ranking_recall = stats.loc[gold_q, 'ranking_gold_at_5']/stats.loc[gold_q, 'gold_count']
    return {'queries': n, 'evidence_precision': correct/issued if issued else np.nan, 'annotated_per_100': 100*correct/max(n, 1), 'volume_per_100': 100*issued/max(n, 1), 'coverage': stats.covered.mean(), 'zero_admission_rate': 1-stats.covered.mean(), 'false_admission_no_evidence': stats.loc[nogold, 'covered'].mean() if nogold.any() else np.nan, 'evidence_recall': recall.mean() if gold_q.any() else np.nan, 'ranking_recall_at_5': ranking_recall.mean() if gold_q.any() else np.nan, 'mrr': stats.loc[gold_q, 'mrr'].mean() if gold_q.any() else np.nan, 'ndcg_at_10': stats.loc[gold_q, 'ndcg_at_10'].mean() if gold_q.any() else np.nan, 'all_facts_recall': stats.loc[gold_q, 'all_facts'].mean() if gold_q.any() else np.nan, 'query_error_rate': stats.has_error.mean(), 'stress_precision': stats.stress_correct.sum()/stress_n if stress_n else np.nan, 'stress_admitted': int(stress_n), 'admitted_total': int(issued), 'correct_total': int(correct)}
