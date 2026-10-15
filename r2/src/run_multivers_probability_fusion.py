"""Bounded exploratory fusion with nested query selection and honest controls."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
from run_multivers_rationale_extension import load_frame
from run_multivers_extension import paired_intervals

R2 = Path(__file__).resolve().parents[1]
SPEC_PATH = R2 / 'data/multivers_fusion_protocol.json'
SPEC = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
OUT = R2 / 'results/multivers_extension/probability_fusion'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_array(frame, group):
    eps = SPEC['probability_clipping']
    def lp(v):
        return logit(np.clip(v, eps, 1-eps))
    cols = [lp(frame.raw_multivers_rationale.to_numpy(float)),
            lp(frame.mv_document_evidence.to_numpy(float))]
    if group in ['mv_semantic', 'mv_direction_consistency']:
        cols.extend(frame[c].to_numpy(float) for c in ['raw_ce', 'raw_e5', 'raw_bge', 'raw_minilm'])
    if group == 'mv_direction_consistency':
        e = frame.mv_document_evidence.to_numpy(float)
        sup = frame.mv_support.to_numpy(float) / np.maximum(e, eps)
        ref = frame.mv_refute.to_numpy(float) / np.maximum(e, eps)
        alignment = sup*frame.raw_nli_entailment.to_numpy(float) + ref*frame.nli_contradiction.to_numpy(float)
        cols.extend([lp(frame.raw_nli_relevance.to_numpy(float)), alignment, np.abs(sup-ref)])
    return np.column_stack(cols)


def score_model(train, apply, name):
    if name == 'raw_multivers':
        return apply.raw_multivers.to_numpy(float), {'baseline': name}
    if name == 'raw_multivers_rationale':
        return apply.raw_multivers_rationale.to_numpy(float), {'baseline': name}
    group, c_str = name.split(':')
    z, zz = feature_array(train, group), feature_array(apply, group)
    scaler = StandardScaler().fit(z)
    counts = train.groupby('query_id').size()
    w = train.query_id.map(1/counts).to_numpy(float)
    w *= len(w)/w.sum()
    fitted = LogisticRegression(C=float(c_str), class_weight='balanced', max_iter=1000,
                                solver='lbfgs', random_state=13)
    fitted.fit(scaler.transform(z), train.gold_evidence.to_numpy(int), sample_weight=w)
    scores = fitted.decision_function(scaler.transform(zz))
    info = {'group': group, 'C': float(c_str), 'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(), 'weights': fitted.coef_[0].tolist(),
            'intercept': float(fitted.intercept_[0])}
    return scores, info


def precision(frame, score):
    rule = budget_rule(frame, score, frame.query_id.nunique())
    admitted, _ = admission.apply_threshold(frame, score, rule)
    return float(frame.gold_evidence.to_numpy(int)[admitted].sum()/admitted.sum())


def candidates():
    return list(SPEC['baseline_scores']) + [f'{g}:{c:g}' for g in SPEC['feature_groups'] for c in SPEC['C']]


def inner_select(fitting, seed):
    names = candidates()
    ids = sorted(fitting.query_id.unique(), key=lambda q: hashlib.sha256(f'{seed}:inner:{q}'.encode()).hexdigest())
    mapping = {q: i % SPEC['inner_query_folds'] for i, q in enumerate(ids)}
    fold = fitting.query_id.map(mapping).to_numpy()
    rows = []
    for f in range(SPEC['inner_query_folds']):
        train = fitting.loc[fold != f].reset_index(drop=True)
        valid = fitting.loc[fold == f].reset_index(drop=True)
        assert set(train.query_id).isdisjoint(valid.query_id)
        for name in names:
            score, _ = score_model(train, valid, name)
            rows.append({'seed': seed, 'fold': f, 'method': name, 'precision': precision(valid, score)})
    data = pd.DataFrame(rows)
    quality = data.groupby('method').precision.mean().to_dict()
    chosen = max(names, key=lambda name: (round(quality[name], 12), -names.index(name)))
    mv_names = [n for n in names if n.startswith('raw_') or n.startswith('mv_probability:')]
    mv_chosen = max(mv_names, key=lambda name: (round(quality[name], 12), -names.index(name)))
    return chosen, mv_chosen, data


def prepare_frame():
    frame = load_frame()
    docs = pd.read_json(R2/'results/strengthening/multivers/document_scores.jsonl', lines=True)
    probs = np.asarray(docs.label_probabilities.tolist(), dtype=float)
    doc_frame = pd.DataFrame({'query_id': docs.query_id.astype(str), 'doc_id': docs.doc_id.astype(str),
                              'mv_refute': probs[:, 0], 'mv_support': probs[:, 2],
                              'mv_document_evidence': 1-probs[:, 1]})
    return frame.merge(doc_frame, on=['query_id', 'doc_id'], validate='many_to_one', sort=False)


def run(smoke):
    start = time.monotonic()
    out = OUT/('smoke' if smoke else 'full')
    if not smoke:
        smoke_audit = json.loads((OUT/'smoke/audit.json').read_text())
        assert smoke_audit['protocol_sha256'] == digest(SPEC_PATH)
        assert smoke_audit['passed']
    out.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame()
    query_info = json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    summaries, selections, cv_rows = [], [], []
    collections = {}
    for seed in [13] if smoke else SPEC['seeds']:
        old = json.loads((R2/f'results/experiments/scifact_full_k100/model_seed{seed}.json').read_text())
        fitids, calids, testids = [old[k] for k in ['fitting_query_ids', 'threshold_query_ids', 'evaluation_query_ids']]
        assert not (set(fitids)&set(calids) or set(fitids)&set(testids) or set(calids)&set(testids))
        fitting = frame[frame.query_id.isin(fitids)].reset_index(drop=True)
        calibration = frame[frame.query_id.isin(calids)].reset_index(drop=True)
        chosen, mv_chosen, cv = inner_select(fitting, seed)
        selections.append({'seed': seed, 'fusion_choice': chosen, 'mv_only_choice': mv_chosen})
        cv_rows.append(cv)
        # Save all choices before reading held-out labels for performance.
        (out/f'choices_seed{seed}.json').write_text(json.dumps(selections[-1], indent=2), encoding='utf-8')
        evalids = sorted(testids)[:12] if smoke else testids
        test = frame[frame.query_id.isin(evalids)].reset_index(drop=True)
        queries = {q: query_info[q] for q in evalids}
        original_scores, stress, _ = admission.score_pipeline(test, old)
        for label, name in [('multivers', 'raw_multivers'), ('multivers_rationale', 'raw_multivers_rationale'),
                            ('mv_only_selected', mv_chosen), ('fusion_selected', chosen)]:
            combined = pd.concat([calibration, test], ignore_index=True)
            output, info = score_model(fitting, combined, name)
            cal_score, test_score = output[:len(calibration)], output[len(calibration):]
            assert np.isfinite(output).all()
            # Assert independence from labels, claim verdict and row order.
            changed = test.copy()
            changed['gold_evidence'] = 1-changed.gold_evidence
            changed['claim_verdict'] = 'unused'
            unchanged, _ = score_model(fitting, changed, name)
            assert np.allclose(unchanged, test_score, rtol=0, atol=1e-10)
            reverse, _ = score_model(fitting, test.iloc[::-1], name)
            assert np.allclose(reverse[::-1], test_score, rtol=0, atol=1e-10)
            prospective = admission.fit_threshold(calibration, cal_score, SPEC['workload_target'])
            batch = budget_rule(test, test_score, len(evalids))
            for analysis, rule in [('prospective_100', prospective), ('exact_budget', batch)]:
                stat = query_statistics(test, test_score, rule, stress, queries)
                if analysis == 'exact_budget':
                    assert stat.admitted.sum() == len(evalids)
                stat.to_csv(out/f'query_seed{seed}_{label}_{analysis}.csv', index=False)
                summaries.append({'seed': seed, 'method': label, 'chosen': name, 'analysis': analysis,
                                  **admission.summarize(stat)})
                collections.setdefault(analysis, {}).setdefault(seed, {})[label] = stat
            info.update({'chosen': name, 'prospective_rule': prospective, 'batch_rule': batch,
                         'fitting_query_ids': fitids, 'threshold_query_ids': calids, 'evaluation_query_ids': evalids})
            (out/f'model_seed{seed}_{label}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print('seed', seed, 'fusion', chosen, 'mv-only', mv_chosen, 'elapsed', round(time.monotonic()-start, 1), flush=True)
    data = pd.DataFrame(summaries)
    data.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(selections).to_csv(out/'selected_models.csv', index=False)
    pd.concat(cv_rows).to_csv(out/'inner_cv.csv', index=False)
    agg = data.groupby(['analysis', 'method'])[['evidence_precision', 'volume_per_100', 'coverage', 'evidence_recall', 'all_facts_recall']].agg(['mean', 'std'])
    agg.to_csv(out/'mean_sd.csv')
    if not smoke:
        intervals = []
        metrics = ['evidence_precision', 'annotated_per_100', 'volume_per_100', 'coverage', 'evidence_recall', 'all_facts_recall']
        for analysis, stats in collections.items():
            for method, reference in [('fusion_selected', 'multivers'), ('fusion_selected', 'multivers_rationale'),
                                      ('fusion_selected', 'mv_only_selected'), ('multivers_rationale', 'multivers')]:
                for row in paired_intervals(stats, method, reference, metrics):
                    intervals.append({'analysis': analysis, **row})
        pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    audit = {'smoke': smoke, 'passed': True, 'protocol_sha256': digest(SPEC_PATH),
             'code_sha256': digest(Path(__file__)), 'exact_budget_verified': True,
             'query_partitions_disjoint': True, 'outcome_and_verdict_invariance': True,
             'row_order_invariance': True, 'selection_uses_fitting_queries_only': True,
             'exploratory_due_to_prior_outcome_inspection': True,
             'elapsed_seconds': time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(data[data.analysis == 'exact_budget'][['seed','method','chosen','evidence_precision']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.smoke)
