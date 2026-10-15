"""One frozen confirmation evaluation with source-family resampling."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
import run_multivers_final_portfolio as portfolio
import run_multivers_dual_calibration as dual

R2 = Path(__file__).resolve().parents[1]
DATA = R2/'data/multivers_confirmation'
OUT = R2/'results/multivers_extension/confirmation_evaluation'
FREEZE = portfolio.OUT/'full/confirmation_freeze.json'
PLAN = R2/'data/multivers_confirmation_analysis.json'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_models():
    freeze = json.loads(FREEZE.read_text())
    assert not freeze['smoke'] and not freeze['confirmation_performance_accessed']
    assert freeze['source_sha256'] == digest(Path(portfolio.__file__))
    models = {}
    for label, info in freeze['models'].items():
        path = FREEZE.parent/info['path_name']
        assert digest(path) == info['sha256']
        models[label] = joblib.load(path)
    return freeze, models


def score_frozen(model, frame):
    # The full-context control uses no isolated-sentence features. Its original
    # helper eagerly constructs those unused columns before returning full x.
    # Apply exactly the fitted full feature matrix without that unused read.
    if model['kind'] == 'control' and model['model'].get('group') == 'full':
        fitted = model['model']
        x = portfolio.context.features(frame, 'scores')
        if fitted['scale'] is not None:
            x = fitted['scale'].transform(x)
        return fitted['classifier'].predict_proba(x)[:, 1]
    return portfolio.score_model(model, frame)


def bootstrap_draws(queries, repetitions=2000):
    families = sorted({q['source_family'] for q in queries.values()})
    lookup = {f: i for i, f in enumerate(families)}
    rng = np.random.default_rng(20261015+843)
    draws = rng.multinomial(len(families), np.full(len(families), 1/len(families)), size=repetitions)
    return draws, lookup


def reselected_precision(frame, scores, queries, draws, family_lookup):
    # Each duplicated family contributes a distinct copy of every query. The
    # original within-query top-five candidates remain eligible in each copy.
    rank = admission.ranks(frame, scores)
    ix = np.flatnonzero(rank <= 5)
    ix = ix[np.lexsort((frame.tie_key.to_numpy(str)[ix], -scores[ix]))]
    codes = np.array([family_lookup[queries[q]['source_family']] for q in frame.query_id.to_numpy()[ix]])
    query_codes = np.array([family_lookup[q['source_family']] for q in queries.values()])
    budgets = draws[:, query_codes].sum(axis=1)
    w = draws[:, codes]
    count = np.cumsum(w, axis=1)
    boundary = (count < budgets[:, None]).sum(axis=1)
    assert (boundary < len(ix)).all()
    y = frame.gold_evidence.to_numpy(int)[ix]
    correct = np.cumsum(w*y[None, :], axis=1)
    row = np.arange(len(draws))
    prev = np.maximum(boundary-1, 0)
    before_count = np.where(boundary > 0, count[row, prev], 0)
    before_correct = np.where(boundary > 0, correct[row, prev], 0)
    total_correct = before_correct+(budgets-before_count)*y[boundary]
    return total_correct/np.maximum(budgets, 1)


def intervals(values):
    lo, hi = np.quantile(values, [.025, .975])
    slo, shi = np.quantile(values, [.0125, .9875])
    return {'ci95_low': float(lo), 'ci95_high': float(hi),
            'simultaneous_ci_low': float(slo), 'simultaneous_ci_high': float(shi)}


def fixed_precision(stat, queries, draws, lookup):
    codes = [lookup[queries[q]['source_family']] for q in stat.query_id]
    sums = draws[:, codes]@stat[['correct', 'admitted']].to_numpy(float)
    return sums[:, 0]/np.maximum(sums[:, 1], 1)


def smoke():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze, models = load_models()
    frame = dual.prepare()
    keep = sorted(frame.query_id.unique())[:12]
    frame = frame[frame.query_id.isin(keep)].reset_index(drop=True)
    oldqueries = json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    families = portfolio.context.source_families()
    queries = {q: {**oldqueries[q], 'source_family': families[q]} for q in keep}
    draws, lookup = bootstrap_draws(queries, 100)
    for label, model in models.items():
        s = score_frozen(model, frame)
        assert np.allclose(s, portfolio.score_model(model, frame), atol=1e-12, rtol=0)
        changed = frame.copy()
        changed['gold_evidence'] = 1-changed.gold_evidence
        changed['claim_verdict'] = 'unused'
        assert np.allclose(score_frozen(model, changed.iloc[::-1])[::-1], s, atol=1e-12, rtol=0)
        rule = budget_rule(frame, s, len(queries))
        stat = query_statistics(frame, s, rule, np.zeros(len(frame), bool), queries)
        assert stat.admitted.sum() == len(queries)
        one_draw = np.ones((1, len(lookup)), dtype=int)
        exact = reselected_precision(frame, s, queries, one_draw, lookup)
        assert np.isclose(exact[0], stat.correct.sum()/stat.admitted.sum())
        repeated = reselected_precision(frame, s, queries, draws, lookup)
        assert np.isfinite(repeated).all() and ((repeated >= 0)&(repeated <= 1)).all()
        assert np.array_equal(repeated-repeated, np.zeros(len(repeated)))
    audit = {'passed': True, 'source_sha256': digest(Path(__file__)), 'plan_sha256': digest(PLAN),
             'freeze_sha256': digest(FREEZE), 'reserved_outcomes_accessed': False,
             'exact_bootstrap_budget_matches_direct_selection': True, 'row_order_outcome_invariance': True}
    (OUT/'smoke_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


def prepare_confirmation_frame():
    for folder, name in [('confirmation_inference', 'inference_audit.json'), ('confirmation_single', 'audit.json'), ('confirmation_scoring', 'audit.json')]:
        audit = json.loads((R2/'results/multivers_extension'/folder/name).read_text())
        assert audit.get('passed', True) and not audit['smoke']
    frame = pd.read_parquet(DATA/'scored.parquet')
    documents = {}
    docpath = R2/'results/multivers_extension/confirmation_inference/document_scores.jsonl'
    for line in docpath.read_text().splitlines():
        r = json.loads(line)
        assert (r['query_id'], r['doc_id']) not in documents
        documents[(r['query_id'], r['doc_id'])] = r
    expected = set(frame[['query_id', 'doc_id']].itertuples(index=False, name=None))
    assert expected == set(documents)
    rows = [documents[(q, d)] for q, d in zip(frame.query_id, frame.doc_id)]
    frame['raw_multivers_rationale'] = [r['rationale_probabilities'][int(s)] for r, s in zip(rows, frame.sentence_id)]
    frame['mv_document_evidence'] = [1-r['label_probabilities'][1] for r in rows]
    frame['mv_support'] = [r['label_probabilities'][2] for r in rows]
    frame['mv_refute'] = [r['label_probabilities'][0] for r in rows]
    frame['raw_multivers'] = frame.raw_multivers_rationale*frame.mv_document_evidence
    single = pd.read_json(R2/'results/multivers_extension/confirmation_single/scores.jsonl', lines=True,
                          dtype={'query_id': str, 'candidate_id': str})
    assert len(single) == len(frame) and not single.duplicated(['query_id', 'candidate_id']).any()
    frame = frame.merge(single[['query_id', 'candidate_id', 'single_rationale']], on=['query_id', 'candidate_id'], validate='one_to_one', sort=False)
    assert frame.single_rationale.notna().all()
    return frame


def run(resume_runtime_repair=False):
    start = time.monotonic()
    smoke_audit = json.loads((OUT/'smoke_audit.json').read_text())
    assert smoke_audit['passed'] and smoke_audit['source_sha256'] == digest(Path(__file__)) and smoke_audit['plan_sha256'] == digest(PLAN)
    started = OUT/'confirmation_started.json'
    if resume_runtime_repair:
        prior = json.loads(started.read_text())
        assert prior['freeze_sha256'] == digest(FREEZE) and prior['plan_sha256'] == digest(PLAN)
        assert not (OUT/'assessment.json').exists() and not (OUT/'summary.csv').exists() and not (OUT/'paired_intervals.csv').exists()
        (OUT/'runtime_repair_audit.json').write_text(json.dumps({'prior_start': prior,
            'continued_source_sha256': digest(Path(__file__)), 'fitted_models_changed': False,
            'thresholds_changed': False, 'candidate_family_changed': False,
            'repair': 'Avoid an unused single_evidence read in the full-context control; identical predictions verified on development input.',
            'completed_confirmation_analysis_before_repair': False}, indent=2))
    else:
        assert not started.exists(), 'The one-shot confirmation was already started; inspect the saved state before any rerun.'
    freeze, models = load_models()
    assert digest(DATA/'candidates.parquet') == freeze['confirmation_inputs_sha256']
    # Check every necessary score artifact before the one-shot outcome gate.
    frame = prepare_confirmation_frame()
    queries = json.loads((DATA/'queries.json').read_text(encoding='utf-8'))
    assert set(frame.query_id) == set(queries)
    if not resume_runtime_repair:
        started.write_text(json.dumps({'time_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'freeze_sha256': digest(FREEZE), 'plan_sha256': digest(PLAN), 'source_sha256': digest(Path(__file__))}, indent=2))
    draws, lookup = bootstrap_draws(queries)
    summaries, comparison_rows, score_rows = [], [], []
    for pool in ['controlled_pool', 'natural_pool']:
        part = frame[frame[pool]].reset_index(drop=True)
        assert part.groupby('query_id').size().eq(100).all()
        all_stat, exact_boot, fixed_boot = {}, {}, {}
        for label, model in models.items():
            s = score_frozen(model, part)
            assert np.isfinite(s).all()
            score_rows.append(pd.DataFrame({'pool': pool, 'method': label, 'query_id': part.query_id,
                                            'candidate_id': part.candidate_id, 'score': s}))
            rules = {'exact_budget': budget_rule(part, s, len(queries)),
                     'prospective_100': freeze['models'][label]['prospective_rule']}
            for analysis, rule in rules.items():
                stat = query_statistics(part, s, rule, np.zeros(len(part), bool), queries)
                if analysis == 'exact_budget':
                    assert stat.admitted.sum() == len(queries)
                stat.to_csv(OUT/f'query_{pool}_{label}_{analysis}.csv', index=False)
                summaries.append({'pool': pool, 'method': label, 'analysis': analysis,
                                  'chosen': freeze['models'][label]['name'], **admission.summarize(stat)})
                all_stat[(label, analysis)] = stat
                fixed_boot[(label, analysis)] = fixed_precision(stat, queries, draws, lookup)
            exact_boot[label] = reselected_precision(part, s, queries, draws, lookup)
        for reference in ['rationale', 'product', 'full_calibration', 'dual_blend', 'best_control']:
            for analysis in ['exact_budget', 'prospective_100']:
                a, b = all_stat[('selected', analysis)], all_stat[(reference, analysis)]
                point = a.correct.sum()/max(a.admitted.sum(), 1)-b.correct.sum()/max(b.admitted.sum(), 1)
                comparison_rows.append({'pool': pool, 'method': 'selected', 'reference': reference, 'analysis': analysis,
                                        'bootstrap': 'fixed_decisions', 'delta_precision': point,
                                        **intervals(fixed_boot[('selected', analysis)]-fixed_boot[(reference, analysis)])})
                if analysis == 'exact_budget':
                    comparison_rows.append({'pool': pool, 'method': 'selected', 'reference': reference, 'analysis': analysis,
                                            'bootstrap': 'budget_reselected', 'delta_precision': point,
                                            **intervals(exact_boot['selected']-exact_boot[reference])})
        print('evaluated', pool, 'elapsed', round(time.monotonic()-start, 1), flush=True)
    summary, comparison = pd.DataFrame(summaries), pd.DataFrame(comparison_rows)
    summary.to_csv(OUT/'summary.csv', index=False)
    comparison.to_csv(OUT/'paired_intervals.csv', index=False)
    pd.concat(score_rows).to_parquet(OUT/'frozen_candidate_scores.parquet', index=False)
    primary = comparison[(comparison.pool == 'controlled_pool')&(comparison.analysis == 'exact_budget')&
                         (comparison.bootstrap == 'budget_reselected')&comparison.reference.isin(['rationale', 'best_control'])]
    supported = bool((primary.delta_precision > 0).all() and (primary.simultaneous_ci_low > 0).all())
    positive_points = bool((primary.delta_precision > 0).all())
    assessment = {'supported_increment': supported, 'positive_point_gains_both_controls': positive_points,
                  'interpretation': 'confirmed_precision_increment' if supported else ('inconclusive_precision_increment' if positive_points else 'added_portfolio_increment_not_established'),
                  'query_count': len(queries), 'source_families': len(lookup), 'bootstrap_repetitions': len(draws),
                  'co_primary': primary.to_dict('records'), 'manuscript_update_authorized_by_evidence': supported,
                  'all_declared_comparisons_reported': True, 'confirmation_tuning_performed': False,
                  'source_sha256': digest(Path(__file__)), 'plan_sha256': digest(PLAN), 'freeze_sha256': digest(FREEZE),
                  'elapsed_seconds': time.monotonic()-start}
    (OUT/'assessment.json').write_text(json.dumps(assessment, indent=2))
    print(summary[['pool', 'analysis', 'method', 'evidence_precision', 'volume_per_100', 'evidence_recall']].to_string(index=False), flush=True)
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume-runtime-repair', action='store_true')
    with threadpool_limits(limits=2):
        args = p.parse_args()
        if args.smoke:
            smoke()
        else:
            run(args.resume_runtime_repair)
