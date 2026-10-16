"""Independent checks of context-feature construction, screening and decisions."""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import r2_admission as a
import r5_context_gain as dev


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    freeze = json.loads((dev.O / 'development/freeze.json').read_text(encoding='utf-8'))[0]
    seed = freeze['seed']
    ids = freeze['evaluation_query_ids'][:12]
    frame = pd.read_parquet(dev.R / 'data/scored/fever_full_k100.parquet', columns=dev.COLS,
                            filters=[('query_id', 'in', ids)]).reset_index(drop=True)
    x, keep = dev.list_inputs(frame)
    variants = {}
    changed = frame.copy()
    changed['gold_evidence'] = 1 - changed.gold_evidence
    variants['flip_all_evidence_labels'] = changed
    variants['random_candidate_order'] = frame.sample(frac=1, random_state=20261016)
    variants['annotated_candidates_first'] = frame.sort_values('gold_evidence', ascending=False, kind='stable')
    variants['unannotated_candidates_first'] = frame.sort_values('gold_evidence', ascending=True, kind='stable')
    models = {}
    base_predictions = {}
    model_hashes = {}
    for group in dev.PLAN['groups']:
        path = dev.O / f'development/model_seed{seed}_{group}.pkl'
        model_hashes[group] = sha(path)
        with path.open('rb') as source:
            models[group] = pickle.load(source)
        base_predictions[group] = dev.predict(models[group], x)
    records = []
    for name, variant in variants.items():
        mapping = variant.index.to_numpy()
        variant = variant.reset_index(drop=True)
        tx, screen = dev.list_inputs(variant)
        assert np.array_equal(tx, x[mapping]), name
        assert np.array_equal(screen, keep[mapping]), name
        decisions = []
        for group, model in models.items():
            predicted = dev.predict(model, tx)
            assert np.array_equal(predicted, base_predictions[group][mapping]), (name, group)
            # One frozen label-free workload rule tests downstream rank/tie invariance.
            original = frame[keep].copy().reset_index(drop=True)
            score = base_predictions[group][keep]
            rule = a.fit_threshold(original, score, 100)
            admit, rank = a.apply_threshold(original, score, rule)
            transformed = variant[screen].copy().reset_index(drop=True)
            tscore = predicted[screen]
            alternate, arank = a.apply_threshold(transformed, tscore, rule)
            left = original[['query_id', 'candidate_id']].assign(score=score, admit=admit, rank=rank).set_index(['query_id', 'candidate_id']).sort_index()
            right = transformed[['query_id', 'candidate_id']].assign(score=tscore, admit=alternate, rank=arank).set_index(['query_id', 'candidate_id']).sort_index()
            assert left.equals(right), (name, group, 'admission')
            decisions.append({'group': group, 'scores_identical': True, 'ranks_and_admission_identical': True})
        records.append({'change': name, 'recomputed_features_identical': True,
                        'recomputed_screen_identical': True, 'group_predictions': decisions})
    isolated = frame[frame.query_id == ids[0]].copy().reset_index(drop=True)
    ix = np.flatnonzero(frame.query_id.to_numpy() == ids[0])
    one_x, one_screen = dev.list_inputs(isolated)
    assert np.array_equal(one_x, x[ix]) and np.array_equal(one_screen, keep[ix])
    report = {
        'passed': True, 'seed': seed, 'sample': 'First twelve frozen held-out query IDs from seed13.',
        'queries': len(ids), 'candidate_rows': len(frame), 'shared_screen_rows': int(keep.sum()),
        'driver_code_sha256': sha(__file__), 'feature_program_sha256': sha(dev.__file__),
        'frozen_models_sha256': model_hashes,
        'actual_feature_constructor_reexecuted': True,
        'shared_multiview_screen_for_both_models': True,
        'query_isolation_in_feature_construction': True,
        'checks': records,
        'scope': 'Checks feature/screen/prediction/rank/admission invariance on twelve held-out queries; no model fitting or evaluation-metric selection.',
        'admission_test_rule': 'Single label-free budget threshold derived once from the original 12-query sample and reused for every transformed input; this rule is only an invariance fixture.',
    }
    target = dev.O / 'independent_feature_audit.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'queries': len(ids), 'rows': len(frame), 'shared_screen_rows': int(keep.sum()), 'changes_checked': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
