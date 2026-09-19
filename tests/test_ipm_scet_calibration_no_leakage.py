from __future__ import annotations

import unittest

import pandas as pd

from ipm_scet_evidence_transport import apply_scet_model, compute_conflict_scores, fit_scet_model


class SCETCalibrationNoLeakageTests(unittest.TestCase):
    def test_fit_scet_model_records_only_calibration_queries(self) -> None:
        calibration = pd.DataFrame(
            {
                "dataset": ["toy"] * 4,
                "split": ["cal"] * 4,
                "query_id": ["c1", "c1", "c2", "c2"],
                "query_text": ["claim"] * 4,
                "query_type": ["claim"] * 4,
                "target_label": ["SUPPORT"] * 4,
                "queue_score": [1.0] * 4,
                "queue_rank": [1, 1, 2, 2],
                "candidate_id": ["a", "b", "c", "d"],
                "evidence_text": ["good", "bad", "good", "bad"],
                "evidence_source": ["doc"] * 4,
                "gold_support": [1, 0, 1, 0],
                "view_payload_json": ["{}"] * 4,
                "good_view_score": [0.95, 0.10, 0.90, 0.20],
                "noisy_view_score": [0.40, 0.95, 0.45, 0.90],
            }
        )
        evaluation = calibration.copy()
        evaluation["query_id"] = ["e1", "e1", "e2", "e2"]
        evaluation["gold_support"] = [0, 1, 0, 1]

        model = fit_scet_model(calibration, view_columns=["good_view_score", "noisy_view_score"], top_k=1)
        applied = apply_scet_model(evaluation, model)

        self.assertEqual(set(model.calibration_query_ids), {"c1", "c2"})
        self.assertTrue(set(model.calibration_query_ids).isdisjoint(set(applied["query_id"])))
        self.assertGreater(model.view_reliability["good_view_score"], model.view_reliability["noisy_view_score"])

    def test_conflict_scores_ignore_claim_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "query_text": ["the drug reduces fever", "the drug reduces fever"],
                "evidence_text": ["the drug reduces fever", "the result contradicts the claim"],
                "target_label": ["SUPPORT", "SUPPORT"],
                "dense_score": [0.9, 0.9],
                "entity_score": [0.8, 0.1],
                "relation_score": [0.8, 0.1],
            }
        )
        views = ["dense_score", "entity_score", "relation_score"]
        original = compute_conflict_scores(frame, views)
        flipped = frame.copy()
        flipped["target_label"] = ["REFUTES", "NOT ENOUGH INFO"]
        changed = compute_conflict_scores(flipped, views)
        pd.testing.assert_series_equal(original.reset_index(drop=True), changed.reset_index(drop=True))
        self.assertGreater(float(original.iloc[1]), float(original.iloc[0]))


if __name__ == "__main__":
    unittest.main()
