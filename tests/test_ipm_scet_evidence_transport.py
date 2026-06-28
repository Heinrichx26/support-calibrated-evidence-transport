from __future__ import annotations

import unittest

import pandas as pd

from ipm_scet_evidence_transport import (
    apply_semantic_portfolio_transport,
    apply_support_transport,
    apply_support_portfolio_transport,
    build_scet_ablation_summary,
    compute_conflict_scores,
    fit_and_apply_scet,
    fit_support_portfolio_transport,
    support_certificates,
)


class SCETEvidenceTransportTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dataset": ["toy"] * 8,
                "split": ["dev"] * 8,
                "query_id": ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"],
                "query_text": [
                    "drug reduces fever",
                    "drug reduces fever",
                    "gene increases expression",
                    "gene increases expression",
                    "protein is not activated",
                    "protein is not activated",
                    "airport delay propagates",
                    "airport delay propagates",
                ],
                "query_type": ["claim"] * 8,
                "target_label": ["SUPPORT"] * 8,
                "queue_score": [1.0] * 8,
                "queue_rank": [1, 1, 2, 2, 3, 3, 4, 4],
                "candidate_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "evidence_text": [
                    "drug reduces fever",
                    "unrelated high similarity fever",
                    "gene increases expression",
                    "gene contradicts expression",
                    "protein is not activated",
                    "protein activated unrelated",
                    "airport delay propagates",
                    "airport schedule unrelated",
                ],
                "evidence_source": ["doc"] * 8,
                "gold_support": [1, 0, 1, 0, 1, 0, 1, 0],
                "view_payload_json": ["{}"] * 8,
                "bm25_score": [0.90, 0.88, 0.85, 0.83, 0.92, 0.90, 0.88, 0.40],
                "dense_score": [0.95, 0.93, 0.90, 0.91, 0.94, 0.92, 0.89, 0.50],
                "entity_score": [0.90, 0.05, 0.85, 0.05, 0.90, 0.10, 0.86, 0.20],
                "relation_score": [0.90, 0.10, 0.80, 0.15, 0.90, 0.20, 0.80, 0.30],
            }
        )

    def test_conflict_penalty_marks_high_similarity_unanchored_candidate(self) -> None:
        frame = self._frame()

        conflicts = compute_conflict_scores(frame, ["bm25_score", "dense_score", "entity_score", "relation_score"])

        self.assertGreater(conflicts.iloc[1], conflicts.iloc[0])
        self.assertGreater(conflicts.iloc[3], conflicts.iloc[2])

    def test_fit_and_apply_scet_outputs_supported_rows_and_certificates(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["q1", "q2"])].copy()
        evaluation = frame[frame["query_id"].isin(["q3", "q4"])].copy()

        detail, model = fit_and_apply_scet(
            calibration,
            evaluation,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score"],
            top_k=1,
        )
        certs = support_certificates(detail, model, per_query=1)

        self.assertEqual(set(detail["retriever"]), {"scet"})
        self.assertTrue(detail["evidence_supported"].isin([0, 1]).all())
        self.assertGreater(detail["scet_support_score"].max(), detail["scet_support_score"].min())
        self.assertGreaterEqual(len(certs), 1)
        self.assertIn("view_reliability", certs[0])

    def test_ablation_summary_reports_core_variants(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["q1", "q2"])].copy()
        evaluation = frame[frame["query_id"].isin(["q3", "q4"])].copy()
        _, model = fit_and_apply_scet(
            calibration,
            evaluation,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score"],
            top_k=1,
        )

        ablation = build_scet_ablation_summary(calibration, evaluation, model)

        self.assertIn("scet", set(ablation["variant"]))
        self.assertIn("scet_equal_views", set(ablation["variant"]))
        self.assertIn("scet_no_conflict_penalty", set(ablation["variant"]))

    def test_support_portfolio_transport_uses_complementary_views(self) -> None:
        rows = []
        for idx in range(1, 7):
            rows.append(
                {
                    "dataset": "toy",
                    "split": "dev",
                    "query_id": f"q{idx}",
                    "query_text": f"claim {idx}",
                    "query_type": "claim",
                    "target_label": "SUPPORT",
                    "queue_score": 1.0,
                    "queue_rank": idx,
                    "candidate_id": f"p{idx}",
                    "evidence_text": f"claim {idx} supported by entity relation",
                    "evidence_source": "doc",
                    "gold_support": 1,
                    "view_payload_json": "{}",
                    "cross_encoder_score": 0.72,
                    "bge_m3_score": 0.70,
                    "entity_score": 0.92,
                    "relation_score": 0.88,
                    "bm25_score": 0.66,
                    "dense_score": 0.70,
                }
            )
            rows.append(
                {
                    "dataset": "toy",
                    "split": "dev",
                    "query_id": f"q{idx}",
                    "query_text": f"claim {idx}",
                    "query_type": "claim",
                    "target_label": "SUPPORT",
                    "queue_score": 1.0,
                    "queue_rank": idx,
                    "candidate_id": f"n{idx}",
                    "evidence_text": f"claim {idx} high similarity unsupported",
                    "evidence_source": "doc",
                    "gold_support": 0,
                    "view_payload_json": "{}",
                    "cross_encoder_score": 0.99,
                    "bge_m3_score": 0.98,
                    "entity_score": 0.05,
                    "relation_score": 0.04,
                    "bm25_score": 0.90,
                    "dense_score": 0.98,
                }
            )
        frame = pd.DataFrame(rows)
        calibration = frame[frame["query_id"].isin(["q1", "q2", "q3", "q4"])].copy()
        evaluation = frame[frame["query_id"].isin(["q5", "q6"])].copy()
        views = ["cross_encoder_score", "bge_m3_score", "entity_score", "relation_score", "bm25_score", "dense_score"]

        portfolio = fit_support_portfolio_transport(calibration, views)
        detail = apply_support_portfolio_transport(evaluation, portfolio)

        selected = detail.sort_values(["query_id", "scet_support_score"], ascending=[True, False]).groupby("query_id").head(1)
        self.assertEqual(selected["candidate_id"].tolist(), ["p5", "p6"])
        self.assertTrue(any(abs(weight) > 0 for weight in portfolio.feature_weights.values()))

    def test_semantic_portfolio_transport_selects_candidate_supported_by_any_semantic_view(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset": ["toy", "toy"],
                "split": ["dev", "dev"],
                "query_id": ["q1", "q1"],
                "query_text": ["claim", "claim"],
                "query_type": ["claim", "claim"],
                "target_label": ["SUPPORT", "SUPPORT"],
                "queue_score": [1.0, 1.0],
                "queue_rank": [1, 1],
                "candidate_id": ["a", "b"],
                "evidence_text": ["support from one encoder", "unsupported but cross high"],
                "evidence_source": ["doc", "doc"],
                "gold_support": [1, 0],
                "view_payload_json": ["{}", "{}"],
                "cross_encoder_score": [0.30, 0.95],
                "bge_m3_score": [0.97, 0.20],
                "e5_small_v2_score": [0.92, 0.25],
                "entity_score": [0.60, 0.05],
                "relation_score": [0.60, 0.05],
            }
        )

        detail = apply_semantic_portfolio_transport(frame, ["cross_encoder_score", "bge_m3_score", "e5_small_v2_score"])

        selected = detail.sort_values("scet_support_score", ascending=False).iloc[0]
        self.assertEqual(selected["candidate_id"], "a")


if __name__ == "__main__":
    unittest.main()
