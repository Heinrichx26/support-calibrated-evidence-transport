from __future__ import annotations

import unittest

import pandas as pd

from ipm_credot_evidence_transport import (
    apply_credot_model,
    build_credot_ablation_summary,
    credot_support_certificates,
    fit_and_apply_credot,
    fit_credot_model,
)


class CREDOTEvidenceTransportTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        for qid, query in [
            ("c1", "aspirin reduces fever adult trial"),
            ("c2", "gene increases protein expression stress"),
            ("c3", "airport delay propagates hub rotation"),
            ("e1", "battery defect causes thermal runaway recall"),
        ]:
            rows.extend(
                [
                    {
                        "dataset": "toy",
                        "split": "dev",
                        "query_id": qid,
                        "query_text": query,
                        "query_type": "claim",
                        "target_label": "SUPPORT",
                        "queue_score": 1.0,
                        "queue_rank": 1,
                        "candidate_id": f"{qid}_support",
                        "evidence_text": query,
                        "evidence_source": "doc",
                        "gold_support": 1,
                        "view_payload_json": "{}",
                        "bm25_score": 0.82,
                        "dense_score": 0.86,
                        "entity_score": 0.88,
                        "relation_score": 0.84,
                        "cross_encoder_score": 0.80,
                    },
                    {
                        "dataset": "toy",
                        "split": "dev",
                        "query_id": qid,
                        "query_text": query,
                        "query_type": "claim",
                        "target_label": "SUPPORT",
                        "queue_score": 1.0,
                        "queue_rank": 2,
                        "candidate_id": f"{qid}_conflict",
                        "evidence_text": f"{query} contradicts unrelated",
                        "evidence_source": "doc",
                        "gold_support": 0,
                        "view_payload_json": "{}",
                        "bm25_score": 0.96,
                        "dense_score": 0.98,
                        "entity_score": 0.05,
                        "relation_score": 0.04,
                        "cross_encoder_score": 0.97,
                    },
                    {
                        "dataset": "toy",
                        "split": "dev",
                        "query_id": qid,
                        "query_text": query,
                        "query_type": "claim",
                        "target_label": "SUPPORT",
                        "queue_score": 1.0,
                        "queue_rank": 3,
                        "candidate_id": f"{qid}_partial",
                        "evidence_text": "partial report with limited anchoring",
                        "evidence_source": "doc",
                        "gold_support": 0,
                        "view_payload_json": "{}",
                        "bm25_score": 0.48,
                        "dense_score": 0.52,
                        "entity_score": 0.28,
                        "relation_score": 0.20,
                        "cross_encoder_score": 0.45,
                    },
                ]
            )
        return pd.DataFrame(rows)

    def test_credot_demotes_conflicting_high_similarity_candidate(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["c1", "c2", "c3"])].copy()
        evaluation = frame[frame["query_id"].eq("e1")].copy()

        detail, _ = fit_and_apply_credot(
            calibration,
            evaluation,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score", "cross_encoder_score"],
            top_k=1,
        )

        selected = detail.sort_values("rank").iloc[0]
        conflict = detail[detail["candidate_id"].eq("e1_conflict")].iloc[0]
        support = detail[detail["candidate_id"].eq("e1_support")].iloc[0]
        self.assertEqual(selected["candidate_id"], "e1_support")
        self.assertGreater(support["credot_support_score"], conflict["credot_support_score"])
        self.assertGreater(conflict["credot_robust_transport_cost"], support["credot_robust_transport_cost"])

    def test_credot_obligation_gap_reduces_support_score(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["c1", "c2", "c3"])].copy()
        evaluation = frame[frame["query_id"].eq("e1")].copy()

        detail, _ = fit_and_apply_credot(
            calibration,
            evaluation,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score", "cross_encoder_score"],
            top_k=1,
        )

        support = detail[detail["candidate_id"].eq("e1_support")].iloc[0]
        partial = detail[detail["candidate_id"].eq("e1_partial")].iloc[0]
        self.assertGreater(support["credot_evidence_mass"], partial["credot_evidence_mass"])
        self.assertGreater(support["credot_support_score"], partial["credot_support_score"])

    def test_credot_transport_cost_is_mass_normalized_liability(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["c1", "c2", "c3"])].copy()
        evaluation = frame[frame["query_id"].eq("e1")].copy()

        detail, _ = fit_and_apply_credot(
            calibration,
            evaluation,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score", "cross_encoder_score"],
            top_k=1,
        )

        support = detail[detail["candidate_id"].eq("e1_support")].iloc[0]
        partial = detail[detail["candidate_id"].eq("e1_partial")].iloc[0]
        self.assertLess(support["credot_robust_transport_cost"], partial["credot_robust_transport_cost"])

    def test_credot_calibration_records_only_calibration_queries(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["c1", "c2"])].copy()
        evaluation = frame[frame["query_id"].isin(["c3", "e1"])].copy()

        model = fit_credot_model(
            calibration,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score", "cross_encoder_score"],
            top_k=1,
        )
        detail = apply_credot_model(evaluation, model)

        self.assertEqual(set(model.calibration_query_ids), {"c1", "c2"})
        self.assertTrue(set(model.calibration_query_ids).isdisjoint(set(detail["query_id"])))
        self.assertTrue(detail["credot_support_score"].between(0.0, 1.0).all())

    def test_credot_certificates_include_transport_and_credal_fields(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["c1", "c2", "c3"])].copy()
        evaluation = frame[frame["query_id"].eq("e1")].copy()

        detail, model = fit_and_apply_credot(
            calibration,
            evaluation,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score", "cross_encoder_score"],
            top_k=1,
        )
        certs = credot_support_certificates(detail, model, per_query=1)

        self.assertEqual(len(certs), 1)
        self.assertIn("transport_plan", certs[0])
        self.assertIn("obligation_distribution", certs[0])
        self.assertIn("supply_distribution", certs[0])
        self.assertIn("credal_reliability_interval", certs[0])
        self.assertIn("robust_transport_cost", certs[0])

    def test_credot_certificates_reconstruct_transport_fields_from_compact_detail(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["c1", "c2", "c3"])].copy()
        evaluation = frame[frame["query_id"].eq("e1")].copy()

        model = fit_credot_model(
            calibration,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score", "cross_encoder_score"],
            top_k=1,
        )
        detail = apply_credot_model(evaluation, model, include_transport_artifacts=False)
        certs = credot_support_certificates(detail, model, per_query=1)

        self.assertNotIn("credot_transport_plan_json", detail.columns)
        self.assertEqual(len(certs), 1)
        self.assertIn("transport_plan", certs[0])
        self.assertIn("obligation_distribution", certs[0])
        self.assertIn("supply_distribution", certs[0])

    def test_credot_ablation_reports_transport_variants(self) -> None:
        frame = self._frame()
        calibration = frame[frame["query_id"].isin(["c1", "c2", "c3"])].copy()
        evaluation = frame[frame["query_id"].eq("e1")].copy()
        _, model = fit_and_apply_credot(
            calibration,
            evaluation,
            view_columns=["bm25_score", "dense_score", "entity_score", "relation_score", "cross_encoder_score"],
            top_k=1,
        )

        ablation = build_credot_ablation_summary(calibration, evaluation, model)

        self.assertIn("credot", set(ablation["variant"]))
        self.assertIn("credot_no_robustness", set(ablation["variant"]))
        self.assertIn("credot_no_conflict", set(ablation["variant"]))


if __name__ == "__main__":
    unittest.main()
