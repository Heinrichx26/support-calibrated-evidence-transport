from __future__ import annotations

import unittest

import pandas as pd

from ipm_scet_open_pool_audit import certificate_metrics, matched_volume_rows, summarize_detail


class SCETOpenPoolAuditTests(unittest.TestCase):
    def _detail(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dataset": ["toy"] * 8,
                "split": ["dev"] * 8,
                "query_id": ["q1", "q1", "q2", "q2", "q1", "q1", "q2", "q2"],
                "query_text": ["claim"] * 8,
                "query_type": ["claim"] * 8,
                "target_label": ["SUPPORT"] * 8,
                "queue_score": [1.0] * 8,
                "queue_rank": [1, 1, 2, 2, 1, 1, 2, 2],
                "candidate_id": ["a", "b", "c", "d", "a", "b", "c", "d"],
                "evidence_text": [
                    "claim support",
                    "claim distractor",
                    "claim support two",
                    "claim conflict",
                    "claim support",
                    "claim distractor",
                    "claim support two",
                    "claim conflict",
                ],
                "evidence_source": ["doc"] * 8,
                "gold_support": [1, 0, 1, 0, 1, 0, 1, 0],
                "view_payload_json": ["{}"] * 8,
                "bm25_score": [0.9, 0.8, 0.9, 0.7, 0.9, 0.8, 0.9, 0.7],
                "entity_score": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
                "relation_score": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
                "retriever": ["bm25", "bm25", "bm25", "bm25", "scet", "scet", "scet", "scet"],
                "retrieval_score": [0.9, 0.8, 0.9, 0.7, 0.95, 0.1, 0.95, 0.1],
                "rank": [1, 2, 1, 2, 1, 2, 1, 2],
                "top_k": [1] * 8,
                "evidence_supported": [1, 0, 1, 0, 1, 0, 1, 0],
                "scet_conflict_score": [0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6],
                "scet_support_threshold": [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9],
            }
        )

    def test_certificate_metrics_reports_certificate_volume(self) -> None:
        metrics = certificate_metrics(self._detail().query("retriever == 'scet'"), top_k=1)

        self.assertEqual(metrics["certificates"], 2)
        self.assertEqual(metrics["cert_per_100"], 100.0)
        self.assertEqual(metrics["support_precision"], 1.0)

    def test_summarize_detail_adds_candidate_scope_metadata(self) -> None:
        summary = summarize_detail(
            self._detail(),
            audit_mode="full",
            candidate_mode="smoke",
            candidates_per_query=500,
            top_k=1,
        )

        self.assertEqual(set(summary["retriever"]), {"bm25", "scet"})
        self.assertTrue((summary["candidates_per_query"] == 500).all())
        self.assertIn("large_candidate_pool_k500", set(summary["candidate_scope"]))

    def test_matched_volume_rows_include_scet_and_comparator(self) -> None:
        rows = matched_volume_rows(self._detail(), top_k=1)

        self.assertEqual(set(rows["method"]), {"SCET", "bm25"})
        scet = rows[rows["method"].eq("SCET")].iloc[0]
        self.assertEqual(float(scet["cert_per_100"]), 100.0)
        comparator = rows[rows["method"].eq("bm25")].iloc[0]
        self.assertGreaterEqual(float(comparator["support_precision"]), 0.0)


if __name__ == "__main__":
    unittest.main()
