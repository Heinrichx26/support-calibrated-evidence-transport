from __future__ import annotations

import unittest

import pandas as pd

from ipm_scet_evidence_transport import (
    SCET_OUTPUT_FILES,
    prepare_scet_scored_candidates,
    validate_scet_schema,
)


class SCETSchemaTests(unittest.TestCase):
    def test_validate_scet_schema_preserves_unified_columns(self) -> None:
        candidates = pd.DataFrame(
            {
                "dataset": ["toy"],
                "split": ["dev"],
                "query_id": ["q1"],
                "query_text": ["aspirin reduces fever"],
                "query_type": ["claim"],
                "target_label": ["SUPPORT"],
                "queue_score": [1.0],
                "queue_rank": [1],
                "candidate_id": ["c1"],
                "evidence_text": ["aspirin can reduce fever"],
                "evidence_source": ["doc1"],
                "gold_support": [1],
                "view_payload_json": ["{}"],
            }
        )

        validated = validate_scet_schema(candidates)

        self.assertEqual(validated["query_id"].iloc[0], "q1")
        self.assertEqual(validated["gold_support"].iloc[0], 1)
        self.assertIn("scet_detail.csv", SCET_OUTPUT_FILES)
        self.assertIn("scet_support_certificates.jsonl", SCET_OUTPUT_FILES)

    def test_prepare_scet_scored_candidates_adds_base_views(self) -> None:
        candidates = pd.DataFrame(
            {
                "dataset": ["toy", "toy"],
                "split": ["dev", "dev"],
                "query_id": ["q1", "q1"],
                "query_text": ["aspirin reduces fever", "aspirin reduces fever"],
                "query_type": ["claim", "claim"],
                "target_label": ["SUPPORT", "SUPPORT"],
                "queue_score": [1.0, 1.0],
                "queue_rank": [1, 1],
                "candidate_id": ["c1", "c2"],
                "evidence_text": ["aspirin reduces fever in patients", "unrelated aircraft delay"],
                "evidence_source": ["doc1", "doc2"],
                "gold_support": [1, 0],
                "view_payload_json": ["{}", "{}"],
            }
        )

        scored = prepare_scet_scored_candidates(candidates, neural_level="none")

        for column in ["bm25_score", "tfidf_score", "dense_score", "entity_score", "relation_score"]:
            self.assertIn(column, scored.columns)
        self.assertEqual(scored["query_id"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
