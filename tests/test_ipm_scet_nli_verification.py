from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ipm_scet_nli_verification import (
    apply_nli_verification_loop,
    attach_nli_scores,
    candidate_pool_audit,
    nli_premise_hypothesis_texts,
    nli_probabilities_to_scores,
    select_nli_candidate_pool,
)


class SCETNLIVerificationTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dataset": "toy",
                    "query_id": "q1",
                    "candidate_id": "a",
                    "query_text": "claim one",
                    "evidence_text": "claim one is supported",
                    "evidence_source": "doc",
                    "gold_support": 1,
                    "bge_m3_score": 0.90,
                    "scet_support_score": 0.92,
                    "evidence_supported": 1,
                    "retriever": "scet",
                },
                {
                    "dataset": "toy",
                    "query_id": "q1",
                    "candidate_id": "b",
                    "query_text": "claim one",
                    "evidence_text": "claim one is contradicted",
                    "evidence_source": "doc",
                    "gold_support": 0,
                    "bge_m3_score": 0.85,
                    "scet_support_score": 0.60,
                    "evidence_supported": 0,
                    "retriever": "scet",
                },
                {
                    "dataset": "toy",
                    "query_id": "q2",
                    "candidate_id": "c",
                    "query_text": "claim two",
                    "evidence_text": "claim two support",
                    "evidence_source": "doc",
                    "gold_support": 1,
                    "bge_m3_score": 0.80,
                    "scet_support_score": 0.91,
                    "evidence_supported": 1,
                    "retriever": "scet",
                },
                {
                    "dataset": "toy",
                    "query_id": "q2",
                    "candidate_id": "d",
                    "query_text": "claim two",
                    "evidence_text": "unrelated evidence",
                    "evidence_source": "doc",
                    "gold_support": 0,
                    "bge_m3_score": 0.40,
                    "scet_support_score": 0.20,
                    "evidence_supported": 0,
                    "retriever": "scet",
                },
            ]
        )

    def test_select_nli_candidate_pool_uses_retrieval_rank_per_query(self) -> None:
        pool = select_nli_candidate_pool(self._frame(), retrieval_col="bge_m3_score", retrieve_k=1)

        self.assertEqual(pool["candidate_id"].tolist(), ["a", "c"])
        self.assertEqual(pool["nli_pool_rank"].tolist(), [1, 1])

    def test_probabilities_are_mapped_by_model_labels(self) -> None:
        scores = nli_probabilities_to_scores(
            np.asarray([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]),
            {0: "contradiction", 1: "entailment", 2: "neutral"},
        )

        self.assertAlmostEqual(float(scores.loc[0, "nli_support_score"]), 0.7)
        self.assertAlmostEqual(float(scores.loc[1, "nli_support_score"]), -0.5)

    def test_nli_text_order_uses_evidence_as_premise(self) -> None:
        pool = self._frame().iloc[[0, 1]].copy()

        premises, hypotheses = nli_premise_hypothesis_texts(pool)

        self.assertEqual(premises[0], "claim one is supported")
        self.assertEqual(hypotheses[0], "claim one")

    def test_attach_nli_scores_marks_unscored_rows_as_outside_pool(self) -> None:
        frame = self._frame()
        score_rows = pd.DataFrame(
            [
                {
                    "dataset": "toy",
                    "query_id": "q1",
                    "candidate_id": "a",
                    "nli_pool_rank": 1,
                    "nli_entailment_score": 0.9,
                    "nli_contradiction_score": 0.05,
                    "nli_neutral_score": 0.05,
                    "nli_support_score": 0.85,
                }
            ]
        )

        scored = attach_nli_scores(frame, score_rows)

        self.assertAlmostEqual(float(scored.loc[scored["candidate_id"].eq("a"), "nli_support_score"].iloc[0]), 0.85)
        self.assertTrue(scored.loc[scored["candidate_id"].eq("b"), "nli_support_score"].isna().all())

    def test_nli_verification_loop_selects_entailing_candidate_inside_pool(self) -> None:
        frame = self._frame()
        score_rows = pd.DataFrame(
            [
                {
                    "dataset": "toy",
                    "query_id": "q1",
                    "candidate_id": "a",
                    "nli_pool_rank": 1,
                    "nli_entailment_score": 0.9,
                    "nli_contradiction_score": 0.05,
                    "nli_neutral_score": 0.05,
                    "nli_support_score": 0.85,
                },
                {
                    "dataset": "toy",
                    "query_id": "q1",
                    "candidate_id": "b",
                    "nli_pool_rank": 2,
                    "nli_entailment_score": 0.1,
                    "nli_contradiction_score": 0.8,
                    "nli_neutral_score": 0.1,
                    "nli_support_score": -0.7,
                },
                {
                    "dataset": "toy",
                    "query_id": "q2",
                    "candidate_id": "c",
                    "nli_pool_rank": 1,
                    "nli_entailment_score": 0.8,
                    "nli_contradiction_score": 0.1,
                    "nli_neutral_score": 0.1,
                    "nli_support_score": 0.7,
                },
            ]
        )

        detail = apply_nli_verification_loop(attach_nli_scores(frame, score_rows), top_k=1)

        selected = detail[detail["evidence_supported"].eq(1)].sort_values("query_id")
        self.assertEqual(selected["candidate_id"].tolist(), ["a", "c"])

    def test_candidate_pool_audit_reports_gold_retention(self) -> None:
        audit = candidate_pool_audit(self._frame())

        self.assertEqual(float(audit.loc[0, "queries"]), 2.0)
        self.assertEqual(float(audit.loc[0, "mean_candidates_per_query"]), 2.0)
        self.assertEqual(float(audit.loc[0, "gold_support_rows"]), 2.0)


if __name__ == "__main__":
    unittest.main()
