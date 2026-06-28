from __future__ import annotations

import unittest

import pandas as pd

from ipm_cross_domain_evidence_benchmark import (
    benchmark_text_candidates,
    cosine_scores_from_embeddings,
    evidence_retrieval_metrics,
    judge_cross_domain_summary,
    pareto_frontier_flags,
    select_mvec_tg_rule,
    split_calibration_evaluation_queries,
    summarize_bts_detail,
    tg_weight_sensitivity_table,
)


class CrossDomainEvidenceBenchmarkTests(unittest.TestCase):
    def test_evidence_retrieval_metrics_counts_supported_positive_and_mrr(self) -> None:
        rows = pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q2", "q2"],
                "gold_support": [0, 1, 0, 0],
                "retrieval_score": [0.2, 0.9, 0.8, 0.1],
                "evidence_supported": [0, 1, 1, 0],
            }
        )

        metrics = evidence_retrieval_metrics(rows, top_k=1)

        self.assertEqual(metrics["queries"], 2)
        self.assertAlmostEqual(metrics["recall_at_5"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(metrics["support_precision"], 0.5)
        self.assertEqual(metrics["supported_positive"], 1)
        self.assertEqual(metrics["supported_negative"], 1)

    def test_select_mvec_tg_rule_uses_calibration_side_only(self) -> None:
        calibration = pd.DataFrame(
            {
                "query_id": ["c1", "c1", "c2", "c2"],
                "lexical_score": [0.9, 0.1, 0.3, 0.2],
                "dense_score": [0.8, 0.0, 0.7, 0.1],
                "gold_support": [1, 0, 0, 1],
            }
        )
        evaluation = pd.DataFrame(
            {
                "query_id": ["e1", "e1"],
                "lexical_score": [0.1, 0.9],
                "dense_score": [0.1, 0.9],
                "gold_support": [0, 1],
            }
        )

        rule = select_mvec_tg_rule(calibration, evaluation, view_columns=["lexical_score", "dense_score"])

        self.assertIn(rule["view_threshold"], [1, 2])
        self.assertNotIn("evaluation_support_precision", rule)

    def test_split_calibration_evaluation_queries_has_no_query_overlap(self) -> None:
        rows = pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"],
                "gold_support": [1, 0, 0, 1, 1, 0, 0, 1],
            }
        )

        calibration, evaluation = split_calibration_evaluation_queries(rows, calibration_fraction=0.5, seed=7)

        self.assertGreater(calibration["query_id"].nunique(), 0)
        self.assertGreater(evaluation["query_id"].nunique(), 0)
        self.assertTrue(set(calibration["query_id"]).isdisjoint(set(evaluation["query_id"])))

    def test_select_mvec_tg_rule_can_learn_weighted_consensus(self) -> None:
        calibration = pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
                "bm25_score": [0.9, 0.1, 0.7, 0.2, 0.3, 0.8],
                "e5_score": [0.8, 0.1, 0.2, 0.9, 0.4, 0.7],
                "gold_support": [1, 0, 1, 0, 0, 1],
            }
        )

        rule = select_mvec_tg_rule(
            calibration,
            None,
            view_columns=["bm25_score", "e5_score"],
            allow_weighted_consensus=True,
        )

        self.assertEqual(rule["mode"], "weighted_consensus")
        self.assertIn("view_weights", rule)
        self.assertGreater(rule["view_weights"]["bm25_score"], 0)
        self.assertGreater(rule["view_weights"]["e5_score"], 0)

    def test_benchmark_text_candidates_emits_retriever_rows(self) -> None:
        candidates = pd.DataFrame(
            {
                "dataset": ["scifact", "scifact", "scifact"],
                "split": ["dev", "dev", "dev"],
                "query_id": ["q1", "q1", "q1"],
                "query_text": ["aspirin reduces fever"] * 3,
                "query_type": ["claim"] * 3,
                "target_label": ["SUPPORT"] * 3,
                "queue_score": [1.0, 1.0, 1.0],
                "queue_rank": [1, 1, 1],
                "candidate_id": ["a", "b", "c"],
                "evidence_text": ["aspirin reduces fever", "unrelated", "fever rises"],
                "evidence_source": ["a", "b", "c"],
                "gold_support": [1, 0, 0],
                "view_payload_json": ["{}"] * 3,
            }
        )

        detail = benchmark_text_candidates(candidates, retrievers=["bm25", "tfidf_svd", "mvec_tg"], top_k=2)

        self.assertEqual(set(detail["retriever"]), {"bm25", "tfidf_svd", "mvec_tg"})
        self.assertTrue(detail["retrieval_score"].notna().all())
        self.assertTrue(detail["evidence_supported"].isin([0, 1]).all())

    def test_benchmark_text_candidates_can_evaluate_on_held_out_queries(self) -> None:
        candidates = pd.DataFrame(
            {
                "dataset": ["scifact"] * 8,
                "split": ["dev"] * 8,
                "query_id": ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"],
                "query_text": ["aspirin fever"] * 8,
                "query_type": ["claim"] * 8,
                "target_label": ["SUPPORT"] * 8,
                "queue_score": [1.0] * 8,
                "queue_rank": [1, 1, 2, 2, 3, 3, 4, 4],
                "candidate_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "evidence_text": ["aspirin fever", "noise", "aspirin fever", "noise", "aspirin fever", "noise", "aspirin fever", "noise"],
                "evidence_source": ["doc"] * 8,
                "gold_support": [1, 0, 1, 0, 1, 0, 1, 0],
                "view_payload_json": ["{}"] * 8,
            }
        )

        detail = benchmark_text_candidates(
            candidates,
            retrievers=["bm25", "mvec_tg"],
            top_k=1,
            calibration_query_fraction=0.5,
            calibration_seed=11,
        )

        self.assertEqual(set(detail["retriever"]), {"bm25", "mvec_tg"})
        self.assertEqual(set(detail["evaluation_partition"]), {"evaluation"})
        self.assertLess(detail["query_id"].nunique(), candidates["query_id"].nunique())

    def test_cosine_scores_from_embeddings_ranks_nearest_document(self) -> None:
        scores = cosine_scores_from_embeddings(
            query_vector=[1.0, 0.0],
            document_vectors=[[0.2, 0.0], [0.0, 1.0], [0.8, 0.1]],
        )

        self.assertEqual(int(scores.argmax()), 0)
        self.assertGreater(scores[0], scores[1])

    def test_semantic_sentence_retriever_uses_precomputed_score_column(self) -> None:
        candidates = pd.DataFrame(
            {
                "dataset": ["scifact", "scifact", "scifact"],
                "split": ["dev", "dev", "dev"],
                "query_id": ["q1", "q1", "q1"],
                "query_text": ["claim about aspirin"] * 3,
                "query_type": ["claim"] * 3,
                "target_label": ["SUPPORT"] * 3,
                "queue_score": [1.0, 1.0, 1.0],
                "queue_rank": [1, 1, 1],
                "candidate_id": ["a", "b", "c"],
                "evidence_text": ["candidate a", "candidate b", "candidate c"],
                "evidence_source": ["a", "b", "c"],
                "gold_support": [0, 1, 0],
                "view_payload_json": ["{}"] * 3,
            }
        )

        detail = benchmark_text_candidates(
            candidates,
            retrievers=["sbert_mini_lm"],
            top_k=1,
            precomputed_scores={"sbert_mini_lm_score": [0.1, 0.9, 0.2]},
        )

        self.assertEqual(set(detail["retriever"]), {"sbert_mini_lm"})
        top = detail.sort_values("retrieval_score", ascending=False).iloc[0]
        self.assertEqual(top["candidate_id"], "b")
        self.assertEqual(int(top["evidence_supported"]), 1)

    def test_tg_weight_sensitivity_reports_selected_rules(self) -> None:
        scored = pd.DataFrame(
            {
                "dataset": ["scifact"] * 4,
                "query_id": ["q1", "q1", "q2", "q2"],
                "gold_support": [1, 0, 0, 1],
                "bm25_score": [0.8, 0.1, 0.2, 0.1],
                "dense_score": [0.7, 0.0, 0.8, 0.2],
                "relation_score": [0.6, 0.2, 0.1, 0.9],
            }
        )

        table = tg_weight_sensitivity_table(
            scored,
            view_columns=["bm25_score", "dense_score", "relation_score"],
            precision_weights=[0.5, 2.0],
        )

        self.assertEqual(set(table["precision_weight"]), {0.5, 2.0})
        self.assertTrue(table["selected_rule"].astype(str).str.len().gt(0).all())
        self.assertTrue(table["recall_at_5"].between(0.0, 1.0).all())

    def test_summarize_bts_detail_uses_pooled_support_total(self) -> None:
        detail = pd.DataFrame(
            {
                "evidence_method": ["teacher", "teacher", "mvec_tg", "mvec_tg"],
                "reviewed": [10, 20, 10, 20],
                "captured": [8, 12, 8, 12],
                "supported": [6, 9, 7, 11],
                "support_total": [10, 20, 10, 20],
            }
        )

        summary = summarize_bts_detail(detail)

        mvec = summary[summary["retriever"].eq("mvec_tg")].iloc[0]
        self.assertEqual(mvec["supported_positive"], 18)
        self.assertEqual(mvec["supported_negative"], 12)
        self.assertAlmostEqual(mvec["support_precision"], 18 / 30)

    def test_judge_cross_domain_summary_marks_mvec_pareto_and_gate(self) -> None:
        summary = pd.DataFrame(
            {
                "dataset": ["scifact", "scifact", "scifact"],
                "retriever": ["bm25", "single_view_proto", "mvec_tg"],
                "queries": [10, 10, 10],
                "recall_at_5": [0.60, 0.70, 0.72],
                "mrr": [0.50, 0.65, 0.65],
                "ndcg_at_10": [0.55, 0.68, 0.70],
                "support_precision": [0.60, 0.61, 0.63],
                "supported_positive": [6, 7, 8],
                "supported_negative": [4, 5, 4],
            }
        )

        judged = judge_cross_domain_summary(summary)
        flagged = pareto_frontier_flags(judged)
        row = flagged[flagged["retriever"].eq("mvec_tg")].iloc[0]

        self.assertEqual(row["decision"], "pass")
        self.assertTrue(row["pareto_frontier"])


if __name__ == "__main__":
    unittest.main()
