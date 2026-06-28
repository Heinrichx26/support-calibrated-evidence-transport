from __future__ import annotations

import unittest

import pandas as pd

from ipm_event_evidence_benchmark import compare_to_teacher, paired_method_tests, pareto_frontier_flags, summarize_benchmark_detail


class EventEvidenceBenchmarkTests(unittest.TestCase):
    def test_summarize_benchmark_detail_pools_supported_cases_by_method(self) -> None:
        detail = pd.DataFrame(
            {
                "protocol": ["rolling", "rolling", "rolling", "rolling"],
                "split": ["a", "b", "a", "b"],
                "queue_source": ["LightGBM", "LightGBM", "LightGBM", "LightGBM"],
                "evidence_method": ["teacher", "teacher", "mvec_tg", "mvec_tg"],
                "rows": [100, 200, 100, 200],
                "positives": [20, 30, 20, 30],
                "reviewed": [10, 20, 10, 20],
                "captured": [8, 12, 8, 12],
                "supported": [6, 9, 7, 11],
                "pr_auc": [0.90, 0.91, 0.90, 0.91],
                "top10_capture": [0.80, 0.40, 0.80, 0.40],
                "support_precision": [0.60, 0.75, 0.70, 0.80],
            }
        )

        summary = summarize_benchmark_detail(detail)

        mvec = summary[summary["evidence_method"].eq("mvec_tg")].iloc[0]
        self.assertEqual(mvec["splits"], 2)
        self.assertEqual(mvec["reviewed"], 30)
        self.assertEqual(mvec["captured"], 20)
        self.assertEqual(mvec["supported"], 18)
        self.assertAlmostEqual(mvec["evidence_hit"], 0.9)
        self.assertAlmostEqual(mvec["supported_per_100"], 60.0)
        self.assertAlmostEqual(mvec["mean_pr_auc"], 0.905)

    def test_compare_to_teacher_adds_deltas_and_absolute_supported_gain(self) -> None:
        summary = pd.DataFrame(
            {
                "protocol": ["rolling", "rolling"],
                "queue_source": ["LightGBM", "LightGBM"],
                "evidence_method": ["teacher", "mvec_tg"],
                "splits": [2, 2],
                "rows": [300, 300],
                "positives": [50, 50],
                "reviewed": [30, 30],
                "captured": [20, 20],
                "supported": [15, 18],
                "evidence_hit": [0.75, 0.90],
                "supported_per_100": [50.0, 60.0],
                "support_precision": [0.65, 0.64],
                "mean_pr_auc": [0.90, 0.90],
                "mean_top10_capture": [0.80, 0.80],
            }
        )

        judged = compare_to_teacher(summary)

        row = judged[judged["evidence_method"].eq("mvec_tg")].iloc[0]
        self.assertAlmostEqual(row["evidence_hit_delta"], 0.15)
        self.assertAlmostEqual(row["supported_per_100_delta"], 10.0)
        self.assertEqual(row["absolute_supported_gain"], 3)
        self.assertEqual(row["decision"], "pass")

    def test_paired_method_tests_compare_mvec_against_teacher_by_split(self) -> None:
        detail = pd.DataFrame(
            {
                "protocol": ["rolling"] * 6,
                "split": ["a", "a", "b", "b", "c", "c"],
                "queue_source": ["LightGBM"] * 6,
                "evidence_method": ["teacher", "mvec_tg"] * 3,
                "evidence_hit": [0.80, 0.84, 0.82, 0.85, 0.81, 0.86],
                "supported_per_100": [40.0, 42.0, 41.0, 43.0, 39.0, 41.5],
                "support_precision": [0.62, 0.61, 0.63, 0.62, 0.64, 0.63],
            }
        )

        tests = paired_method_tests(detail, method="mvec_tg", reference="teacher")

        hit = tests[tests["metric"].eq("evidence_hit")].iloc[0]
        self.assertEqual(hit["paired_splits"], 3)
        self.assertGreater(hit["mean_delta"], 0.0)
        self.assertIn("wilcoxon_p", hit)

    def test_pareto_frontier_flags_methods_not_dominated_on_evidence_metrics(self) -> None:
        table = pd.DataFrame(
            {
                "protocol": ["rolling", "rolling", "rolling"],
                "queue_source": ["LightGBM", "LightGBM", "LightGBM"],
                "evidence_method": ["teacher", "high_recall", "balanced"],
                "evidence_hit": [0.70, 0.95, 0.90],
                "supported_per_100": [40.0, 62.0, 60.0],
                "support_precision": [0.50, 0.55, 0.70],
            }
        )

        flagged = pareto_frontier_flags(table)

        self.assertFalse(flagged.loc[flagged["evidence_method"].eq("teacher"), "pareto_frontier"].iloc[0])
        self.assertTrue(flagged.loc[flagged["evidence_method"].eq("high_recall"), "pareto_frontier"].iloc[0])
        self.assertTrue(flagged.loc[flagged["evidence_method"].eq("balanced"), "pareto_frontier"].iloc[0])


if __name__ == "__main__":
    unittest.main()
