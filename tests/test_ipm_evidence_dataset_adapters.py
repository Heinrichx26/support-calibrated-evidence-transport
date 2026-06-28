from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ipm_evidence_dataset_adapters import (
    UNIFIED_COLUMNS,
    build_hover_candidates,
    build_fever_candidates,
    build_scifact_candidates,
    collect_fever_candidate_pool,
    collect_hover_candidate_pool,
    load_hotpotqa_context_index,
    validate_unified_schema,
)


class EvidenceDatasetAdapterTests(unittest.TestCase):
    def test_validate_unified_schema_accepts_complete_candidate_table(self) -> None:
        table = pd.DataFrame(
            {
                "dataset": ["scifact"],
                "split": ["dev"],
                "query_id": ["c1"],
                "query_text": ["claim text"],
                "query_type": ["claim"],
                "target_label": ["SUPPORT"],
                "queue_score": [0.9],
                "queue_rank": [1],
                "candidate_id": ["doc1::0"],
                "evidence_text": ["evidence sentence"],
                "evidence_source": ["doc1"],
                "gold_support": [1],
                "view_payload_json": [json.dumps({"lexical": "claim evidence"})],
            }
        )

        validated = validate_unified_schema(table)

        self.assertEqual(validated.columns.tolist(), UNIFIED_COLUMNS)
        self.assertEqual(validated.iloc[0]["gold_support"], 1)

    def test_build_scifact_candidates_marks_cited_document_sentences_as_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus.jsonl").write_text(
                json.dumps({"doc_id": 10, "title": "Paper A", "abstract": ["first sentence", "second sentence"]}) + "\n"
                + json.dumps({"doc_id": 20, "title": "Paper B", "abstract": ["negative sentence"]}) + "\n",
                encoding="utf-8",
            )
            (root / "claims_dev.jsonl").write_text(
                json.dumps(
                    {
                        "id": 1,
                        "claim": "claim text",
                        "evidence": {"10": [{"sentences": [1], "label": "SUPPORT"}]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            table = build_scifact_candidates(root, split="dev", limit=1, candidates_per_query=3)

        self.assertEqual(set(UNIFIED_COLUMNS), set(table.columns))
        gold = table[table["gold_support"].eq(1)].iloc[0]
        self.assertEqual(gold["candidate_id"], "10::1")
        self.assertIn("second sentence", gold["evidence_text"])
        self.assertGreaterEqual(len(table), 2)

    def test_build_fever_candidates_uses_official_evidence_title_sentence_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared_task_dev.jsonl"
            rows = [
                {
                    "id": 1,
                    "claim": "claim one",
                    "label": "SUPPORTS",
                    "evidence": [[[0, 0, "Page_A", 2]]],
                },
                {
                    "id": 2,
                    "claim": "claim two",
                    "label": "REFUTES",
                    "evidence": [[[0, 0, "Page_B", 4]]],
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            table = build_fever_candidates(path, split="dev", limit=1, candidates_per_query=2)

        self.assertEqual(set(UNIFIED_COLUMNS), set(table.columns))
        self.assertEqual(table.iloc[0]["dataset"], "fever")
        self.assertTrue(table["gold_support"].isin([0, 1]).all())
        self.assertIn("Page_A", table.loc[table["gold_support"].eq(1), "evidence_source"].iloc[0])

    def test_build_fever_candidates_can_use_train_plus_dev_open_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root / "shared_task_dev.jsonl"
            train = root / "train.jsonl"
            dev.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "claim": "Page C claim mentions shuttle evidence",
                        "label": "SUPPORTS",
                        "evidence": [[[0, 0, "Page_C", 1]]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train.write_text(
                json.dumps(
                    {
                        "id": 2,
                        "claim": "training claim",
                        "label": "SUPPORTS",
                        "evidence": [[[0, 0, "Shuttle_Evidence", 7]]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            pool = collect_fever_candidate_pool([dev, train])
            table = build_fever_candidates(dev, split="dev", limit=1, candidates_per_query=4, candidate_pool=pool)

        self.assertIn("Shuttle_Evidence::7", set(table["candidate_id"]))
        train_pool_row = table[table["candidate_id"].eq("Shuttle_Evidence::7")].iloc[0]
        self.assertEqual(int(train_pool_row["gold_support"]), 0)
        self.assertIn("open_pool", train_pool_row["view_payload_json"])

    def test_build_hover_candidates_uses_multi_hop_supporting_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.jsonl"
            rows = [
                {
                    "uid": "h1",
                    "claim": "Page A and Page B are linked by an event",
                    "label": "SUPPORTED",
                    "supporting_facts": [["Page_A", 1], ["Page_B", 2]],
                },
                {
                    "uid": "h2",
                    "claim": "Page C is unrelated",
                    "label": "NOT_SUPPORTED",
                    "supporting_facts": [["Page_C", 3]],
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            table = build_hover_candidates(path, split="validation", limit=1, candidates_per_query=4)

        self.assertEqual(set(UNIFIED_COLUMNS), set(table.columns))
        self.assertEqual(table.iloc[0]["dataset"], "hover")
        self.assertEqual(table["query_type"].iloc[0], "multi_hop_claim")
        self.assertEqual(int(table["gold_support"].sum()), 2)

    def test_build_hover_candidates_uses_hotpot_context_sentence_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "uid": "h1",
                        "hpqa_id": "hp1",
                        "claim": "Page A and Page B are linked by an event",
                        "label": "SUPPORTED",
                        "supporting_facts": [["Page A", 1], ["Page B", 0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            context_index = {
                "hp1": [
                    {"candidate_id": "Page A::0", "evidence_text": "Page A. distractor sentence", "evidence_source": "Page A"},
                    {"candidate_id": "Page A::1", "evidence_text": "Page A. actual support sentence", "evidence_source": "Page A"},
                    {"candidate_id": "Page B::0", "evidence_text": "Page B. second support sentence", "evidence_source": "Page B"},
                ]
            }

            table = build_hover_candidates(path, split="validation", limit=1, candidates_per_query=3, hotpot_context_index=context_index)

        gold_text = " ".join(table.loc[table["gold_support"].eq(1), "evidence_text"].tolist())
        self.assertIn("actual support sentence", gold_text)
        self.assertIn("second support sentence", gold_text)
        self.assertNotIn("Sentence 1", gold_text)

    def test_build_hover_candidates_uses_sentence_cache_when_context_misses_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "uid": "h1",
                        "hpqa_id": "hp1",
                        "claim": "Page C is linked to Page D",
                        "label": "SUPPORTED",
                        "supporting_facts": [["Page C", 2]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            context_index = {
                "hp1": [
                    {"candidate_id": "Page D::0", "evidence_text": "Page D. distractor sentence", "evidence_source": "Page D"},
                ]
            }
            sentence_cache = {"Page C::2": "Page C. recovered support sentence"}

            table = build_hover_candidates(
                path,
                split="validation",
                limit=1,
                candidates_per_query=3,
                hotpot_context_index=context_index,
                sentence_cache=sentence_cache,
            )

        gold_text = " ".join(table.loc[table["gold_support"].eq(1), "evidence_text"].tolist())
        self.assertIn("recovered support sentence", gold_text)
        self.assertNotIn("Sentence 2", gold_text)

    def test_build_hover_candidates_can_use_open_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root / "validation.jsonl"
            train = root / "train.jsonl"
            dev.write_text(
                json.dumps(
                    {
                        "uid": "h1",
                        "claim": "Page A mentions shuttle evidence",
                        "label": "SUPPORTED",
                        "supporting_facts": [["Page_A", 1]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train.write_text(
                json.dumps(
                    {
                        "uid": "h2",
                        "claim": "training claim",
                        "label": "SUPPORTED",
                        "supporting_facts": [["Shuttle_Evidence", 7]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            pool = collect_hover_candidate_pool([dev, train])
            table = build_hover_candidates(dev, split="validation", limit=1, candidates_per_query=4, candidate_pool=pool)

        self.assertIn("Shuttle_Evidence::7", set(table["candidate_id"]))
        train_pool_row = table[table["candidate_id"].eq("Shuttle_Evidence::7")].iloc[0]
        self.assertEqual(int(train_pool_row["gold_support"]), 0)
        self.assertIn("open_pool", train_pool_row["view_payload_json"])

    def test_hotpot_context_index_skips_parquet_when_engine_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hotpotqa_fullwiki_validation.parquet").write_bytes(b"placeholder")

            with patch("ipm_evidence_dataset_adapters.pd.read_parquet", side_effect=ImportError("missing engine")):
                context_index = load_hotpotqa_context_index(root)

        self.assertEqual(context_index, {})


if __name__ == "__main__":
    unittest.main()
