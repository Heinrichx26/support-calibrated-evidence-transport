from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = PROJECT / "data" / "raw_open" / "evidence_retrieval"
OUT_DEFAULT = PROJECT / "data" / "interim" / "evidence_retrieval"

UNIFIED_COLUMNS = [
    "dataset",
    "split",
    "query_id",
    "query_text",
    "query_type",
    "target_label",
    "queue_score",
    "queue_rank",
    "candidate_id",
    "evidence_text",
    "evidence_source",
    "gold_support",
    "view_payload_json",
]

SCIFACT_URLS = {
    "data.tar.gz": [
        "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
    ],
}

FEVER_URLS = {
    "train.jsonl": [
        "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl",
        "https://fever.ai/download/fever/train.jsonl",
    ],
    "shared_task_dev.jsonl": [
        "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl",
        "https://fever.ai/download/fever/shared_task_dev.jsonl",
    ],
}

HOTPOTQA_CONTEXT_FILENAMES = [
    "hotpotqa_fullwiki_validation.parquet",
    "hotpotqa_fullwiki_train_0000.parquet",
    "hotpotqa_fullwiki_train_0001.parquet",
]

SMOKE_LIMITS = {"scifact": 200, "fever": 1000, "hover": 500, "bts": 200}
FULL_LIMITS = {"scifact": None, "fever": None, "hover": None, "bts": None}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_file(path: Path, urls: list[str]) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
            if path.stat().st_size > 0:
                return path
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Could not download {path.name}. Tried: {' | '.join(errors)}")


def ensure_scifact_files(raw_root: Path = RAW_DEFAULT / "scifact") -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    for name, urls in SCIFACT_URLS.items():
        download_file(raw_root / name, urls)
    expected = [raw_root / "corpus.jsonl", raw_root / "claims_train.jsonl", raw_root / "claims_dev.jsonl"]
    if not all(path.exists() for path in expected):
        with tarfile.open(raw_root / "data.tar.gz", "r:gz") as archive:
            archive.extractall(raw_root)
        nested = raw_root / "data"
        if nested.exists():
            for path in nested.glob("*.jsonl"):
                target = raw_root / path.name
                if not target.exists():
                    target.write_bytes(path.read_bytes())
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"SciFact archive did not contain expected files: {missing}")
    return raw_root


def ensure_fever_files(raw_root: Path = RAW_DEFAULT / "fever") -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    for name, urls in FEVER_URLS.items():
        download_file(raw_root / name, urls)
    return raw_root


def validate_unified_schema(table: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in UNIFIED_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"Unified evidence schema missing columns: {missing}")
    out = table.loc[:, UNIFIED_COLUMNS].copy()
    out["query_id"] = out["query_id"].astype(str)
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["query_text"] = out["query_text"].fillna("").astype(str)
    out["evidence_text"] = out["evidence_text"].fillna("").astype(str)
    out["evidence_source"] = out["evidence_source"].fillna("").astype(str)
    out["queue_score"] = pd.to_numeric(out["queue_score"], errors="coerce").fillna(0.0)
    out["queue_rank"] = pd.to_numeric(out["queue_rank"], errors="coerce").fillna(0).astype(int)
    out["gold_support"] = pd.to_numeric(out["gold_support"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["view_payload_json"] = out["view_payload_json"].map(lambda value: value if isinstance(value, str) and value else "{}")
    return out


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", str(text).lower()))


def lexical_overlap(query: str, evidence: str) -> float:
    q = tokens(query)
    e = tokens(evidence)
    if not q or not e:
        return 0.0
    return len(q & e) / math.sqrt(len(q) * len(e))


def top_candidates_by_overlap(query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda row: lexical_overlap(query, row["evidence_text"]), reverse=True)
    return ranked[:limit]


def build_candidate_token_index(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    token_to_indices: dict[str, set[int]] = {}
    candidate_tokens: list[set[str]] = []
    for idx, row in enumerate(candidates):
        toks = tokens(str(row.get("evidence_text", "")))
        candidate_tokens.append(toks)
        for token in toks:
            token_to_indices.setdefault(token, set()).add(idx)
    return {"token_to_indices": token_to_indices, "candidate_tokens": candidate_tokens}


def top_candidates_by_index(
    query: str,
    candidates: list[dict[str, Any]],
    index: dict[str, Any],
    limit: int,
    min_pool: int = 500,
) -> list[dict[str, Any]]:
    query_tokens = tokens(query)
    token_to_indices: dict[str, set[int]] = index["token_to_indices"]
    candidate_tokens: list[set[str]] = index["candidate_tokens"]
    candidate_ids: set[int] = set()
    for token in query_tokens:
        candidate_ids.update(token_to_indices.get(token, set()))
    if len(candidate_ids) < min_pool:
        candidate_ids.update(range(min(len(candidates), min_pool)))
    scored = []
    denom_query = math.sqrt(max(len(query_tokens), 1))
    for idx in candidate_ids:
        e_tokens = candidate_tokens[idx]
        score = len(query_tokens & e_tokens) / max(denom_query * math.sqrt(max(len(e_tokens), 1)), 1e-12)
        scored.append((score, idx))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [candidates[idx] for _, idx in scored[:limit]]


def _scifact_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    docs = read_jsonl(corpus_path)
    rows: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id"))
        title = str(doc.get("title", ""))
        abstract = doc.get("abstract", []) or []
        for idx, sentence in enumerate(abstract):
            evidence_text = f"{title}. {sentence}".strip()
            rows.append(
                {
                    "candidate_id": f"{doc_id}::{idx}",
                    "evidence_text": evidence_text,
                    "evidence_source": doc_id,
                    "doc_id": doc_id,
                    "sentence_id": idx,
                }
            )
    return rows


def _scifact_gold(claim: dict[str, Any]) -> tuple[set[str], str]:
    gold: set[str] = set()
    labels: list[str] = []
    evidence = claim.get("evidence", {}) or {}
    for doc_id, groups in evidence.items():
        for group in groups or []:
            labels.append(str(group.get("label", "")))
            for sent in group.get("sentences", []) or []:
                gold.add(f"{doc_id}::{sent}")
    label = next((item for item in labels if item), str(claim.get("label", "")))
    return gold, label or "UNKNOWN"


def build_scifact_candidates(
    root: Path,
    split: str = "dev",
    limit: int | None = None,
    candidates_per_query: int = 50,
) -> pd.DataFrame:
    corpus = _scifact_corpus(root / "corpus.jsonl")
    corpus_index = build_candidate_token_index(corpus)
    claims = read_jsonl(root / f"claims_{split}.jsonl", limit=limit)
    rows: list[dict[str, Any]] = []
    for rank, claim in enumerate(claims, start=1):
        query_id = str(claim.get("id"))
        query = str(claim.get("claim", ""))
        gold_ids, label = _scifact_gold(claim)
        candidates = top_candidates_by_index(query, corpus, corpus_index, candidates_per_query)
        gold_candidates = [row for row in corpus if row["candidate_id"] in gold_ids]
        merged = {row["candidate_id"]: row for row in candidates}
        for row in gold_candidates:
            merged[row["candidate_id"]] = row
        ordered = sorted(merged.values(), key=lambda row: (row["candidate_id"] not in gold_ids, -lexical_overlap(query, row["evidence_text"])))
        for cand in ordered[: max(candidates_per_query, len(gold_candidates))]:
            payload = {
                "lexical": query,
                "dense": cand["evidence_text"],
                "entity": cand["evidence_source"],
                "source_context": cand["evidence_source"],
            }
            rows.append(
                {
                    "dataset": "scifact",
                    "split": split,
                    "query_id": query_id,
                    "query_text": query,
                    "query_type": "claim",
                    "target_label": label,
                    "queue_score": 1.0,
                    "queue_rank": rank,
                    "candidate_id": cand["candidate_id"],
                    "evidence_text": cand["evidence_text"],
                    "evidence_source": cand["evidence_source"],
                    "gold_support": int(cand["candidate_id"] in gold_ids),
                    "view_payload_json": json.dumps(payload, ensure_ascii=False),
                }
            )
    return validate_unified_schema(pd.DataFrame(rows))


def fever_evidence_pairs(row: dict[str, Any]) -> set[str]:
    pairs: set[str] = set()
    for group in row.get("evidence", []) or []:
        for item in group or []:
            if len(item) >= 4 and item[2] is not None and item[3] is not None:
                pairs.add(f"{item[2]}::{item[3]}")
    return pairs


def load_fever_sentence_cache(raw_root: Path = RAW_DEFAULT / "fever") -> dict[str, str]:
    path = raw_root / "wikipedia_sentence_cache.json"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items() if value}
    return {}


def _fever_candidate_pool(rows: list[dict[str, Any]], sentence_cache: dict[str, str] | None = None) -> list[dict[str, Any]]:
    cache = sentence_cache or {}
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        for pair in fever_evidence_pairs(row):
            title, sentence_id = pair.rsplit("::", 1)
            clean_title = title.replace("_", " ")
            cached_text = cache.get(pair)
            seen[pair] = {
                "candidate_id": pair,
                "evidence_text": cached_text or f"{clean_title}. Sentence {sentence_id}.",
                "evidence_source": title,
            }
    return list(seen.values())


def build_fever_candidates(
    path: Path,
    split: str = "dev",
    limit: int | None = None,
    candidates_per_query: int = 50,
    candidate_pool: list[dict[str, Any]] | None = None,
    sentence_cache: dict[str, str] | None = None,
) -> pd.DataFrame:
    claims = read_jsonl(path, limit=limit)
    pool = candidate_pool if candidate_pool is not None else _fever_candidate_pool(claims, sentence_cache=sentence_cache)
    pool_index = build_candidate_token_index(pool)
    pool_scope = "open_pool" if candidate_pool is not None else "split_pool"
    rows: list[dict[str, Any]] = []
    for rank, claim in enumerate(claims, start=1):
        query_id = str(claim.get("id"))
        query = str(claim.get("claim", ""))
        label = str(claim.get("label", "UNKNOWN"))
        gold_ids = fever_evidence_pairs(claim)
        candidates = top_candidates_by_index(query, pool, pool_index, candidates_per_query)
        gold_candidates = [row for row in pool if row["candidate_id"] in gold_ids]
        merged = {row["candidate_id"]: row for row in candidates}
        for row in gold_candidates:
            merged[row["candidate_id"]] = row
        ordered = sorted(merged.values(), key=lambda row: (row["candidate_id"] not in gold_ids, -lexical_overlap(query, row["evidence_text"])))
        for cand in ordered[: max(candidates_per_query, len(gold_candidates))]:
            payload = {
                "lexical": query,
                "dense": cand["evidence_text"],
                "entity": cand["evidence_source"],
                "source_context": cand["evidence_source"],
                "candidate_scope": pool_scope,
            }
            rows.append(
                {
                    "dataset": "fever",
                    "split": split,
                    "query_id": query_id,
                    "query_text": query,
                    "query_type": "claim",
                    "target_label": label,
                    "queue_score": 1.0,
                    "queue_rank": rank,
                    "candidate_id": cand["candidate_id"],
                    "evidence_text": cand["evidence_text"],
                    "evidence_source": cand["evidence_source"],
                    "gold_support": int(cand["candidate_id"] in gold_ids),
                    "view_payload_json": json.dumps(payload, ensure_ascii=False),
                }
            )
    return validate_unified_schema(pd.DataFrame(rows))


def collect_fever_candidate_pool(
    paths: list[Path],
    limit_per_path: int | None = None,
    sentence_cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Collect a de-duplicated FEVER title-sentence evidence pool from one or more claim files."""
    seen: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path, limit=limit_per_path):
            for candidate in _fever_candidate_pool([row], sentence_cache=sentence_cache):
                seen[candidate["candidate_id"]] = candidate
    return list(seen.values())


def _read_hover_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path, limit=limit)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", [])
        return rows[:limit] if limit is not None else rows
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        if limit is not None:
            frame = frame.head(limit)
        return frame.to_dict(orient="records")
    raise ValueError(f"Unsupported HoVer file format: {path}")


def _hover_split_path(root: Path, split: str) -> Path:
    aliases = {
        "dev": ["dev", "validation", "valid"],
        "validation": ["validation", "dev", "valid"],
        "train": ["train"],
        "test": ["test"],
    }
    names = aliases.get(split, [split])
    for name in names:
        for suffix in [".jsonl", ".json", ".parquet"]:
            path = root / f"{name}{suffix}"
            if path.exists():
                return path
    for name in names:
        matches = sorted(root.glob(f"**/{name}*.parquet")) + sorted(root.glob(f"**/{name}*.jsonl")) + sorted(root.glob(f"**/{name}*.json"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No HoVer split file found for {split} under {root}")


def ensure_hover_files(raw_root: Path = RAW_DEFAULT / "hover") -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    try:
        _hover_split_path(raw_root, "validation")
        return raw_root
    except FileNotFoundError:
        pass
    try:
        from datasets import load_dataset

        dataset = load_dataset("vincentkoc/hover-parquet")
        for split_name, rows in dataset.items():
            target = raw_root / f"{split_name}.jsonl"
            if target.exists() and target.stat().st_size > 0:
                continue
            write_jsonl(target, rows)
        return raw_root
    except Exception as exc:
        raise RuntimeError(
            "HoVer files are not available locally and automatic download failed. "
            f"Place train/validation JSONL, JSON, or Parquet files under {raw_root}. Original error: {exc}"
        ) from exc


def hover_supporting_facts(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", row.get("supporting_facts_json", []))
    if isinstance(facts, str):
        try:
            facts = json.loads(facts)
        except json.JSONDecodeError:
            facts = []
    if facts is None:
        facts_iter = []
    else:
        facts_iter = list(facts)
    pairs: set[str] = set()
    for item in facts_iter:
        if isinstance(item, dict):
            title = item.get("title") or item.get("page") or item.get("doc_id") or item.get("key")
            sentence_id = item.get("sent_id", item.get("sentence_id", item.get("sid", item.get("value"))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sentence_id = item[0], item[1]
        else:
            continue
        if title is not None and sentence_id is not None:
            pairs.add(f"{str(title)}::{str(sentence_id)}")
    return pairs


def _hotpot_context_sentence_candidates(context: Any) -> list[dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    raw_titles = context.get("title", [])
    raw_sentence_groups = context.get("sentences", [])
    titles = [] if raw_titles is None else list(raw_titles)
    sentence_groups = [] if raw_sentence_groups is None else list(raw_sentence_groups)
    candidates: list[dict[str, Any]] = []
    for title, sentences in zip(titles, sentence_groups):
        title_text = str(title)
        sentence_list = [] if sentences is None else list(sentences)
        for sent_id, sentence in enumerate(sentence_list):
            sentence_text = str(sentence).strip()
            if not sentence_text:
                continue
            candidate_id = f"{title_text}::{sent_id}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "evidence_text": f"{title_text}. {sentence_text}",
                    "evidence_source": title_text,
                }
            )
    return candidates


def load_hotpotqa_context_index(raw_root: Path = RAW_DEFAULT / "hotpotqa") -> dict[str, list[dict[str, Any]]]:
    """Load local HotpotQA contexts keyed by question id."""
    context_index: dict[str, list[dict[str, Any]]] = {}
    for name in HOTPOTQA_CONTEXT_FILENAMES:
        path = raw_root / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_parquet(path, columns=["id", "context"])
        except ImportError:
            continue
        for row in frame.itertuples(index=False):
            hpqa_id = str(getattr(row, "id"))
            if hpqa_id in context_index:
                continue
            candidates = _hotpot_context_sentence_candidates(getattr(row, "context"))
            if candidates:
                context_index[hpqa_id] = candidates
    return context_index


def load_hover_sentence_cache(raw_root: Path = RAW_DEFAULT / "hover") -> dict[str, str]:
    path = raw_root / "wikipedia_sentence_cache.json"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items() if value}
    if isinstance(data, list):
        cache: dict[str, str] = {}
        for row in data:
            if isinstance(row, dict) and row.get("candidate_id") and row.get("evidence_text"):
                cache[str(row["candidate_id"])] = str(row["evidence_text"])
        return cache
    return {}


def _hover_candidates_from_context(
    row: dict[str, Any],
    context_index: dict[str, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if not context_index:
        return []
    hpqa_id = row.get("hpqa_id")
    if hpqa_id is None:
        return []
    return list(context_index.get(str(hpqa_id), []))


def _hover_candidate_pool(
    rows: list[dict[str, Any]],
    context_index: dict[str, list[dict[str, Any]]] | None = None,
    sentence_cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    sentence_cache = sentence_cache or {}
    for row in rows:
        for candidate in _hover_candidates_from_context(row, context_index):
            seen[candidate["candidate_id"]] = candidate
        for pair in hover_supporting_facts(row):
            title, sentence_id = pair.rsplit("::", 1)
            clean_title = title.replace("_", " ")
            cached_text = sentence_cache.get(pair)
            seen.setdefault(
                pair,
                {
                    "candidate_id": pair,
                    "evidence_text": cached_text or f"{clean_title}. Sentence {sentence_id}.",
                    "evidence_source": title,
                },
            )
    return list(seen.values())


def collect_hover_candidate_pool(
    paths: list[Path],
    limit_per_path: int | None = None,
    context_index: dict[str, list[dict[str, Any]]] | None = None,
    sentence_cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for candidate in _hover_candidate_pool(
            _read_hover_rows(path, limit=limit_per_path),
            context_index=context_index,
            sentence_cache=sentence_cache,
        ):
            seen[candidate["candidate_id"]] = candidate
    return list(seen.values())


def build_hover_candidates(
    path: Path,
    split: str = "validation",
    limit: int | None = None,
    candidates_per_query: int = 50,
    candidate_pool: list[dict[str, Any]] | None = None,
    hotpot_context_index: dict[str, list[dict[str, Any]]] | None = None,
    sentence_cache: dict[str, str] | None = None,
) -> pd.DataFrame:
    claims = _read_hover_rows(path, limit=limit)
    fallback_pool = candidate_pool if candidate_pool is not None else _hover_candidate_pool(claims, sentence_cache=sentence_cache)
    fallback_index = build_candidate_token_index(fallback_pool) if fallback_pool else {"token_to_indices": {}, "candidate_tokens": []}
    pool_scope = "open_pool" if candidate_pool is not None else "split_pool"
    rows: list[dict[str, Any]] = []
    for rank, claim in enumerate(claims, start=1):
        query_id = str(claim.get("uid", claim.get("id", rank)))
        query = str(claim.get("claim", claim.get("question", "")))
        label = str(claim.get("label", claim.get("answer", "UNKNOWN")))
        gold_ids = hover_supporting_facts(claim)
        context_pool = _hover_candidate_pool([claim], context_index=hotpot_context_index, sentence_cache=sentence_cache)
        active_pool = context_pool if context_pool else fallback_pool
        active_index = build_candidate_token_index(active_pool) if context_pool else fallback_index
        candidates = top_candidates_by_index(query, active_pool, active_index, candidates_per_query) if active_pool else []
        if candidate_pool is not None and len(candidates) < candidates_per_query:
            existing_ids = {row["candidate_id"] for row in candidates}
            supplement = top_candidates_by_index(query, fallback_pool, fallback_index, candidates_per_query)
            candidates.extend(row for row in supplement if row["candidate_id"] not in existing_ids)
        gold_candidates = [row for row in active_pool if row["candidate_id"] in gold_ids]
        merged = {row["candidate_id"]: row for row in candidates}
        for row in gold_candidates:
            merged[row["candidate_id"]] = row
        ordered = sorted(merged.values(), key=lambda row: (row["candidate_id"] not in gold_ids, -lexical_overlap(query, row["evidence_text"])))
        for cand in ordered[: max(candidates_per_query, len(gold_candidates))]:
            payload = {
                "lexical": query,
                "dense": cand["evidence_text"],
                "entity": cand["evidence_source"],
                "source_context": cand["evidence_source"],
                "candidate_scope": pool_scope,
            }
            rows.append(
                {
                    "dataset": "hover",
                    "split": split,
                    "query_id": query_id,
                    "query_text": query,
                    "query_type": "multi_hop_claim",
                    "target_label": label,
                    "queue_score": 1.0,
                    "queue_rank": rank,
                    "candidate_id": cand["candidate_id"],
                    "evidence_text": cand["evidence_text"],
                    "evidence_source": cand["evidence_source"],
                    "gold_support": int(cand["candidate_id"] in gold_ids),
                    "view_payload_json": json.dumps(payload, ensure_ascii=False),
                }
            )
    return validate_unified_schema(pd.DataFrame(rows))


def build_bts_candidates(
    source: Path = PROJECT / "results" / "ipm_event_evidence_benchmark_final" / "ipm_event_evidence_benchmark_judgement.csv",
    limit: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(source)
    df = df[df["evidence_method"].isin(["teacher", "mvec_tg"])].copy()
    if limit is not None:
        df = df.head(limit)
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        protocol = getattr(row, "protocol")
        method = getattr(row, "evidence_method")
        supported = int(getattr(row, "supported", 0))
        rows.append(
            {
                "dataset": "bts",
                "split": str(protocol),
                "query_id": f"{protocol}-{method}",
                "query_text": f"{protocol} fixed aircraft-rotation review queue",
                "query_type": "structured_event_queue",
                "target_label": "severe_late_aircraft_delay",
                "queue_score": float(getattr(row, "evidence_hit", 0.0)),
                "queue_rank": rank,
                "candidate_id": f"{protocol}-{method}-summary",
                "evidence_text": f"{method} supported {supported:,} severe events in the fixed queue.",
                "evidence_source": "BTS validated protocol summary",
                "gold_support": int(supported > 0),
                "view_payload_json": json.dumps({"protocol": protocol, "method": method}, ensure_ascii=False),
            }
        )
    return validate_unified_schema(pd.DataFrame(rows))


def build_dataset(dataset: str, mode: str, out: Path, raw_root: Path = RAW_DEFAULT, candidates_per_query: int = 50) -> Path:
    dataset = dataset.lower()
    limit = SMOKE_LIMITS[dataset] if mode == "smoke" else FULL_LIMITS[dataset]
    out.mkdir(parents=True, exist_ok=True)
    if dataset == "scifact":
        root = ensure_scifact_files(raw_root / "scifact")
        table = build_scifact_candidates(root, split="dev", limit=limit, candidates_per_query=candidates_per_query)
    elif dataset == "fever":
        root = ensure_fever_files(raw_root / "fever")
        fever_cache = load_fever_sentence_cache(root)
        pool = collect_fever_candidate_pool(
            [root / "train.jsonl", root / "shared_task_dev.jsonl"],
            sentence_cache=fever_cache,
        )
        table = build_fever_candidates(
            root / "shared_task_dev.jsonl",
            split="dev",
            limit=limit,
            candidates_per_query=candidates_per_query,
            candidate_pool=pool,
            sentence_cache=fever_cache,
        )
    elif dataset == "hover":
        root = ensure_hover_files(raw_root / "hover")
        validation = _hover_split_path(root, "validation")
        train_candidates = []
        try:
            train_candidates.append(_hover_split_path(root, "train"))
        except FileNotFoundError:
            pass
        context_index = load_hotpotqa_context_index(raw_root / "hotpotqa")
        sentence_cache = load_hover_sentence_cache(root)
        pool = None if context_index else collect_hover_candidate_pool([*train_candidates, validation], sentence_cache=sentence_cache)
        table = build_hover_candidates(
            validation,
            split="validation",
            limit=limit,
            candidates_per_query=candidates_per_query,
            candidate_pool=pool,
            hotpot_context_index=context_index,
            sentence_cache=sentence_cache,
        )
    elif dataset == "bts":
        table = build_bts_candidates(limit=limit)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    path = out / f"{dataset}_{mode}_candidates_k{candidates_per_query}.csv"
    table.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bts", "scifact", "fever", "hover"], required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--candidates-per-query", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_dataset(args.dataset, args.mode, args.out, args.raw_root, args.candidates_per_query)
    print(path)


if __name__ == "__main__":
    main()
