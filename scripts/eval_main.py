"""
Main non-RAGAS evaluation entrypoint.

This is the recommended standard evaluation file to keep alongside
`scripts/eval_ragas.py`.

What it covers:
1. Dataset-backed retrieval metrics using original KB queries
2. Dense vs sparse vs hybrid comparison
3. Standard retrieval metrics: Hit@1, Hit@3, Hit@5, MRR@5
4. Optional sample miss inspection

Run:
    python -m scripts.eval_main
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.bm25_store import sparse_search
from src.retrieval.qdrant_store import dense_search
from src.retrieval.reranker import rerank

MERGED_PATH = "data/knowledge_base/kb_chunks_merged.jsonl"
BASE_PATH = "data/knowledge_base/kb_chunks.jsonl"
SAMPLE_PER_CATEGORY = 40
TOP_K = 5


def _rrf(dense_results: list, sparse_results: list, k: int = 60) -> list:
    scores = {}
    all_docs = {}
    for rank, doc in enumerate(dense_results):
        did = doc["id"]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
        all_docs[did] = doc
    for rank, doc in enumerate(sparse_results):
        did = doc["id"]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
        all_docs[did] = doc
    return sorted(all_docs.values(), key=lambda d: scores[d["id"]], reverse=True)


def _load_queries() -> list:
    path = MERGED_PATH if os.path.exists(MERGED_PATH) else BASE_PATH
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not row["id"].startswith("chunk_"):
                continue
            query = str(row.get("query", "") or "").strip()
            category = str(row.get("category", "") or "").strip()
            if query and category:
                rows.append({
                    "id": row["id"],
                    "query": query,
                    "category": category,
                    "subject": str(row.get("subject", "") or "").strip(),
                    "tags": row.get("tags", []),
                })
    return rows


def _sample_queries(rows: list) -> list:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    sample = []
    rng = random.Random(42)
    for category, items in grouped.items():
        rng.shuffle(items)
        sample.extend(items[: min(SAMPLE_PER_CATEGORY, len(items))])
    return sample


def _rank_of(target_id: str, docs: list, top_k: int) -> int | None:
    for idx, doc in enumerate(docs[:top_k], start=1):
        if doc["id"] == target_id:
            return idx
    return None


def _score(prefix: str, rank: int | None) -> dict:
    return {
        f"{prefix}_hit@1": 1 if rank == 1 else 0,
        f"{prefix}_hit@3": 1 if rank is not None and rank <= 3 else 0,
        f"{prefix}_hit@5": 1 if rank is not None and rank <= 5 else 0,
        f"{prefix}_precision@1": 1.0 if rank == 1 else 0.0,
        f"{prefix}_precision@3": (1 / 3) if rank is not None and rank <= 3 else 0.0,
        f"{prefix}_precision@5": (1 / 5) if rank is not None and rank <= 5 else 0.0,
        f"{prefix}_mrr@5": 0 if rank is None else 1 / rank,
    }


def _avg(counter: Counter, key: str, denom: int) -> float:
    return counter[key] / max(1, denom)


def main():
    rows = _load_queries()
    sample = _sample_queries(rows)

    print("=" * 72)
    print("DATASET RETRIEVAL EVALUATION")
    print("=" * 72)
    print(f"Loaded {len(rows)} original KB chunks with queries")
    print(f"Evaluating {len(sample)} sampled queries across categories")

    totals = Counter()
    category_totals = defaultdict(Counter)
    examples = []

    for row in sample:
        query = row["query"]
        category = row["category"]
        target_id = row["id"]

        dense = dense_search(query, category, top_k=20)
        sparse = sparse_search(query, category, top_k=20)
        hybrid = rerank(query, _rrf(dense, sparse), top_k=TOP_K)

        ranks = {
            "dense": _rank_of(target_id, dense, TOP_K),
            "sparse": _rank_of(target_id, sparse, TOP_K),
            "hybrid": _rank_of(target_id, hybrid, TOP_K),
        }

        for method, rank in ranks.items():
            for key, value in _score(method, rank).items():
                totals[key] += value
                category_totals[category][key] += value

        if len(examples) < 8 and ranks["hybrid"] is None:
            examples.append({
                "category": category,
                "query": query[:140],
                "subject": row["subject"][:60],
                "tags": row["tags"][:3],
                "top_subjects": [doc.get("subject", "")[:50] for doc in hybrid[:3]],
            })

    print("\nOverall")
    print(f"{'Method':<10} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'P@3':>8} {'P@5':>8} {'MRR@5':>8}")
    for method in ["dense", "sparse", "hybrid"]:
        print(
            f"{method:<10} "
            f"{_avg(totals, f'{method}_hit@1', len(sample)):>8.3f} "
            f"{_avg(totals, f'{method}_hit@3', len(sample)):>8.3f} "
            f"{_avg(totals, f'{method}_hit@5', len(sample)):>8.3f} "
            f"{_avg(totals, f'{method}_precision@3', len(sample)):>8.3f} "
            f"{_avg(totals, f'{method}_precision@5', len(sample)):>8.3f} "
            f"{_avg(totals, f'{method}_mrr@5', len(sample)):>8.3f}"
        )

    print("\nBy category")
    print(f"{'Category':<28} {'Method':<10} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'P@3':>8} {'P@5':>8} {'MRR@5':>8}")
    for category, counter in sorted(category_totals.items()):
        denom = min(SAMPLE_PER_CATEGORY, sum(1 for row in rows if row["category"] == category))
        for method in ["dense", "sparse", "hybrid"]:
            print(
                f"{category:<28} {method:<10} "
                f"{_avg(counter, f'{method}_hit@1', denom):>8.3f} "
                f"{_avg(counter, f'{method}_hit@3', denom):>8.3f} "
                f"{_avg(counter, f'{method}_hit@5', denom):>8.3f} "
                f"{_avg(counter, f'{method}_precision@3', denom):>8.3f} "
                f"{_avg(counter, f'{method}_precision@5', denom):>8.3f} "
                f"{_avg(counter, f'{method}_mrr@5', denom):>8.3f}"
            )

    if examples:
        print("\nSample misses from hybrid")
        for example in examples:
            print(f"- [{example['category']}] {example['query']}")
            print(f"  subject={example['subject']}")
            print(f"  tags={example['tags']}")
            print(f"  top_subjects={example['top_subjects']}")

    print("\nInterpretation")
    print("- Hit@1: exact source chunk retrieved at rank 1")
    print("- Hit@3 / Hit@5: source chunk retrieved somewhere in top 3 / top 5")
    print("- Precision@3 / Precision@5: source-chunk precision under a single-relevant-doc assumption")
    print("- MRR@5: ranking quality of the first correct source chunk")
    print("- Sparse being perfect often means lexical overlap in your templated dataset is very high")


if __name__ == "__main__":
    main()
