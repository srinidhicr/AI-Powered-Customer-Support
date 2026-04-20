# scripts/test_retrieval.py
#
# Tests retrieval pipeline in isolation — zero LLM calls, zero API cost.
# Run: python -m scripts.test_retrieval
#
# What this tests:
#   1. KB health check (chunk count, category distribution)
#   2. Dense search (Qdrant) quality per query
#   3. Sparse search (BM25) quality per query
#   4. Hybrid RRF + rerank quality per query
#   5. Side-by-side comparison table

import sys, os, json, textwrap
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── test queries — one per category ──────────────────────────────────────────
TEST_CASES = [
    {
        "label"   : "Technical — WiFi / firmware",
        "query"   : "My laptop cannot connect to the office WiFi after a router firmware update. Restarting doesn't help.",
        "category": "Technical",
        "keywords": ["wifi", "network", "router", "firmware", "connectivity"],
    },
    {
        "label"   : "Technical — cloud outage",
        "query"   : "We're experiencing a full outage on our cloud storage platform. Dashboard won't load.",
        "category": "Technical",
        "keywords": ["cloud", "outage", "dashboard", "access", "storage"],
    },
    {
        "label"   : "Billing — duplicate charge",
        "query"   : "I was charged twice for my subscription this month. Please advise on refund.",
        "category": "Billing and Payments",
        "keywords": ["charged", "duplicate", "subscription", "refund", "billing"],
    },
    {
        "label"   : "Product Inquiry — CRM integration",
        "query"   : "Does your analytics dashboard integrate with Salesforce and HubSpot for bidirectional data sync?",
        "category": "Product Inquiry",
        "keywords": ["integrate", "salesforce", "hubspot", "analytics", "crm"],
    },
    {
        "label"   : "Returns — defective device",
        "query"   : "I'd like to return a device purchased 3 weeks ago. Build quality not as described.",
        "category": "Returns and Exchanges",
        "keywords": ["return", "device", "quality", "purchase", "exchange"],
    },
]

# ── RRF (copy from retrieve.py — no import to avoid LLM loading) ─────────────
def _rrf(dense_results: list, sparse_results: list, k: int = 60) -> list:
    scores   = {}
    all_docs = {}
    for rank, doc in enumerate(dense_results):
        did = doc['id']
        scores[did]   = scores.get(did, 0) + 1 / (k + rank + 1)
        all_docs[did] = doc
    for rank, doc in enumerate(sparse_results):
        did = doc['id']
        scores[did]   = scores.get(did, 0) + 1 / (k + rank + 1)
        all_docs[did] = doc
    return sorted(all_docs.values(), key=lambda d: scores[d['id']], reverse=True)


# ── display helpers ───────────────────────────────────────────────────────────
def sep(title: str = '', width: int = 72):
    if title:
        pad = (width - len(title) - 2) // 2
        print('\n' + '─' * pad + f' {title} ' + '─' * (width - pad - len(title) - 2))
    else:
        print('─' * width)


def show_hits(hits: list, label: str, keywords: list, top_n: int = 5):
    print(f"\n  [{label}]")
    if not hits:
        print("    (no results)")
        return
    for i, doc in enumerate(hits[:top_n]):
        text    = doc.get("answer", doc.get("text", ""))[:140].replace("\n", " ")
        score   = doc.get("score", 0)
        subject = doc.get("subject", "")[:40]
        # Highlight keyword matches
        matched = [kw for kw in keywords if kw.lower() in text.lower()]
        kw_str  = f"  ✓ keywords: {matched}" if matched else "  ✗ no keyword match"
        print(f"    {i+1}. score={score:.4f} | subject='{subject}'")
        print(f"       {textwrap.fill(text, width=65, subsequent_indent='       ')}")
        print(f"       {kw_str}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    from src.retrieval.qdrant_store import dense_search
    from src.retrieval.bm25_store   import sparse_search
    from src.retrieval.reranker     import rerank
    from qdrant_client import QdrantClient
    from configs.config import config

    # ── 1. KB health check ───────────────────────────────────────────────────
    sep("KB HEALTH CHECK")
    client = QdrantClient(host=config.qdrant_url, port=config.qdrant_port)
    info   = client.get_collection(config.collection_name)
    print(f"  Collection : {config.collection_name}")
    print(f"  Total chunks indexed: {info.points_count}")

    # Category distribution from a scroll sample
    sample, _ = client.scroll(
        collection_name=config.collection_name,
        limit=5000,
        with_payload=True,
        with_vectors=False
    )
    cat_counts = {}
    for pt in sample:
        cat = pt.payload.get("category", "MISSING")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"\n  Category distribution (sample of {len(sample)}):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<30} {cnt}")

    if info.points_count == 0:
        print("\n  ⚠ KB is empty — run build_kb_script first")
        return

    # ── 2. Per-query retrieval test ──────────────────────────────────────────
    results_summary = []

    for tc in TEST_CASES:
        sep(tc["label"])
        query    = tc["query"]
        category = tc["category"]
        keywords = tc["keywords"]

        print(f"  Query    : {textwrap.fill(query, width=65, subsequent_indent='             ')}")
        print(f"  Category : {category}")

        # Dense
        dense  = dense_search(query, category, top_k=20)
        # Sparse
        sparse = sparse_search(query, category, top_k=20)
        # Fused
        fused  = _rrf(dense, sparse)
        # Reranked
        final  = rerank(query, fused, top_k=5)

        show_hits(dense[:5],  "DENSE  (Qdrant)", keywords)
        show_hits(sparse[:5], "SPARSE (BM25)  ", keywords)
        show_hits(final,      "FINAL  (hybrid+rerank)", keywords)

        # Score for summary table
        def kw_hit_rate(hits, top_k=5):
            if not hits:
                return 0.0
            matched = sum(
                1 for doc in hits[:top_k]
                if any(kw.lower() in doc.get("answer", doc.get("text","")).lower()
                       for kw in keywords)
            )
            return matched / min(top_k, len(hits))

        results_summary.append({
            "label"        : tc["label"],
            "dense_hr"     : kw_hit_rate(dense),
            "sparse_hr"    : kw_hit_rate(sparse),
            "hybrid_hr"    : kw_hit_rate(final),
            "n_dense"      : len(dense),
            "n_sparse"     : len(sparse),
            "n_final"      : len(final),
        })

    # ── 3. Summary table ─────────────────────────────────────────────────────
    sep("RETRIEVAL QUALITY SUMMARY")
    print(f"\n  {'Query':<35} {'Dense HR':>9} {'Sparse HR':>10} {'Hybrid HR':>10} {'Final n':>8}")
    sep()
    for r in results_summary:
        print(
            f"  {r['label']:<35} "
            f"{r['dense_hr']:>8.0%}  "
            f"{r['sparse_hr']:>9.0%}  "
            f"{r['hybrid_hr']:>9.0%}  "
            f"{r['n_final']:>7}"
        )
    sep()

    avg_dense  = sum(r['dense_hr']  for r in results_summary) / len(results_summary)
    avg_sparse = sum(r['sparse_hr'] for r in results_summary) / len(results_summary)
    avg_hybrid = sum(r['hybrid_hr'] for r in results_summary) / len(results_summary)
    print(f"\n  {'AVERAGE':<35} {avg_dense:>8.0%}  {avg_sparse:>9.0%}  {avg_hybrid:>9.0%}")

    print("""
  HR = Keyword Hit Rate @ 5 (fraction of top-5 results containing
       at least one expected keyword for that query type).
  This is a proxy metric — not ground truth MRR. Use for quick
  diagnosis of whether retrieval is on-topic.
""")

    # ── 4. Diagnosis advice ──────────────────────────────────────────────────
    sep("DIAGNOSIS")
    if avg_hybrid < 0.4:
        print("  ⚠ Hybrid HR < 40% — retrieval is poorly aligned.")
        print("  Likely causes:")
        print("    • KB chunks still contain near-duplicates (lower dedup threshold to 0.70)")
        print("    • Category filter mismatch (check category names in KB vs classifier output)")
        print("    • Dataset answers are too generic to match specific queries")
    elif avg_hybrid < 0.6:
        print("  △ Hybrid HR 40-60% — moderate quality, room for improvement.")
        print("  Try: lower dedup threshold, or add subject to embedded text")
    else:
        print("  ✓ Hybrid HR > 60% — retrieval is reasonably on-topic.")
        print("  Proceed to pipeline testing.")


if __name__ == '__main__':
    main()