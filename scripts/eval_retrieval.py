# scripts/eval_retrieval_proper.py
#
# More honest retrieval evaluation using:
# 1. Mean Reciprocal Rank (MRR) — where does the first relevant doc appear?
# 2. Precision@K — what fraction of top-K are relevant?
# 3. Answer overlap — do retrieved docs contain terms from the KB answer?
# 4. Category purity — are all retrieved docs from the right category?

import sys, os, json, textwrap
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.qdrant_store import dense_search
from src.retrieval.bm25_store   import sparse_search
from src.retrieval.reranker     import rerank

def _rrf(dense, sparse, k=60):
    scores, all_docs = {}, {}
    for rank, doc in enumerate(dense):
        did = doc['id']; scores[did] = scores.get(did,0) + 1/(k+rank+1); all_docs[did]=doc
    for rank, doc in enumerate(sparse):
        did = doc['id']; scores[did] = scores.get(did,0) + 1/(k+rank+1); all_docs[did]=doc
    return sorted(all_docs.values(), key=lambda d: scores[d['id']], reverse=True)

# These are harder test cases — deliberately tricky
TEST_CASES = [
    {
        "query"        : "printer shows offline even though it is connected",
        "category"     : "Technical",
        "relevant_terms": ["printer", "offline", "driver", "connection", "port"],
        "irrelevant_check": ["billing", "refund", "return", "invoice"],
    },
    {
        "query"        : "got charged for a service i cancelled last month",
        "category"     : "Billing and Payments",
        "relevant_terms": ["charge", "cancel", "refund", "billing", "subscription"],
        "irrelevant_check": ["network", "wifi", "router", "outage"],
    },
    {
        "query"        : "two factor authentication not sending sms code",
        "category"     : "Technical",
        "relevant_terms": ["authentication", "2fa", "sms", "code", "login", "verify"],
        "irrelevant_check": ["billing", "invoice", "payment"],
    },
    {
        "query"        : "product arrived damaged want replacement",
        "category"     : "Returns and Exchanges",
        "relevant_terms": ["damaged", "return", "replacement", "exchange", "product"],
        "irrelevant_check": ["billing", "technical", "network"],
    },
    {
        "query"        : "api rate limit exceeded on enterprise plan",
        "category"     : "Product Inquiry",
        "relevant_terms": ["api", "rate", "limit", "enterprise", "plan"],
        "irrelevant_check": ["billing", "return", "shipment"],
    },
]

def evaluate_results(docs, relevant_terms, irrelevant_check, top_k=5):
    """Returns multiple quality signals for a set of retrieved docs."""
    
    # 1. Category purity — all docs should be from same category
    categories = [d.get('category', '') for d in docs[:top_k]]
    purity = len(set(categories)) == 1  # True if all same category
    
    # 2. Keyword precision — fraction of top-K with at least one relevant term
    def has_relevant(doc):
        text = (doc.get('answer', '') + ' ' + doc.get('text', '')).lower()
        return any(t in text for t in relevant_terms)
    
    def has_irrelevant(doc):
        text = (doc.get('answer', '') + ' ' + doc.get('text', '')).lower()
        return any(t in text for t in irrelevant_check)
    
    relevant_hits = sum(1 for d in docs[:top_k] if has_relevant(d))
    irrelevant_hits = sum(1 for d in docs[:top_k] if has_irrelevant(d))
    precision = relevant_hits / min(top_k, len(docs)) if docs else 0
    contamination = irrelevant_hits / min(top_k, len(docs)) if docs else 0
    
    # 3. MRR — position of first relevant result
    mrr = 0.0
    for rank, doc in enumerate(docs[:top_k]):
        if has_relevant(doc):
            mrr = 1 / (rank + 1)
            break
    
    # 4. Answer diversity — are all top-K results different?
    texts = [doc.get('answer', doc.get('text', ''))[:100] for doc in docs[:top_k]]
    unique_texts = len(set(texts))
    diversity = unique_texts / len(texts) if texts else 0
    
    return {
        'precision_at_k' : round(precision, 3),
        'mrr'            : round(mrr, 3),
        'contamination'  : round(contamination, 3),  # lower is better
        'diversity'      : round(diversity, 3),
        'category_pure'  : purity,
        'n_docs'         : len(docs),
    }

def main():
    print("=" * 65)
    print("RETRIEVAL EVALUATION")
    print("=" * 65)
    
    all_dense_scores, all_sparse_scores, all_hybrid_scores = [], [], []
    
    for tc in TEST_CASES:
        query    = tc['query']
        category = tc['category']
        rel_terms = tc['relevant_terms']
        irrel     = tc['irrelevant_check']
        
        print(f"\nQuery: {query}")
        print(f"Category: {category}")
        
        dense   = dense_search(query, category, top_k=20)
        sparse  = sparse_search(query, category, top_k=20)
        fused   = _rrf(dense, sparse)
        final   = rerank(query, fused, top_k=5)
        
        d_scores = evaluate_results(dense,  rel_terms, irrel)
        s_scores = evaluate_results(sparse, rel_terms, irrel)
        h_scores = evaluate_results(final,  rel_terms, irrel)
        
        all_dense_scores.append(d_scores)
        all_sparse_scores.append(s_scores)
        all_hybrid_scores.append(h_scores)
        
        print(f"\n  {'Metric':<20} {'Dense':>8} {'Sparse':>8} {'Hybrid':>8}")
        print(f"  {'-'*44}")
        for metric in ['precision_at_k', 'mrr', 'contamination', 'diversity']:
            print(
                f"  {metric:<20} "
                f"{d_scores[metric]:>8.3f} "
                f"{s_scores[metric]:>8.3f} "
                f"{h_scores[metric]:>8.3f}"
            )
        print(f"  {'category_pure':<20} "
              f"{'Yes' if d_scores['category_pure'] else 'No':>8} "
              f"{'Yes' if s_scores['category_pure'] else 'No':>8} "
              f"{'Yes' if h_scores['category_pure'] else 'No':>8}")
        
        # Show what was actually retrieved
        print(f"\n  Top-3 hybrid results:")
        for i, doc in enumerate(final[:3]):
            text = (doc.get('answer', doc.get('text', '')))[:120].replace('\n',' ')
            print(f"    {i+1}. {text}...")
    
    # Summary
    def avg(scores, key):
        return sum(s[key] for s in scores) / len(scores)
    
    print(f"\n{'='*65}")
    print("AVERAGE ACROSS ALL QUERIES")
    print(f"{'='*65}")
    print(f"{'Metric':<22} {'Dense':>8} {'Sparse':>8} {'Hybrid':>8}")
    print(f"{'-'*50}")
    for metric in ['precision_at_k', 'mrr', 'contamination', 'diversity']:
        print(
            f"{metric:<22} "
            f"{avg(all_dense_scores, metric):>8.3f} "
            f"{avg(all_sparse_scores, metric):>8.3f} "
            f"{avg(all_hybrid_scores, metric):>8.3f}"
        )
    
    print(f"\nNote: contamination = fraction of top-K containing cross-category terms")
    print(f"      diversity     = fraction of top-K with unique answer text")
    print(f"      MRR           = Mean Reciprocal Rank (1.0 = first result is relevant)")

if __name__ == '__main__':
    main()