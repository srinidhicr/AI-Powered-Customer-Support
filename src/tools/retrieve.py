# src/tools/retrieve.py
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from configs.config import config

_llm = ChatOpenAI(
    model=config.llm_model,
    api_key=config.openai_api_key,
    max_tokens=100,
    temperature=0
)

def _llm_invoke_with_retry(prompt: str) -> str:
    for attempt in range(3):
        try:
            return _llm.invoke(prompt).content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"[Rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

def _rewrite_query(query: str, category: str) -> str:
    prompt = f"""Rewrite this customer support query to improve document retrieval.
Be specific and include domain keywords relevant to the category.

Category: {category}
Original query: {query}
Rewritten query (one line only, no explanation):"""
    return _llm_invoke_with_retry(prompt)

def _rrf(dense_results: list, sparse_results: list, k: int = 60) -> list:
    scores = {}
    for rank, doc in enumerate(dense_results):
        did = doc['id']
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        did = doc['id']
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
    all_docs = {d['id']: d for d in dense_results + sparse_results}
    return sorted(all_docs.values(), key=lambda d: scores[d['id']], reverse=True)


# ── Pydantic input schema ─────────────────────────────────────────────────────

class RetrieveInput(BaseModel):
    query: str = Field(description="The customer query text.")
    category: str = Field(description="The predicted category from classify() tool.")
    confidence: float = Field(
        default=0.0,
        description="Classifier confidence score from classify() tool. Pass this through."
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieveInput)
def retrieve(query: str, category: str, confidence: float = 0.0) -> str:
    """Retrieves relevant knowledge base documents for a customer query.
    Uses hybrid search: dense (Qdrant) + sparse (BM25) with RRF fusion and reranking.
    ALWAYS call classify() first to get the category and confidence before calling this tool.
    Pass the confidence score from classify() so retrieval can skip query rewriting when not needed.
    Returns a JSON string with keys: query_rewritten, documents (list), n_retrieved, best_rerank_score.
    """
    from src.retrieval.qdrant_store import dense_search
    from src.retrieval.bm25_store   import sparse_search
    from src.retrieval.reranker     import rerank

    # ── Change 2a: Skip query rewrite when classifier is confident ────────────
    skip_rewrite = confidence >= config.skip_rewrite_confidence
    if skip_rewrite:
        rewritten = query
        print(f"[Retrieve] Skipping query rewrite (confidence={confidence:.2f} >= {config.skip_rewrite_confidence})")
    else:
        rewritten = _rewrite_query(query, category)
        print(f"[Retrieve] Query rewritten (confidence={confidence:.2f})")

    # ── Change 2b: Parallel dense + sparse search ─────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dense  = executor.submit(dense_search,  rewritten, category, config.top_k_dense)
        future_sparse = executor.submit(sparse_search, rewritten, category, config.top_k_sparse)
        dense  = future_dense.result()
        sparse = future_sparse.result()

    fused    = _rrf(dense, sparse)
    reranked = rerank(query, fused, top_k=config.top_k_final)

    best_rerank_score = max(
        (float(d.get("rerank_score", 0)) for d in reranked),
        default=0.0
    )

    return json.dumps({
        "query_rewritten"  : rewritten,
        "documents"        : reranked,
        "n_retrieved"      : len(reranked),
        "best_rerank_score": round(best_rerank_score, 4),
    })