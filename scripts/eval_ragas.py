# scripts/eval_ragas.py

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from openai import AsyncOpenAI, OpenAI
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from datasets import Dataset
from src.retrieval.qdrant_store import dense_search
from src.retrieval.bm25_store   import sparse_search
from src.retrieval.reranker     import rerank
from src.agents.orchestrator    import run, get_final_draft

# RAGAS now requires explicit client instances for LLM and OpenAI embeddings.
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required to run scripts.eval_ragas")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

ragas_llm = llm_factory("gpt-4o-mini", client=openai_client)
ragas_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=openai_client,
)

faithfulness      = Faithfulness(llm=ragas_llm)
answer_relevancy  = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_precision = ContextPrecision(llm=ragas_llm, embeddings=ragas_embeddings)
context_recall    = ContextRecall(llm=ragas_llm, embeddings=ragas_embeddings)

def _rrf(dense, sparse, k=60):
    scores, all_docs = {}, {}
    for rank, doc in enumerate(dense):
        did = doc['id']; scores[did] = scores.get(did,0)+1/(k+rank+1); all_docs[did]=doc
    for rank, doc in enumerate(sparse):
        did = doc['id']; scores[did] = scores.get(did,0)+1/(k+rank+1); all_docs[did]=doc
    return sorted(all_docs.values(), key=lambda d: scores[d['id']], reverse=True)

TEST_SET = [
    {
        "question"    : "My WiFi keeps dropping after a firmware update. What should I do?",
        "category"    : "Technical",
        "ground_truth": (
            "We understand your WiFi is dropping after a firmware update. "
            "Please try restarting your router and checking if the firmware can be "
            "rolled back from the admin panel. If the issue persists, our technical "
            "team can schedule a call to investigate further."
        ),
    },
    {
        "question"    : "I was charged twice for my subscription this month.",
        "category"    : "Billing and Payments",
        "ground_truth": (
            "We apologise for the duplicate charge on your subscription. "
            "Please provide your account number and the dates of the duplicate "
            "transactions so we can investigate and process a refund within 24-48 hours."
        ),
    },
    {
        "question"    : "How do I return a damaged product?",
        "category"    : "Returns and Exchanges",
        "ground_truth": (
            "We're sorry to hear your product arrived damaged. "
            "You can initiate a return within 30 days by visiting our returns portal "
            "with your order number. Damaged items are eligible for a prepaid return "
            "label and refund within 5-7 business days."
        ),
    },
]

def main():
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
        q, category, gt = item['question'], item['category'], item['ground_truth']

        dense  = dense_search(q, category, top_k=20)
        sparse = sparse_search(q, category, top_k=20)
        fused  = _rrf(dense, sparse)
        final  = rerank(q, fused, top_k=5)

        result = run(q, use_cache=False)

        # Debug — see what's actually in messages
        msgs = result.get("messages", [])
        print(f"  Messages: {len(msgs)}, last type: {type(msgs[-1]).__name__ if msgs else 'none'}")
        if msgs:
            last_content = getattr(msgs[-1], 'content', '')
            print(f"  Last content type: {type(last_content).__name__}, preview: {str(last_content)[:80]}")

        answer = get_final_draft(result) or "No response generated."
        ctx    = [doc.get('answer', doc.get('text', ''))[:500] for doc in final]

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

        print(f"Processed: {q[:60]}")
        print(f"  Answer : {answer[:100]}...")

    dataset = Dataset.from_dict({
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "reference": ground_truths,
    })
    print(faithfulness)
    print(type(faithfulness))
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== RAGAS EVALUATION RESULTS ===")
    print(scores)
    df = scores.to_pandas()
    cols = ['question', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    print(df[[c for c in cols if c in df.columns]].to_string())

if __name__ == '__main__':
    main()
