# src/retrieval/reranker.py

from sentence_transformers import CrossEncoder
import os

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#MODEL_PATH = os.path.join(BASE_DIR, "transformer-models", "cross-encoder-ms-marco")


_cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#_cross = CrossEncoder(MODEL_PATH)

def rerank(query: str, documents: list, top_k: int = 5) -> list:
    if not documents:
        return []
    pairs  = [(query, d['text']) for d in documents]
    scores = _cross.predict(pairs)
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, _ in ranked[:top_k]]