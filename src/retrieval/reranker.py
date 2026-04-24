# src/retrieval/reranker.py

import os
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOCAL_MODEL_ROOT = os.path.join(
    BASE_DIR,
    "transformer-models",
    "models--cross-encoder--ms-marco-MiniLM-L-6-v2",
)


def _resolve_local_snapshot(model_root: str) -> str | None:
    refs_main = os.path.join(model_root, "refs", "main")
    if not os.path.exists(refs_main):
        return None
    with open(refs_main) as f:
        snapshot = f.read().strip()
    snapshot_path = os.path.join(model_root, "snapshots", snapshot)
    return snapshot_path if os.path.exists(snapshot_path) else None

_cross = CrossEncoder(_resolve_local_snapshot(LOCAL_MODEL_ROOT) or 'cross-encoder/ms-marco-MiniLM-L-6-v2')

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
    return [
        {**doc, "rerank_score": float(score)}
        for doc, score in ranked[:top_k]
    ]
