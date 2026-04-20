# src/retrieval/bm25_store.py

import os, sys, json, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from rank_bm25 import BM25Okapi

INDEX_PATH  = 'data/knowledge_base/bm25_index.pkl'
CHUNKS_PATH = 'data/knowledge_base/kb_chunks.jsonl'

_index = None  # lazy — not loaded at import time

def _load_or_build():
    global _index
    if _index is not None:
        return _index

    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'rb') as f:
            _index = pickle.load(f)
        return _index

    # Build from JSONL
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(json.loads(line))

    corpus = [c['text'].lower().split() for c in chunks]
    bm25   = BM25Okapi(corpus)
    _index = {"bm25": bm25, "chunks": chunks}

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(_index, f)
    return _index


def sparse_search(query: str, category: str, top_k: int = 20) -> list:
    idx    = _load_or_build()          # load on first actual call
    tokens = query.lower().split()
    scores = idx["bm25"].get_scores(tokens)
    chunks = idx["chunks"]

    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked:
        if chunks[i].get("category") == category:
            results.append({**chunks[i], "score": float(scores[i])})
        if len(results) == top_k:
            break
    return results


if __name__ == '__main__':
    _load_or_build()