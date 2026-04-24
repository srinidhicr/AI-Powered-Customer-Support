# src/retrieval/semantic_cache.py

import os, sys, json, hashlib
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, ScoredPoint
)
from sentence_transformers import SentenceTransformer
from configs.config import config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOCAL_MODEL_ROOT = os.path.join(
    BASE_DIR,
    "transformer-models",
    "models--sentence-transformers--all-MiniLM-L6-v2",
)


def _resolve_local_snapshot(model_root: str) -> str | None:
    refs_main = os.path.join(model_root, "refs", "main")
    if not os.path.exists(refs_main):
        return None
    with open(refs_main) as f:
        snapshot = f.read().strip()
    snapshot_path = os.path.join(model_root, "snapshots", snapshot)
    return snapshot_path if os.path.exists(snapshot_path) else None

_client  = QdrantClient(host=config.qdrant_url, port=config.qdrant_port)
_encoder = SentenceTransformer(_resolve_local_snapshot(LOCAL_MODEL_ROOT) or 'all-MiniLM-L6-v2')

CACHE_COLLECTION = "response_cache"
DIM              = 384
SIMILARITY_THRESHOLD = 0.80   # tune this — lower = more cache hits, less precision


def _ensure_collection():
    if not _client.collection_exists(CACHE_COLLECTION):
        _client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
        )


def lookup(query: str) -> dict | None:
    """Check if a semantically similar query has been cached.
    Returns cached result dict or None if no hit."""
    _ensure_collection()
    vec = _encoder.encode([query], normalize_embeddings=True)[0].tolist()
    response = _client.query_points(
        collection_name=CACHE_COLLECTION,
        query=vec,
        limit=1,
        with_payload=True,
        with_vectors=False,
        score_threshold=SIMILARITY_THRESHOLD
    )

    results = response.points
    if results and len(results) > 0:
        hit = results[0]
        orig_query = hit.payload.get("query", "")
        print(f"[Cache HIT] similarity={hit.score:.3f} | original: '{orig_query[:60]}...'")
        cached = hit.payload.get("cached_result", {})

        # 🔥 extra safety
        if not isinstance(cached.get("final_draft"), str):
            cached["final_draft"] = str(cached.get("final_draft", ""))

        return cached
    print("[Cache MISS]")
    return None


def store(query: str, result: dict):
    """Store a query + its result in the cache."""
    _ensure_collection()
    vec = _encoder.encode([query], normalize_embeddings=True)[0].tolist()

    # Use a hash of the query as a stable integer ID
    uid = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    safe_result = {
        "final_draft": str(result.get("final_draft", "")),
        "query": query,
        "chunks": result.get("chunks", []),
    }
    _client.upsert(
    collection_name=CACHE_COLLECTION,
    points=[PointStruct(
        id=uid,
        vector=vec,
        payload={
            "query": query,
            "cached_result": safe_result
        }
        )]
    )
    print(f"[Cache STORE] '{query[:60]}...'")


def clear():
    """Wipe the cache — useful during testing."""
    if _client.collection_exists(CACHE_COLLECTION):
        _client.delete_collection(CACHE_COLLECTION)
    print("Cache cleared.")
