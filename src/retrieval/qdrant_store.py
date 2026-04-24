# src/retrieval/qdrant_store.py

import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchValue
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
_encoder = SentenceTransformer(
    _resolve_local_snapshot(LOCAL_MODEL_ROOT) or 'all-MiniLM-L6-v2'
)   # 384-dim, fast, good

COLLECTION = config.collection_name
DIM        = 384


def build_index(chunks_path: str = 'data/knowledge_base/kb_chunks.jsonl'):
    """Run once to create collection and upload all chunks."""
    # Re-create collection
    if _client.collection_exists(COLLECTION):
        _client.delete_collection(COLLECTION)
    _client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )

    chunks = []
    with open(chunks_path) as f:
        for line in f:
            chunks.append(json.loads(line))

    # Batch embed
    texts   = [c['text'] for c in chunks]
    vectors = _encoder.encode(texts, batch_size=64, show_progress_bar=True)

    points = [
        PointStruct(
            id      = i,
            vector  = vectors[i].tolist(),
            payload = chunks[i]          # store full chunk as payload
        )
        for i in range(len(chunks))
    ]

    # Upload in batches of 500
    for start in range(0, len(points), 500):
        _client.upsert(
            collection_name=COLLECTION,
            points=points[start:start+500]
        )
    print(f"Indexed {len(points)} chunks into Qdrant.")


def dense_search(query: str, category: str, top_k: int = 20) -> list:
    vec = _encoder.encode([query], normalize_embeddings=True)[0].tolist()

    results = _client.query_points(
        collection_name=COLLECTION,
        query=vec,
        query_filter=Filter(
                must=[FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )]
            ),
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    points = results.points or []

    print("\n[QDRANT RESULTS]")
    for i, hit in enumerate(points[:5]):
        text_preview = hit.payload.get("text", "")[:120].replace("\n", " ")
        print(f"{i+1}. score={hit.score:.4f} | {text_preview}...")

    return [
        {**hit.payload, "score": hit.score}
        for hit in points
    ]


if __name__ == '__main__':
    build_index()
