# scripts/build_kb_script.py
"""
This script is to build the complete knowledge base.
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def merge_chunks() -> str:
    """Merge original KB chunks with generated resolution chunks.
    Returns path to the merged file."""
    original   = 'data/knowledge_base/kb_chunks.jsonl'
    resolutions = 'data/knowledge_base/resolution_chunks.jsonl'
    merged     = 'data/knowledge_base/kb_chunks_merged.jsonl'

    chunks = []
    with open(original) as f:
        for line in f:
            chunks.append(json.loads(line))
    original_count = len(chunks)

    if os.path.exists(resolutions):
        with open(resolutions) as f:
            for line in f:
                chunks.append(json.loads(line))
        print(f"  Original chunks  : {original_count}")
        print(f"  Resolution chunks: {len(chunks) - original_count}")
        print(f"  Total merged     : {len(chunks)}")
    else:
        print("  No resolution_chunks.jsonl found — using original only")

    with open(merged, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

    return merged


def main():
    print("=" * 55)
    print("  KB BUILD PIPELINE")
    print("=" * 55)

    print("\n[1/4] Preprocessing dataset → kb_chunks.jsonl ...")
    from scripts.prepare_kb_database import prepare
    prepare()

    print("\n[2/4] Merging with resolution chunks (if available) ...")
    merged_path = merge_chunks()

    print("\n[3/4] Indexing into Qdrant (dense vectors) ...")
    from src.retrieval.qdrant_store import build_index
    build_index(chunks_path=merged_path)          # ← pass merged path here

    print("\n[4/4] Building BM25 sparse index ...")
    from src.retrieval.bm25_store import _load_or_build, CHUNKS_PATH
    # BM25 also needs to read the merged file
    import src.retrieval.bm25_store as bm25_module
    bm25_module.CHUNKS_PATH = merged_path         # ← point BM25 at merged file
    _load_or_build()

    print("\n" + "=" * 55)
    print(f"  Done. Indexed from: {merged_path}")
    print("  Run: python -m scripts.test_pipeline")
    print("=" * 55)


if __name__ == '__main__':
    main()