# ingestion/pipeline.py
"""
Master pipeline: load → chunk → embed → ready
Called by the UI when user submits a repo/folder.
"""
from ingestion.loader import load_from_directory, load_from_github
from ingestion.chunker import chunk_all_files
from ingestion.embedder import run_full_embedding_pipeline


def run_ingestion_pipeline(source: str, is_github: bool = False) -> dict:
    """
    Full pipeline from source → vector store.
    Returns stats dict for the UI to display.
    """
    from collections import Counter

    print(f"\n🚀 Starting ingestion pipeline...")
    print(f"   Source : {source}")
    print(f"   Type   : {'GitHub' if is_github else 'Local'}")

    # Step 1: Load files
    if is_github:
        files = load_from_github(source)
    else:
        files = load_from_directory(source)

    if not files:
        return {"error": "No supported code files found in this source."}

    # Step 2: Chunk
    chunks = chunk_all_files(files)

    if not chunks:
        return {"error": "Could not extract any functions or classes."}

    # Step 3: Embed + save to FAISS
    run_full_embedding_pipeline(chunks)

    # Step 4: Build stats for UI
    lang_counts = Counter(f["language"] for f in files)
    chunk_types = Counter(c["chunk_type"] for c in chunks)

    stats = {
        "total_files": len(files),
        "total_chunks": len(chunks),
        "languages": dict(lang_counts),
        "chunk_types": dict(chunk_types),
        "source": source,
        "is_github": is_github
    }

    print(f"\n✅ Pipeline complete!")
    print(f"   Files  : {stats['total_files']}")
    print(f"   Chunks : {stats['total_chunks']}")

    return stats