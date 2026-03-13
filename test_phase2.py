# test_phase2.py
from ingestion.loader import load_from_directory, summarize_loaded_files
from ingestion.chunker import chunk_all_files
from ingestion.embedder import run_full_embedding_pipeline
from retrieval.retriever import CodeRetriever

print("=" * 55)
print("🧠 PHASE 2 TEST — Embeddings + Vector Store + Retrieval")
print("=" * 55)

# ── Step 1: Load + Chunk (same as Phase 1) ──────────────
files = load_from_directory(".")
summarize_loaded_files(files)
chunks = chunk_all_files(files)

# ── Step 2: Embed + Save to FAISS ───────────────────────
print("\n📐 Embedding chunks into vectors...")
run_full_embedding_pipeline(chunks)

# ── Step 3: Load retriever + Test search ────────────────
print("\n🔎 Testing retrieval with sample queries...\n")
retriever = CodeRetriever()
retriever.load()

# Test with 3 different queries
test_queries = [
    "load files from a directory",
    "how are python functions chunked",
    "save index to disk",
]

for query in test_queries:
    retriever.search_and_display(query, top_k=2)
    print()

print("✅ Phase 2 complete! Retrieval is working.")