# ingestion/embedder.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Best model for code understanding
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Where to save the index
VECTOR_STORE_PATH = "data/vectorstore"
INDEX_FILE = os.path.join(VECTOR_STORE_PATH, "index.faiss")
METADATA_FILE = os.path.join(VECTOR_STORE_PATH, "metadata.json")


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer model."""
    print(f"🤖 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Model loaded!")
    return model


def build_text_for_embedding(chunk: dict) -> str:
    """
    Build a rich text string from a chunk for better embedding quality.
    We include the name + file path + code so the vector captures context.
    """
    return f"""
File: {chunk['file_path']}
Type: {chunk['chunk_type']}
Name: {chunk['name']}
Language: {chunk['language']}
Code:
{chunk['code']}
""".strip()


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Convert all chunks into embedding vectors.
    Returns a numpy array of shape (num_chunks, embedding_dim)
    """
    print(f"⚙️  Embedding {len(chunks)} chunks...")

    texts = [build_text_for_embedding(chunk) for chunk in chunks]

    # Batch embed (much faster than one by one)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"✅ Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from embeddings.
    Uses cosine similarity (IndexFlatIP with normalized vectors).
    """
    dim = embeddings.shape[1]
    print(f"🏗️  Building FAISS index (dim={dim})...")

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # IndexFlatIP = exact search with inner product (cosine after normalization)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"✅ FAISS index built with {index.ntotal} vectors")
    return index


def save_vector_store(index: faiss.Index, chunks: list[dict]) -> None:
    """Save FAISS index + chunk metadata to disk."""
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)
    print(f"💾 FAISS index saved → {INDEX_FILE}")

    # Save metadata (everything except the code preview for size)
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "id": i,
            "chunk_id": chunk["chunk_id"],
            "file_path": chunk["file_path"],
            "language": chunk["language"],
            "chunk_type": chunk["chunk_type"],
            "name": chunk["name"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "code": chunk["code"],        # full code stored here
            "char_count": chunk["char_count"]
        })

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Metadata saved → {METADATA_FILE}")
    print(f"✅ Vector store complete! {len(metadata)} chunks stored.")


def load_vector_store() -> tuple[faiss.Index, list[dict]]:
    """Load FAISS index + metadata from disk."""
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"No index found at {INDEX_FILE}. Run embedder first!")

    print("📂 Loading vector store from disk...")
    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"✅ Loaded {index.ntotal} vectors + {len(metadata)} metadata entries")
    return index, metadata


def run_full_embedding_pipeline(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    Master function: takes chunks → returns saved index + metadata.
    Call this from your main pipeline.
    """
    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    save_vector_store(index, chunks)
    return index, chunks