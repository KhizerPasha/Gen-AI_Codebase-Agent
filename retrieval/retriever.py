# retrieval/retriever.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.embedder import load_vector_store, load_embedding_model, MODEL_NAME


class CodeRetriever:
    """
    Handles semantic search over the embedded codebase.
    Give it a natural language query → get back relevant code chunks.
    """

    def __init__(self):
        self.index = None
        self.metadata = None
        self.model = None

    def load(self):
        """Load the vector store and embedding model into memory."""
        self.index, self.metadata = load_vector_store()
        self.model = load_embedding_model()
        print("✅ CodeRetriever ready!")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for code chunks most relevant to the query.

        Args:
            query: Natural language question or keyword
            top_k: Number of results to return

        Returns:
            List of chunk dicts with similarity scores
        """
        if not self.index:
            raise RuntimeError("Call .load() before searching!")

        # Embed the query
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)

        # Search FAISS
        scores, indices = self.index.search(query_vector, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue

            chunk = self.metadata[idx].copy()
            chunk["similarity_score"] = round(float(score), 4)
            results.append(chunk)

        return results

    def search_and_display(self, query: str, top_k: int = 3) -> list[dict]:
        """Search and pretty-print results."""
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)

        results = self.search(query, top_k)

        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] {result['name']} ({result['chunk_type']})")
            print(f"      File  : {result['file_path']}")
            print(f"      Lines : {result['start_line']} → {result['end_line']}")
            print(f"      Score : {result['similarity_score']}")
            print(f"      Code  :\n")
            # Show first 200 chars of code
            preview = result['code'][:200].strip()
            for line in preview.split("\n"):
                print(f"        {line}")
            print(f"        ...")

        return results