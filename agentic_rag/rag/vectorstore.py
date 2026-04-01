from typing import List, Tuple
import numpy as np

try:
    import faiss

    class VectorStore:
        """FAISS-backed vector store for fast similarity search."""

        def __init__(self, dim: int):
            self.index = faiss.IndexFlatL2(dim)
            self.docs: List[str] = []

        def add(self, texts: List[str], embeddings: np.ndarray) -> None:
            assert embeddings.shape[0] == len(texts), "Embedding count must match document count"
            self.index.add(embeddings)
            self.docs.extend(texts)

        def search(self, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
            k = min(k, len(self.docs)) if self.docs else 1
            distances, idx = self.index.search(query_emb, k)
            retrieved = [[self.docs[i] for i in row if 0 <= i < len(self.docs)] for row in idx]
            return distances, idx, retrieved

        @property
        def size(self) -> int:
            return len(self.docs)

except ImportError:
    # NumPy brute-force fallback when FAISS is not available
    class VectorStore:  # type: ignore[no-redef]
        """Pure-NumPy fallback vector store (no FAISS required)."""

        def __init__(self, dim: int):
            self.dim = dim
            self.embeddings: np.ndarray = np.empty((0, dim), dtype="float32")
            self.docs: List[str] = []

        def add(self, texts: List[str], embeddings: np.ndarray) -> None:
            assert embeddings.shape[0] == len(texts), "Embedding count must match document count"
            self.embeddings = (
                np.vstack([self.embeddings, embeddings])
                if self.embeddings.size
                else embeddings.copy()
            )
            self.docs.extend(texts)

        def search(self, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
            if not self.docs:
                return np.array([[]]), np.array([[-1]]), [[]]
            k = min(k, len(self.docs))
            # L2 distance
            diff = self.embeddings - query_emb
            dists = np.sum(diff ** 2, axis=1)
            idx = np.argsort(dists)[:k]
            distances = dists[idx].reshape(1, -1)
            idx_2d = idx.reshape(1, -1)
            retrieved = [[self.docs[i] for i in idx]]
            return distances, idx_2d, retrieved

        @property
        def size(self) -> int:
            return len(self.docs)
