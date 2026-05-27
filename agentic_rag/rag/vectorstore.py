from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import faiss

    class VectorStore:
        """FAISS-backed vector store for fast similarity search."""

        def __init__(self, dim: int):
            self.index = faiss.IndexFlatL2(dim)
            self.docs: List[str] = []
            self.metas: List[Dict[str, str]] = []
            self.embeddings: np.ndarray = np.empty((0, dim), dtype="float32")

        def add(
            self,
            texts: List[str],
            embeddings: np.ndarray,
            metadatas: Optional[List[Dict[str, str]]] = None,
        ) -> None:
            assert embeddings.shape[0] == len(texts), "Embedding count must match document count"
            if metadatas is None:
                metadatas = [{} for _ in texts]
            if len(metadatas) != len(texts):
                raise ValueError("Metadata count must match document count")
            self.index.add(embeddings)
            self.embeddings = (
                np.vstack([self.embeddings, embeddings])
                if self.embeddings.size
                else embeddings.copy()
            )
            self.docs.extend(texts)
            self.metas.extend(metadatas)

        def search(
            self,
            query_emb: np.ndarray,
            k: int,
            *,
            doc_id: Optional[str] = None,
        ) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[List[Dict[str, str]]]]:
            if not self.docs:
                return np.array([[]]), np.array([[-1]]), [[]], [[]]

            if doc_id:
                return self._search_scoped(query_emb, k, doc_id)

            k = min(k, len(self.docs))
            distances, idx = self.index.search(query_emb, k)
            retrieved = [[self.docs[i] for i in row if 0 <= i < len(self.docs)] for row in idx]
            metas = [[self.metas[i] for i in row if 0 <= i < len(self.metas)] for row in idx]
            return distances, idx, retrieved, metas

        def _search_scoped(
            self,
            query_emb: np.ndarray,
            k: int,
            doc_id: str,
        ) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[List[Dict[str, str]]]]:
            indices = [i for i, m in enumerate(self.metas) if m.get("doc_id") == doc_id]
            if not indices:
                return np.array([[]]), np.array([[-1]]), [[]], [[]]
            scoped_embs = self.embeddings[indices]
            diff = scoped_embs - query_emb
            dists = np.sum(diff ** 2, axis=1)
            order = np.argsort(dists)[: min(k, len(indices))]
            distances = dists[order].reshape(1, -1)
            idx = np.array([[indices[i] for i in order]], dtype=int)
            retrieved = [[self.docs[i] for i in idx[0]]]
            metas = [[self.metas[i] for i in idx[0]]]
            return distances, idx, retrieved, metas

        def has_doc_id(self, doc_id: str) -> bool:
            return any(m.get("doc_id") == doc_id for m in self.metas)

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
            self.metas: List[Dict[str, str]] = []

        def add(
            self,
            texts: List[str],
            embeddings: np.ndarray,
            metadatas: Optional[List[Dict[str, str]]] = None,
        ) -> None:
            assert embeddings.shape[0] == len(texts), "Embedding count must match document count"
            if metadatas is None:
                metadatas = [{} for _ in texts]
            if len(metadatas) != len(texts):
                raise ValueError("Metadata count must match document count")
            self.embeddings = (
                np.vstack([self.embeddings, embeddings])
                if self.embeddings.size
                else embeddings.copy()
            )
            self.docs.extend(texts)
            self.metas.extend(metadatas)

        def search(
            self,
            query_emb: np.ndarray,
            k: int,
            *,
            doc_id: Optional[str] = None,
        ) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[List[Dict[str, str]]]]:
            if not self.docs:
                return np.array([[]]), np.array([[-1]]), [[]], [[]]

            if doc_id:
                return self._search_scoped(query_emb, k, doc_id)

            k = min(k, len(self.docs))
            diff = self.embeddings - query_emb
            dists = np.sum(diff ** 2, axis=1)
            idx = np.argsort(dists)[:k]
            distances = dists[idx].reshape(1, -1)
            idx_2d = idx.reshape(1, -1)
            retrieved = [[self.docs[i] for i in idx]]
            metas = [[self.metas[i] for i in idx]]
            return distances, idx_2d, retrieved, metas

        def _search_scoped(
            self,
            query_emb: np.ndarray,
            k: int,
            doc_id: str,
        ) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[List[Dict[str, str]]]]:
            indices = [i for i, m in enumerate(self.metas) if m.get("doc_id") == doc_id]
            if not indices:
                return np.array([[]]), np.array([[-1]]), [[]], [[]]
            scoped_embs = self.embeddings[indices]
            diff = scoped_embs - query_emb
            dists = np.sum(diff ** 2, axis=1)
            order = np.argsort(dists)[: min(k, len(indices))]
            distances = dists[order].reshape(1, -1)
            idx = np.array([[indices[i] for i in order]], dtype=int)
            retrieved = [[self.docs[i] for i in idx[0]]]
            metas = [[self.metas[i] for i in idx[0]]]
            return distances, idx, retrieved, metas

        def has_doc_id(self, doc_id: str) -> bool:
            return any(m.get("doc_id") == doc_id for m in self.metas)

        @property
        def size(self) -> int:
            return len(self.docs)
