from typing import List, Tuple
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.docs: List[str] = []

    def add(self, texts: List[str], embeddings: np.ndarray) -> None:
        assert embeddings.shape[0] == len(texts)
        self.index.add(embeddings)
        self.docs.extend(texts)

    def search(self, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        distances, idx = self.index.search(query_emb, k)
        retrieved = [[self.docs[i] for i in row if i < len(self.docs)] for row in idx]
        return distances, idx, retrieved
