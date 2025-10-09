from typing import List
from .embeddings import Embeddings
from .vectorstore import VectorStore

class Retriever:
    def __init__(self, embedder: Embeddings, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int) -> List[str]:
        q = self.embedder.encode([query])
        _, _, docs = self.store.search(q, top_k)
        return docs[0] if docs else []
