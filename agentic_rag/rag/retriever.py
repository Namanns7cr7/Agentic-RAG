from typing import Any, Dict, List, Optional
from .embeddings import Embeddings
from .vectorstore import VectorStore

class Retriever:
    def __init__(self, embedder: Embeddings, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = self.embedder.encode([query])
        distances, idx, docs, metas = self.store.search(q, top_k, doc_id=doc_id)
        if not docs or not docs[0]:
            return []
        results: List[Dict[str, Any]] = []
        for pos, text in enumerate(docs[0]):
            meta = metas[0][pos] if metas and metas[0] else {}
            dist = float(distances[0][pos]) if distances.size else 0.0
            score = 1.0 / (1.0 + dist)
            results.append(
                {
                    "text": text,
                    "doc_id": meta.get("doc_id", ""),
                    "score": score,
                    "index": int(idx[0][pos]) if idx.size else -1,
                }
            )
        return results

    def has_doc_id(self, doc_id: str) -> bool:
        return self.store.has_doc_id(doc_id)
