from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        arr = self.model.encode(texts, convert_to_numpy=True).astype('float32')
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
