import hashlib
from typing import Dict, List, Optional

import numpy as np

from llm_client import LLMClient


class Embedder:
    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        batch_size: int = 128,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self._client = LLMClient(api_key=api_key).client
        self._cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        out: List[Optional[np.ndarray]] = [None] * len(texts)
        miss_idx: List[int] = []
        miss_texts: List[str] = []
        miss_keys: List[str] = []

        for i, t in enumerate(texts):
            k = self._key(t)
            cached = self._cache.get(k)
            if cached is not None:
                out[i] = cached
            else:
                miss_idx.append(i)
                miss_texts.append(t)
                miss_keys.append(k)

        for start in range(0, len(miss_texts), self.batch_size):
            batch = miss_texts[start : start + self.batch_size]
            batch_keys = miss_keys[start : start + self.batch_size]
            batch_idx = miss_idx[start : start + self.batch_size]

            resp = self._client.embeddings.create(model=self.model, input=batch)
            for j, item in enumerate(resp.data):
                vec = np.asarray(item.embedding, dtype=np.float32)
                self._cache[batch_keys[j]] = vec
                out[batch_idx[j]] = vec

        return [v for v in out if v is not None]

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
