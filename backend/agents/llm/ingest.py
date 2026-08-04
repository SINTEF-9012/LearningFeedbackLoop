from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
import os
import numpy as np

# Delay optional imports until actually used to avoid import-time errors
SentenceTransformer: Any = None
faiss: Any = None

def _optional_imports():
    global SentenceTransformer, faiss
    if SentenceTransformer is None or faiss is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            import faiss as _faiss
        except Exception as e:
            raise RuntimeError("Please install optional dependencies: sentence-transformers, faiss-cpu") from e
        SentenceTransformer = _ST
        faiss = _faiss


class Ingestor:
    """Ingest documents and build a FAISS index using sentence-transformers.

    Documents are dicts: {"id": str, "text": str, "meta": {...}}
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None):
        _optional_imports()
        self._model = SentenceTransformer(model_name)
        self._index = None
        self._docs: List[Dict] = []
        self._index_path = index_path

    def _build_index(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(embeddings)

    def ingest(self, docs: List[Dict[str, Any]], persist: bool = False):
        texts = [d["text"] for d in docs]
        embs = self._model.encode(texts, convert_to_numpy=True)
        if self._index is None:
            self._build_index(embs)
        else:
            self._index.add(embs)
        start = len(self._docs)
        for i, d in enumerate(docs):
            self._docs.append(d)

        if persist and self._index_path:
            self.save(self._index_path)

    def save(self, path: str):
        if self._index is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self._index, path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._index = faiss.read_index(path)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if self._index is None:
            return []
        q_emb = self._model.encode([text], convert_to_numpy=True)
        D, I = self._index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            results.append((self._docs[idx], float(score)))
        return results
