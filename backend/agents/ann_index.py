"""ANN (Approximate Nearest Neighbor) index manager using FAISS.

Provides vector indexing and search for:
1. Numeric feature vectors (for similarity-based memory retrieval)
2. Text embeddings (for semantic search on annotations)

Uses lazy imports to avoid import-time failures if faiss is not installed.
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np


class ANNIndex:
    """FAISS-based approximate nearest neighbor index.
    
    Supports:
    - Building index from vectors
    - Incremental insertion
    - Top-K search
    - Persistence (save/load)
    
    Maintains a separate ID mapping since FAISS uses integer indices internally.
    """
    
    # Default dimension for numeric metrics (can be overridden)
    DEFAULT_DIMENSION = 64
    
    def __init__(self, dimension: Optional[int] = None, index_type: str = "flat"):
        """Initialize the ANN index.
        
        Args:
            dimension: Vector dimension (must be consistent across all vectors).
                       If None, will be set on first vector insertion.
            index_type: Type of FAISS index:
                - "flat": Exact search (IndexFlatL2) - good for <10K vectors
                - "ivf": Approximate search (IndexIVFFlat) - good for >10K vectors
                - "hnsw": Graph-based search - good balance of speed/accuracy
        """
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._id_to_idx: Dict[str, int] = {}  # memory_id -> faiss index
        self._idx_to_id: Dict[int, str] = {}  # faiss index -> memory_id
        self._next_idx = 0
        self._faiss = None  # Lazy import
    
    def _ensure_faiss(self):
        """Lazy import faiss to avoid import-time errors."""
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
            except ImportError as e:
                raise RuntimeError(
                    "FAISS is required for ANN index. "
                    "Install with: pip install faiss-cpu"
                ) from e
    
    def _create_index(self):
        """Create the FAISS index based on index_type."""
        if self.dimension is None:
            raise ValueError("Cannot create index without dimension. Call build() or insert() first with vectors.")
        
        self._ensure_faiss()
        faiss = self._faiss
        
        if self.index_type == "flat":
            # Exact L2 search - simple and accurate
            self._index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            # IVF index - faster for large datasets
            # Requires training before use
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # Number of clusters
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "hnsw":
            # HNSW - good balance of speed and accuracy
            M = 32  # Number of connections per layer
            self._index = faiss.IndexHNSWFlat(self.dimension, M)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")
    
    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Build index from scratch with given vectors and IDs.
        
        Args:
            vectors: (N, dimension) float32 array of vectors
            ids: List of N memory IDs corresponding to vectors
        """
        if len(vectors) != len(ids):
            raise ValueError(f"vectors ({len(vectors)}) and ids ({len(ids)}) must have same length")
        
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if len(vectors) == 0:
            if self.dimension is None:
                self.dimension = self.DEFAULT_DIMENSION
            self._create_index()
            return
        
        # Set dimension from first vectors if not already set
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        
        self._create_index()
        
        # For IVF index, we need to train first
        if self.index_type == "ivf" and not self._index.is_trained:
            self._index.train(vectors)
        
        # Add vectors
        self._index.add(vectors)
        
        # Build ID mappings
        self._id_to_idx = {}
        self._idx_to_id = {}
        for i, mem_id in enumerate(ids):
            self._id_to_idx[mem_id] = i
            self._idx_to_id[i] = mem_id
        self._next_idx = len(ids)
    
    def insert(self, vector: np.ndarray, memory_id: str) -> None:
        """Insert a single vector into the index.
        
        Args:
            vector: (dimension,) float32 vector
            memory_id: Unique identifier for this vector
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        # Set dimension from first vector if not already set
        if self.dimension is None:
            self.dimension = vector.shape[1]
        
        if self._index is None:
            self._create_index()
        
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vector.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        
        # For IVF, need at least some vectors to train
        if self.index_type == "ivf" and not self._index.is_trained:
            # Can't insert without training; caller should use build() first
            raise RuntimeError(
                "IVF index requires training before insertion. "
                "Use build() with initial vectors first."
            )
        
        # Check if ID already exists
        if memory_id in self._id_to_idx:
            # Update: FAISS doesn't support in-place updates easily
            # For now, we just add a new entry (old one becomes orphaned)
            # A full rebuild would be needed to truly update
            pass
        
        self._index.add(vector)
        idx = self._next_idx
        self._id_to_idx[memory_id] = idx
        self._idx_to_id[idx] = memory_id
        self._next_idx += 1
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors.
        
        Args:
            query_vector: (dimension,) query vector
            k: Number of neighbors to return
        
        Returns:
            List of (memory_id, distance) tuples, sorted by distance ascending
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        
        # Limit k to available vectors
        k = min(k, self._index.ntotal)
        if k == 0:
            return []
        
        # Search
        distances, indices = self._index.search(query_vector, k)
        
        # Convert to (memory_id, distance) pairs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx in self._idx_to_id:
                mem_id = self._idx_to_id[idx]
                results.append((mem_id, float(dist)))
        
        return results
    
    def remove(self, memory_id: str) -> bool:
        """Mark a memory ID as removed.
        
        Note: FAISS doesn't support true deletion easily. This marks the ID
        as removed in our mapping, but the vector remains in the index.
        A periodic rebuild is recommended to reclaim space.
        
        Args:
            memory_id: ID to remove
        
        Returns:
            True if ID was found and removed, False otherwise
        """
        if memory_id not in self._id_to_idx:
            return False
        
        idx = self._id_to_idx[memory_id]
        del self._id_to_idx[memory_id]
        if idx in self._idx_to_id:
            del self._idx_to_id[idx]
        return True
    
    def save(self, path: str) -> None:
        """Save index and ID mappings to disk.
        
        Creates two files:
        - {path}.index: FAISS index binary
        - {path}.meta: ID mappings (JSON)
        
        Args:
            path: Base path for saving (without extension)
        """
        if self._index is None:
            return
        
        self._ensure_faiss()
        
        # Save FAISS index
        index_path = f"{path}.index"
        self._faiss.write_index(self._index, index_path)
        
        # Save metadata
        meta_path = f"{path}.meta"
        meta = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "next_idx": self._next_idx,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    
    def load(self, path: str) -> bool:
        """Load index and ID mappings from disk.
        
        Args:
            path: Base path (same as used in save())
        
        Returns:
            True if loaded successfully, False if files don't exist
        """
        index_path = f"{path}.index"
        meta_path = f"{path}.meta"
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False
        
        self._ensure_faiss()
        
        # Load FAISS index
        self._index = self._faiss.read_index(index_path)
        
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        self.dimension = meta["dimension"]
        self.index_type = meta["index_type"]
        self._id_to_idx = meta["id_to_idx"]
        self._idx_to_id = {int(k): v for k, v in meta["idx_to_id"].items()}
        self._next_idx = meta["next_idx"]
        
        return True
    
    def __len__(self) -> int:
        """Return number of vectors in index."""
        if self._index is None:
            return 0
        return self._index.ntotal
    
    @property
    def size(self) -> int:
        """Alias for len()."""
        return len(self)


class DualANNIndex:
    """Manager for both numeric and text embedding indices.
    
    Provides a unified interface for memory retrieval using both
    numeric feature vectors and text embeddings.
    """
    
    def __init__(
        self,
        numeric_dimension: int = 64,
        text_dimension: int = 384,  # Default for all-MiniLM-L6-v2
        index_type: str = "flat",
        data_dir: str = "backend/data"
    ):
        """Initialize dual index manager.
        
        Args:
            numeric_dimension: Dimension of numeric feature vectors
            text_dimension: Dimension of text embeddings
            index_type: FAISS index type
            data_dir: Directory for persisting indices
        """
        self.data_dir = data_dir
        self.numeric_index = ANNIndex(numeric_dimension, index_type)
        self.text_index = ANNIndex(text_dimension, index_type)
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def insert_memory(
        self,
        memory_id: str,
        numeric_vector: Optional[np.ndarray] = None,
        text_embedding: Optional[np.ndarray] = None
    ) -> None:
        """Insert vectors for a memory.
        
        Args:
            memory_id: Unique memory identifier
            numeric_vector: Numeric feature vector (optional)
            text_embedding: Text embedding vector (optional)
        """
        if numeric_vector is not None:
            self.numeric_index.insert(numeric_vector, memory_id)
        if text_embedding is not None:
            self.text_index.insert(text_embedding, memory_id)
    
    def search_numeric(
        self,
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search by numeric vector similarity."""
        return self.numeric_index.search(query_vector, k)
    
    def search_text(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search by text embedding similarity."""
        return self.text_index.search(query_embedding, k)
    
    def search_combined(
        self,
        numeric_vector: Optional[np.ndarray] = None,
        text_embedding: Optional[np.ndarray] = None,
        k: int = 10,
        numeric_weight: float = 0.5,
        text_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Combined search using both indices.
        
        Merges results from both indices with weighted scoring.
        
        Args:
            numeric_vector: Numeric query vector (optional)
            text_embedding: Text query embedding (optional)
            k: Number of results to return
            numeric_weight: Weight for numeric similarity (0-1)
            text_weight: Weight for text similarity (0-1)
        
        Returns:
            List of (memory_id, combined_score) tuples
        """
        scores: Dict[str, float] = {}
        
        if numeric_vector is not None and len(self.numeric_index) > 0:
            numeric_results = self.numeric_index.search(numeric_vector, k * 2)
            # Convert distances to similarities (inverse)
            if numeric_results:
                max_dist = max(r[1] for r in numeric_results) or 1.0
                for mem_id, dist in numeric_results:
                    sim = 1.0 - (dist / (max_dist + 1e-6))
                    scores[mem_id] = scores.get(mem_id, 0) + numeric_weight * sim
        
        if text_embedding is not None and len(self.text_index) > 0:
            text_results = self.text_index.search(text_embedding, k * 2)
            if text_results:
                max_dist = max(r[1] for r in text_results) or 1.0
                for mem_id, dist in text_results:
                    sim = 1.0 - (dist / (max_dist + 1e-6))
                    scores[mem_id] = scores.get(mem_id, 0) + text_weight * sim
        
        # Sort by combined score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from both indices."""
        self.numeric_index.remove(memory_id)
        self.text_index.remove(memory_id)
    
    def save(self) -> None:
        """Save both indices to disk."""
        self.numeric_index.save(os.path.join(self.data_dir, "ann_numeric"))
        self.text_index.save(os.path.join(self.data_dir, "ann_text"))
    
    def load(self) -> bool:
        """Load both indices from disk.
        
        Returns:
            True if both loaded successfully
        """
        num_ok = self.numeric_index.load(os.path.join(self.data_dir, "ann_numeric"))
        text_ok = self.text_index.load(os.path.join(self.data_dir, "ann_text"))
        return num_ok and text_ok
    
    def rebuild(
        self,
        numeric_vectors: Dict[str, np.ndarray],
        text_embeddings: Dict[str, np.ndarray]
    ) -> None:
        """Rebuild both indices from scratch.
        
        Args:
            numeric_vectors: Dict of memory_id -> numeric vector
            text_embeddings: Dict of memory_id -> text embedding
        """
        if numeric_vectors:
            ids = list(numeric_vectors.keys())
            vectors = np.array([numeric_vectors[id_] for id_ in ids], dtype=np.float32)
            self.numeric_index.build(vectors, ids)
        
        if text_embeddings:
            ids = list(text_embeddings.keys())
            vectors = np.array([text_embeddings[id_] for id_ in ids], dtype=np.float32)
            self.text_index.build(vectors, ids)
