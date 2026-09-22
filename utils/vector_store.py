"""
utils/vector_store.py
======================
COMPONENTS: Vector Database + Similarity Search + Retrieval
------------------------------------------------------------
This module wraps Facebook's FAISS library to provide:

  Vector Database   — an in-memory index that stores all document embeddings
  Similarity Search — given a query vector, find the nearest stored vectors
  Retrieval         — return the DocumentChunk objects that match a query

Why FAISS?
  - Runs 100% locally (no cloud account, no API key)
  - Handles millions of vectors efficiently
  - Supported on all platforms via the faiss-cpu package
  - Perfect for Streamlit demos and student projects

Index type used: IndexFlatL2
  - Exact (brute-force) L2-distance search — always correct
  - Sufficient for small to medium document sets
  - For large corpora, swap to IndexIVFFlat or IndexHNSWFlat
"""

from typing import List, Tuple
import numpy as np
import faiss

from utils.document_processor import DocumentChunk


class FAISSVectorStore:
    """
    In-memory FAISS vector store that pairs each vector with its DocumentChunk.

    Lifecycle
    ---------
    1. Instantiate: store = FAISSVectorStore()
    2. Add chunks:  store.add_chunks(chunks, embeddings)
    3. Search:      results = store.similarity_search(query_vector, top_k=5)
    4. Reset:       store.reset()
    """

    def __init__(self):
        self._index: faiss.Index | None = None  # FAISS index
        self._chunks: List[DocumentChunk] = []   # parallel list of chunks
        self._dimension: int = 0

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        VECTOR DATABASE — Store chunk embeddings in the FAISS index.

        Parameters
        ----------
        chunks     : list of DocumentChunk (must match embeddings row-count)
        embeddings : float32 array of shape (N, D)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )
        if len(chunks) == 0:
            raise ValueError("No chunks to add to the vector store.")

        dimension = embeddings.shape[1]

        if self._index is None:
            # First batch — create index
            self._dimension = dimension
            self._index = faiss.IndexFlatL2(dimension)
        elif dimension != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {dimension}."
            )

        # FAISS requires contiguous float32 arrays
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        SIMILARITY SEARCH + RETRIEVAL
        --------------------------------
        Convert a query embedding into the top-k most similar document chunks.

        Steps:
          1. Reshape query to (1, D) for FAISS
          2. Run IndexFlatL2.search — returns distances + integer IDs
          3. Map integer IDs back to DocumentChunk objects
          4. Return (chunk, distance) pairs sorted by distance (closest first)

        Parameters
        ----------
        query_vector : 1-D float32 array of shape (D,)
        top_k        : number of results to return

        Returns
        -------
        List of (DocumentChunk, float) sorted by ascending L2 distance
        """
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError(
                "Vector store is empty. Upload and process documents first."
            )

        # Clamp top_k so we never ask for more results than indexed vectors
        top_k = min(top_k, self._index.ntotal)

        query = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )

        distances, indices = self._index.search(query, top_k)

        results: List[Tuple[DocumentChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:          # FAISS returns -1 for padding when fewer results exist
                continue
            results.append((self._chunks[idx], float(dist)))

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored vectors and chunks."""
        self._index = None
        self._chunks = []
        self._dimension = 0

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently stored in the index."""
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        """True when the store has at least one chunk indexed."""
        return self._index is not None and self._index.ntotal > 0
