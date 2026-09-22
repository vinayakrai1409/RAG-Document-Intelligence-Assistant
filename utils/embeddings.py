"""
utils/embeddings.py
====================
COMPONENT: Embeddings
---------------------
This module converts text (either document chunks or user questions) into
dense numerical vectors called "embeddings".

Embeddings capture the *semantic meaning* of text so that similar meanings
map to nearby points in vector space — this is what makes similarity search
possible.

Model used: sentence-transformers/all-MiniLM-L6-v2
  - Lightweight (~80 MB), runs entirely on CPU
  - 384-dimensional output vectors
  - No API key required — runs locally

The module is intentionally kept thin so you can swap in a different model
(e.g., OpenAI text-embedding-3-small, Google textembedding-gecko) by
changing only this file.
"""

from typing import List
import numpy as np

# SentenceTransformer is loaded lazily to avoid slowing down Streamlit startup
_model = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    """
    Lazily load and cache the embedding model.
    The model is downloaded once and cached by the sentence-transformers library.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    EMBEDDINGS — Convert a list of text strings into an (N, D) float32 array.

    Parameters
    ----------
    texts : list of str
        The text strings to embed (document chunks or queries).

    Returns
    -------
    np.ndarray, shape (len(texts), embedding_dim), dtype float32
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")

    model = _get_model()
    # batch_size=32 works well on CPU; show_progress_bar=False keeps Streamlit clean
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Convenience wrapper: embed a single query string.

    Returns a 1-D float32 array of shape (embedding_dim,).
    """
    if not query or not query.strip():
        raise ValueError("Query text is empty.")
    result = embed_texts([query])
    return result[0]


def get_embedding_dimension() -> int:
    """Return the output dimension of the current embedding model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
