"""Embedding utilities for baseline RAG solution."""

import numpy as np
from sentence_transformers import SentenceTransformer


def load_encoder(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cuda") -> SentenceTransformer:
    """Load the sentence transformer encoder.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device.

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name, device=device)


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into normalized embeddings.

    Args:
        model: Loaded SentenceTransformer.
        texts: List of text strings to encode.
        batch_size: Encoding batch size.

    Returns:
        Numpy array of shape (len(texts), embedding_dim), L2-normalized.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
