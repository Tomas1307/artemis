"""Embeds document chunks using sentence-transformers models."""

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.schemas.chunk_schema import ChunkCollection

ENCODER_MODEL = "BAAI/bge-small-en-v1.5"


class ChunkEmbedder:
    """Generates dense vector embeddings for document chunks.

    Uses sentence-transformers to encode chunk texts into normalized vectors
    suitable for cosine similarity search via FAISS IndexFlatIP.

    Attributes:
        _model_name: HuggingFace model identifier.
        _model: Loaded SentenceTransformer instance (lazy-loaded).
        _batch_size: Encoding batch size.
    """

    def __init__(self, model_name: str = ENCODER_MODEL, batch_size: int = 64) -> None:
        """Initialize the embedder.

        Args:
            model_name: HuggingFace model identifier for the encoder.
                Defaults to bge-small-en-v1.5 (ARTEMIS competition encoder).
            batch_size: Number of texts to encode per batch. Higher values
                use more GPU memory but are faster.
        """
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._batch_size = batch_size

    def load_model(self, device: str = "cuda") -> None:
        """Load the sentence-transformers model onto the specified device.

        Args:
            device: Target device ('cuda', 'cpu', or 'cuda:N').
        """
        logger.info(f"Loading encoder model: {self._model_name} on {device}")
        self._model = SentenceTransformer(self._model_name, device=device)
        logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")

    def embed_chunks(self, collection: ChunkCollection, show_progress: bool = True) -> np.ndarray:
        """Encode all chunks in a collection into dense vectors.

        Args:
            collection: ChunkCollection with embedding_text populated.
            show_progress: Whether to display a progress bar.

        Returns:
            Numpy array of shape (num_chunks, embedding_dim) with L2-normalized
            vectors ready for cosine similarity via inner product.

        Raises:
            RuntimeError: If model has not been loaded via load_model().
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        texts = collection.get_all_embedding_texts()
        logger.info(f"Encoding {len(texts)} chunks with batch_size={self._batch_size}")

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode a list of query strings into dense vectors.

        Args:
            queries: List of query texts to encode.
            show_progress: Whether to display a progress bar.

        Returns:
            Numpy array of shape (num_queries, embedding_dim), L2-normalized.

        Raises:
            RuntimeError: If model has not been loaded via load_model().
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        embeddings = self._model.encode(
            queries,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Save embeddings to a numpy file.

        Args:
            embeddings: Numpy array of embeddings.
            output_path: Destination .npy file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), embeddings)
        logger.info(f"Embeddings saved to {output_path} ({embeddings.shape})")

    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Load embeddings from a numpy file.

        Args:
            input_path: Source .npy file path.

        Returns:
            Numpy array of embeddings.
        """
        embeddings = np.load(str(input_path))
        logger.info(f"Embeddings loaded from {input_path} ({embeddings.shape})")
        return embeddings

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model.

        Raises:
            RuntimeError: If model has not been loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model.get_sentence_embedding_dimension()
