"""FAISS index builder and searcher for chunk-level retrieval."""

from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from app.schemas.chunk_schema import ChunkCollection


class FaissIndexManager:
    """Builds and queries a FAISS index over document chunk embeddings.

    Uses IndexFlatIP (inner product) which computes cosine similarity
    when vectors are L2-normalized. This matches the student competition
    setup where bge-small-en-v1.5 embeddings are normalized.

    Attributes:
        _index: FAISS index instance.
        _chunk_collection: Reference to the chunk metadata for result mapping.
    """

    def __init__(self) -> None:
        """Initialize an empty index manager."""
        self._index: faiss.Index | None = None
        self._chunk_collection: ChunkCollection | None = None

    def build_index(self, embeddings: np.ndarray, collection: ChunkCollection) -> None:
        """Build a FAISS IndexFlatIP from chunk embeddings.

        Args:
            embeddings: Numpy array of shape (num_chunks, embedding_dim),
                must be L2-normalized for cosine similarity.
            collection: ChunkCollection providing metadata for each embedding row.

        Raises:
            ValueError: If embeddings and collection have mismatched sizes.
        """
        if embeddings.shape[0] != collection.total_chunks:
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]}) and chunks ({collection.total_chunks}) count mismatch"
            )

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._chunk_collection = collection
        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def search(self, query_embeddings: np.ndarray, top_k: int = 10) -> list[list[dict]]:
        """Search the index for nearest chunks to each query.

        Args:
            query_embeddings: Numpy array of shape (num_queries, embedding_dim),
                must be L2-normalized.
            top_k: Number of nearest neighbors to return per query.

        Returns:
            List of lists, one per query. Each inner list contains top_k dicts
            with keys: 'chunk_index', 'doc_id', 'section_title', 'score',
            'content_preview'.

        Raises:
            RuntimeError: If index has not been built.
        """
        if self._index is None or self._chunk_collection is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        scores, indices = self._index.search(query_embeddings.astype(np.float32), top_k)

        results: list[list[dict]] = []
        for query_idx in range(len(query_embeddings)):
            query_results: list[dict] = []
            for rank in range(top_k):
                chunk_idx = int(indices[query_idx][rank])
                if chunk_idx < 0:
                    continue
                chunk = self._chunk_collection.chunks[chunk_idx]
                query_results.append({
                    "rank": rank + 1,
                    "chunk_global_index": chunk_idx,
                    "doc_id": chunk.doc_id,
                    "subtopic": chunk.subtopic,
                    "score": float(scores[query_idx][rank]),
                    "content_preview": chunk.content[:200],
                })
            results.append(query_results)

        return results

    def search_single(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Search for a single query vector.

        Args:
            query_embedding: 1D numpy array of shape (embedding_dim,).
            top_k: Number of results to return.

        Returns:
            List of top_k result dicts.
        """
        query_2d = query_embedding.reshape(1, -1)
        return self.search(query_2d, top_k)[0]

    def find_doc_rank(self, query_embedding: np.ndarray, target_doc_id: str, max_k: int = 50) -> dict:
        """Find the rank of the first chunk from a specific document.

        Used for validating consultas_centro_control.json: given a query,
        how far down the ranking does the assigned doc_id first appear?

        Args:
            query_embedding: 1D query vector.
            target_doc_id: Document ID to search for in results.
            max_k: Maximum depth to search.

        Returns:
            Dict with 'rank' (1-based position of first chunk from target doc,
            or -1 if not found), 'score' of that chunk, 'top_docs' showing
            the doc_ids of the top-5 results.
        """
        results = self.search_single(query_embedding, top_k=max_k)

        target_rank = -1
        target_score = 0.0
        for result in results:
            if result["doc_id"] == target_doc_id:
                target_rank = result["rank"]
                target_score = result["score"]
                break

        top_docs = [r["doc_id"] for r in results[:5]]
        top_scores = [r["score"] for r in results[:5]]

        return {
            "target_doc_id": target_doc_id,
            "rank": target_rank,
            "score": target_score,
            "top5_docs": top_docs,
            "top5_scores": top_scores,
        }

    def save_index(self, output_path: Path) -> None:
        """Persist the FAISS index to disk.

        Args:
            output_path: Destination file path.

        Raises:
            RuntimeError: If index has not been built.
        """
        if self._index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(output_path))
        logger.info(f"FAISS index saved to {output_path}")

    def load_index(self, index_path: Path, collection: ChunkCollection) -> None:
        """Load a FAISS index from disk.

        Args:
            index_path: Path to the saved FAISS index file.
            collection: ChunkCollection for metadata mapping.

        Raises:
            FileNotFoundError: If index file does not exist.
        """
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self._index = faiss.read_index(str(index_path))
        self._chunk_collection = collection
        logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal if self._index else 0
