"""Reciprocal Rank Fusion (RRF) reranker for combining multiple retrieval lists."""

from loguru import logger


RRF_K = 60


class RRFReranker:
    """Combines multiple ranked retrieval lists using Reciprocal Rank Fusion.

    RRF is a model-free reranking strategy that merges results from different
    retrieval runs (e.g., multiple query variations through the same encoder).
    Each chunk receives a fused score: RRF(d) = sum(1 / (k + rank_i)) across
    all rank lists where it appears.

    This approach is particularly effective when query variations capture
    different semantic aspects, and the combination surfaces chunks that
    consistently rank well across formulations.

    Attributes:
        _k: RRF constant controlling rank importance decay. Standard value is 60.
    """

    def __init__(self, k: int = RRF_K) -> None:
        """Initialize the RRF reranker.

        Args:
            k: RRF constant. Higher values give more weight to lower-ranked
                results. Standard value is 60 (from the original RRF paper).
        """
        self._k = k

    def fuse(
        self,
        rank_lists: list[list[dict]],
        top_k: int = 5,
        id_key: str = "chunk_global_index",
    ) -> list[dict]:
        """Fuse multiple ranked lists into a single reranked list.

        Args:
            rank_lists: List of ranked result lists. Each inner list contains
                dicts with at least the id_key field and a 'score' field.
                Results should be ordered by rank (best first).
            top_k: Number of top results to return after fusion.
            id_key: Key used to identify unique chunks across lists.

        Returns:
            Top-K chunks sorted by RRF score (descending), each augmented
            with 'rrf_score' and 'appearances' fields.
        """
        rrf_scores: dict[int, float] = {}
        chunk_data: dict[int, dict] = {}
        appearance_count: dict[int, int] = {}

        for list_idx, rank_list in enumerate(rank_lists):
            for rank, result in enumerate(rank_list, start=1):
                chunk_id = result[id_key]
                rrf_score = 1.0 / (self._k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score
                appearance_count[chunk_id] = appearance_count.get(chunk_id, 0) + 1

                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = dict(result)

        fused_results = []
        for chunk_id, score in rrf_scores.items():
            entry = dict(chunk_data[chunk_id])
            entry["rrf_score"] = round(score, 6)
            entry["appearances"] = appearance_count[chunk_id]
            entry["appeared_in_lists"] = appearance_count[chunk_id]
            fused_results.append(entry)

        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        logger.debug(
            f"RRF fused {len(rank_lists)} lists, "
            f"{len(rrf_scores)} unique chunks, returning top-{top_k}"
        )

        return fused_results[:top_k]

    def fuse_batch(
        self,
        batch_rank_lists: list[list[list[dict]]],
        top_k: int = 5,
        id_key: str = "chunk_global_index",
    ) -> list[list[dict]]:
        """Fuse rank lists for multiple queries.

        Args:
            batch_rank_lists: For each query, a list of rank lists to fuse.
                Shape: [num_queries][num_variations][num_results].
            top_k: Number of top results per query.
            id_key: Key for chunk identification.

        Returns:
            List of fused top-K results, one per query.
        """
        return [
            self.fuse(rank_lists, top_k, id_key)
            for rank_lists in batch_rank_lists
        ]
