"""Data audit pipeline: chunk, embed, validate consultas, analyze overlap.

Runs the full validation chain at chunk level to identify data quality issues
before Phase 5 encoder/decoder fine-tuning.

Usage (on ml-server03):
    export CUDA_VISIBLE_DEVICES=1
    python -m app.scripts.run_data_audit
    python -m app.scripts.run_data_audit --device cpu
    python -m app.scripts.run_data_audit --skip-chunking  (reuse existing chunks.json)
    python -m app.scripts.run_data_audit --skip-overlap
"""

import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from app.llms.llm_nvidia import NvidiaLLM
from app.processors.chunk_embedder import ChunkEmbedder
from app.processors.document_chunker import DocumentChunker
from app.processors.faiss_index_manager import FaissIndexManager
from app.schemas.chunk_schema import ChunkCollection

PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "proyecto_artemis" / "base_conocimiento"
DOCS_INDEX_PATH = DOCS_DIR / "documentos_masa.json"
CONSULTAS_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "consultas_centro_control.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "data_audit"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "chunk_embeddings.npy"
INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
REPORT_PATH = ARTIFACTS_DIR / "audit_report.json"


def step_01_chunk_documents() -> ChunkCollection:
    """Chunk all 54 MASA documents with LLM-generated summaries.

    Returns:
        ChunkCollection with all chunks and metadata.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Chunking documents with LLM summaries")
    logger.info("=" * 60)

    docs_index = json.loads(DOCS_INDEX_PATH.read_text(encoding="utf-8"))
    llm = NvidiaLLM(temperature=0.1, max_tokens=200)
    chunker = DocumentChunker(llm=llm, max_tokens=384, overlap_tokens=50)
    collection = chunker.chunk_all_documents(DOCS_DIR, docs_index)
    chunker.save_chunks(collection, CHUNKS_PATH)

    doc_chunk_counts = {}
    for chunk in collection.chunks:
        doc_chunk_counts[chunk.doc_id] = doc_chunk_counts.get(chunk.doc_id, 0) + 1

    logger.info(
        f"Chunk distribution: min={min(doc_chunk_counts.values())}, "
        f"max={max(doc_chunk_counts.values())}, "
        f"mean={np.mean(list(doc_chunk_counts.values())):.1f}"
    )

    return collection


def step_01_load_existing_chunks() -> ChunkCollection:
    """Load previously generated chunks from disk.

    Returns:
        ChunkCollection loaded from chunks.json.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Loading existing chunks (--skip-chunking)")
    logger.info("=" * 60)

    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    collection = ChunkCollection(**data)
    logger.info(f"Loaded {collection.total_chunks} chunks from {CHUNKS_PATH}")
    return collection


def step_02_embed_chunks(collection: ChunkCollection, device: str) -> tuple[np.ndarray, ChunkEmbedder]:
    """Embed all chunks with the base encoder (no fine-tuning).

    Args:
        collection: ChunkCollection to embed.
        device: Torch device string.

    Returns:
        Tuple of (embeddings array, loaded embedder for query encoding).
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Embedding chunks")
    logger.info("=" * 60)

    embedder = ChunkEmbedder()
    embedder.load_model(device=device)
    embeddings = embedder.embed_chunks(collection)
    embedder.save_embeddings(embeddings, EMBEDDINGS_PATH)

    return embeddings, embedder


def step_03_build_index(embeddings: np.ndarray, collection: ChunkCollection) -> FaissIndexManager:
    """Build FAISS index from chunk embeddings.

    Args:
        embeddings: Chunk embedding array.
        collection: ChunkCollection for metadata.

    Returns:
        Configured FaissIndexManager.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Building FAISS index")
    logger.info("=" * 60)

    index_manager = FaissIndexManager()
    index_manager.build_index(embeddings, collection)
    index_manager.save_index(INDEX_PATH)

    return index_manager


def step_04_validate_consultas(
    index_manager: FaissIndexManager,
    embedder: ChunkEmbedder,
    collection: ChunkCollection,
) -> dict:
    """Validate consultas_centro_control.json at chunk level.

    For each (query, doc_id) pair, retrieves top-50 chunks and checks
    where the assigned doc first appears. Reports rank distribution,
    confusion clusters, and broken pairs.

    Args:
        index_manager: Built FAISS index.
        embedder: Loaded ChunkEmbedder for query encoding.
        collection: ChunkCollection for metadata.

    Returns:
        Dict with validation results and statistics.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Validating consultas at chunk level")
    logger.info("=" * 60)

    consultas = json.loads(CONSULTAS_PATH.read_text(encoding="utf-8"))
    queries = [c["query"] for c in consultas]
    query_embeddings = embedder.embed_queries(queries, show_progress=True)

    rank_buckets = {"rank_1": 0, "rank_2_3": 0, "rank_4_5": 0, "rank_6_10": 0, "rank_11_plus": 0, "not_found": 0}
    confusion_clusters: dict[str, dict[str, int]] = {}
    broken_pairs: list[dict] = []
    all_results: list[dict] = []

    for idx, consulta in enumerate(consultas):
        target_doc = consulta["doc_id"]
        result = index_manager.find_doc_rank(query_embeddings[idx], target_doc)
        rank = result["rank"]

        entry = {
            "query_index": idx,
            "query": consulta["query"][:150],
            "assigned_doc": target_doc,
            "chunk_rank": rank,
            "top5_docs": result["top5_docs"],
            "top5_scores": [round(s, 4) for s in result["top5_scores"]],
        }
        all_results.append(entry)

        if rank == 1:
            rank_buckets["rank_1"] += 1
        elif rank <= 3:
            rank_buckets["rank_2_3"] += 1
        elif rank <= 5:
            rank_buckets["rank_4_5"] += 1
        elif rank <= 10:
            rank_buckets["rank_6_10"] += 1
        elif rank > 10:
            rank_buckets["rank_11_plus"] += 1
        else:
            rank_buckets["not_found"] += 1

        if rank > 3:
            broken_pairs.append(entry)
            top1_doc = result["top5_docs"][0] if result["top5_docs"] else "N/A"
            if target_doc not in confusion_clusters:
                confusion_clusters[target_doc] = {}
            confusion_clusters[target_doc][top1_doc] = confusion_clusters[target_doc].get(top1_doc, 0) + 1

    logger.info(f"Rank distribution: {rank_buckets}")
    logger.info(f"Broken pairs (rank > 3): {len(broken_pairs)}")

    confuser_summary = {}
    for doc_id, confusers in confusion_clusters.items():
        top_confuser = max(confusers, key=confusers.get)
        confuser_summary[doc_id] = {
            "total_broken": sum(confusers.values()),
            "top_confuser": top_confuser,
            "confuser_count": confusers[top_confuser],
            "all_confusers": confusers,
        }

    return {
        "total_consultas": len(consultas),
        "rank_distribution": rank_buckets,
        "broken_pairs_count": len(broken_pairs),
        "broken_pairs": broken_pairs[:50],
        "confusion_clusters": confuser_summary,
        "all_results": all_results,
    }


def step_05_analyze_overlap(embeddings: np.ndarray, collection: ChunkCollection) -> dict:
    """Compute cross-document similarity to find confusion-prone doc pairs.

    Averages chunk embeddings per document, then computes pairwise cosine
    similarity. Flags document pairs with high similarity.

    Args:
        embeddings: Chunk embedding array.
        collection: ChunkCollection for doc-level aggregation.

    Returns:
        Dict with pairwise similarity matrix and flagged pairs.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Cross-document overlap analysis")
    logger.info("=" * 60)

    doc_ids = collection.get_unique_doc_ids()
    doc_embeddings = []

    for doc_id in doc_ids:
        doc_chunks = collection.get_chunks_for_doc(doc_id)
        chunk_indices = [collection.chunks.index(c) for c in doc_chunks]
        doc_emb = embeddings[chunk_indices].mean(axis=0)
        doc_emb = doc_emb / np.linalg.norm(doc_emb)
        doc_embeddings.append(doc_emb)

    doc_matrix = np.array(doc_embeddings)
    similarity_matrix = np.dot(doc_matrix, doc_matrix.T)

    high_similarity_pairs: list[dict] = []
    threshold = 0.75

    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            sim = float(similarity_matrix[i][j])
            if sim >= threshold:
                high_similarity_pairs.append({
                    "doc_a": doc_ids[i],
                    "doc_b": doc_ids[j],
                    "similarity": round(sim, 4),
                })

    high_similarity_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    super_doc_scores: list[dict] = []
    for i, doc_id in enumerate(doc_ids):
        avg_sim = float(np.mean([similarity_matrix[i][j] for j in range(len(doc_ids)) if j != i]))
        super_doc_scores.append({"doc_id": doc_id, "avg_similarity": round(avg_sim, 4)})
    super_doc_scores.sort(key=lambda x: x["avg_similarity"], reverse=True)

    logger.info(f"High similarity pairs (>= {threshold}): {len(high_similarity_pairs)}")
    logger.info(f"Top 'super document': {super_doc_scores[0]}")

    return {
        "threshold": threshold,
        "high_similarity_pairs": high_similarity_pairs,
        "super_doc_ranking": super_doc_scores[:10],
    }


def main() -> None:
    """Run the full data audit pipeline."""
    args = sys.argv[1:]
    device = "cuda"
    skip_overlap = "--skip-overlap" in args
    skip_chunking = "--skip-chunking" in args

    if "--device" in args:
        device = args[args.index("--device") + 1]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("ARTEMIS Data Audit Pipeline")
    logger.info(f"Device: {device}, Skip chunking: {skip_chunking}, Skip overlap: {skip_overlap}")
    logger.info("=" * 60)

    if skip_chunking and CHUNKS_PATH.exists():
        collection = step_01_load_existing_chunks()
    else:
        collection = step_01_chunk_documents()

    embeddings, embedder = step_02_embed_chunks(collection, device)

    index_manager = step_03_build_index(embeddings, collection)

    consultas_report = step_04_validate_consultas(index_manager, embedder, collection)

    overlap_report = {}
    if not skip_overlap:
        overlap_report = step_05_analyze_overlap(embeddings, collection)

    report = {
        "chunk_stats": {
            "total_documents": collection.total_documents,
            "total_chunks": collection.total_chunks,
            "chunks_per_doc": {
                doc_id: len(collection.get_chunks_for_doc(doc_id))
                for doc_id in collection.get_unique_doc_ids()
            },
        },
        "consultas_validation": consultas_report,
        "overlap_analysis": overlap_report,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Full audit report saved to {REPORT_PATH}")

    print_summary(report)


def print_summary(report: dict) -> None:
    """Print a human-readable summary of the audit results.

    Args:
        report: Full audit report dict.
    """
    print("\n" + "=" * 70)
    print("ARTEMIS DATA AUDIT SUMMARY")
    print("=" * 70)

    cs = report["chunk_stats"]
    print(f"\nDocuments: {cs['total_documents']} -> Chunks: {cs['total_chunks']}")
    chunk_counts = list(cs["chunks_per_doc"].values())
    print(f"Chunks per doc: min={min(chunk_counts)}, max={max(chunk_counts)}, "
          f"mean={np.mean(chunk_counts):.1f}")

    cv = report["consultas_validation"]
    rd = cv["rank_distribution"]
    print(f"\nConsultas validation ({cv['total_consultas']} pairs):")
    print(f"  Rank 1:     {rd['rank_1']:>4}  (encoder finds right doc chunk first)")
    print(f"  Rank 2-3:   {rd['rank_2_3']:>4}  (close, might work with fine-tuning)")
    print(f"  Rank 4-5:   {rd['rank_4_5']:>4}  (needs fine-tuning)")
    print(f"  Rank 6-10:  {rd['rank_6_10']:>4}  (problematic)")
    print(f"  Rank 11+:   {rd['rank_11_plus']:>4}  (likely broken pairs)")
    print(f"  Not found:  {rd['not_found']:>4}  (doc not in top-50)")

    if cv.get("confusion_clusters"):
        print(f"\nTop confusion clusters (docs most often confused):")
        sorted_clusters = sorted(
            cv["confusion_clusters"].items(),
            key=lambda x: x[1]["total_broken"],
            reverse=True,
        )
        for doc_id, info in sorted_clusters[:10]:
            print(f"  {doc_id}: {info['total_broken']} broken, top confuser: {info['top_confuser']}")

    if report.get("overlap_analysis"):
        oa = report["overlap_analysis"]
        if oa.get("super_doc_ranking"):
            print(f"\nTop 'super documents' (highest avg similarity to all others):")
            for entry in oa["super_doc_ranking"][:5]:
                print(f"  {entry['doc_id']}: avg_sim={entry['avg_similarity']}")

        if oa.get("high_similarity_pairs"):
            print(f"\nHigh similarity document pairs (>= {oa['threshold']}):")
            for pair in oa["high_similarity_pairs"][:10]:
                print(f"  {pair['doc_a']} <-> {pair['doc_b']}: {pair['similarity']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
