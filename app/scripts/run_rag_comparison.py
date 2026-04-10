"""RAG strategy comparison: basic vs RRF vs full pipeline with CoT reasoning.

Compares three retrieval-augmented generation strategies on data.csv and
test_queries.csv to validate data quality and measure tool-call accuracy.

Strategies:
  1. Basic RAG: bge-small top-5 → Qwen direct → tool_call
  2. RRF RAG: 3 query variations → 3 × bge-small → RRF top-5 → Qwen direct
  3. Full Pipeline: variations → RRF → Qwen CoT reasoning → tool_call

Usage (on ml-server03):
    export CUDA_VISIBLE_DEVICES=0
    python -m app.scripts.run_rag_comparison --sample 300 --device cuda
    python -m app.scripts.run_rag_comparison --test-only --device cuda
    python -m app.scripts.run_rag_comparison --strategy basic --device cuda
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from app.chain_methods.llm_query_reprompt import LLMQueryReprompt
from app.chain_methods.llm_tool_reasoner import LLMToolReasoner
from app.processors.chunk_embedder import ChunkEmbedder
from app.processors.faiss_index_manager import FaissIndexManager
from app.processors.local_llm import LocalLLM
from app.processors.reranker import RRFReranker
from app.schemas.chunk_schema import ChunkCollection

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_PATH = PROJECT_ROOT / "artifacts" / "data_audit" / "chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "artifacts" / "data_audit" / "chunk_embeddings.npy"
DATA_CSV_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "data.csv"
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_CSV_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_queries.csv"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"
TOOLS_PATH = PROJECT_ROOT / "proyecto_artemis" / "tools_definition.json"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "rag_comparison"


def load_data(sample_size: int | None, test_only: bool) -> tuple[list[dict], list[dict]]:
    """Load train and test queries with gold standard answers.

    Args:
        sample_size: If set, randomly sample this many train queries.
        test_only: If True, skip train queries entirely.

    Returns:
        Tuple of (train_queries, test_queries). Each query is a dict
        with 'id', 'query', and 'gold_tool_call' keys.
    """
    train_queries = []
    if not test_only:
        with open(DATA_CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_queries.append({
                    "id": row["id"],
                    "query": row["query"],
                    "gold_tool_call": row["tool_call"],
                })

        if sample_size and sample_size < len(train_queries):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(train_queries), sample_size, replace=False)
            train_queries = [train_queries[i] for i in sorted(indices)]

    test_queries = []
    test_gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))
    gold_by_id = {g["question_id"]: g["tool_call"] for g in test_gold}
    with open(TEST_CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_queries.append({
                "id": row["id"],
                "query": row["query"],
                "gold_tool_call": gold_by_id.get(row["id"], ""),
            })

    logger.info(f"Loaded {len(train_queries)} train, {len(test_queries)} test queries")
    return train_queries, test_queries


def load_infrastructure(device: str) -> tuple[ChunkCollection, np.ndarray, FaissIndexManager, ChunkEmbedder]:
    """Load chunks, embeddings, FAISS index, and encoder.

    Args:
        device: Torch device for the encoder.

    Returns:
        Tuple of (collection, embeddings, index_manager, embedder).
    """
    chunk_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    collection = ChunkCollection(**chunk_data)
    logger.info(f"Loaded {collection.total_chunks} chunks")

    embeddings = np.load(str(EMBEDDINGS_PATH))

    index_manager = FaissIndexManager()
    index_manager.build_index(embeddings, collection)

    embedder = ChunkEmbedder()
    embedder.load_model(device=device)

    return collection, embeddings, index_manager, embedder


def retrieve_basic(
    queries: list[str],
    embedder: ChunkEmbedder,
    index_manager: FaissIndexManager,
    top_k: int = 5,
) -> list[list[dict]]:
    """Basic retrieval: single query → bi-encoder → top-K chunks.

    Args:
        queries: List of query strings.
        embedder: Loaded ChunkEmbedder.
        index_manager: Built FAISS index.
        top_k: Number of chunks to retrieve per query.

    Returns:
        List of chunk result lists, one per query.
    """
    query_embeddings = embedder.embed_queries(queries, show_progress=True)
    return index_manager.search(query_embeddings, top_k=top_k)


def retrieve_rrf(
    queries: list[str],
    variations_per_query: list[list[str]],
    embedder: ChunkEmbedder,
    index_manager: FaissIndexManager,
    rrf_reranker: RRFReranker,
    retrieve_k: int = 20,
    final_k: int = 5,
) -> list[list[dict]]:
    """RRF retrieval: multiple variations → multiple searches → RRF fusion.

    Args:
        queries: Original query strings (for logging).
        variations_per_query: List of variation lists, one per query.
        embedder: Loaded ChunkEmbedder.
        index_manager: Built FAISS index.
        rrf_reranker: RRFReranker instance.
        retrieve_k: Chunks to retrieve per variation.
        final_k: Chunks to keep after RRF fusion.

    Returns:
        List of fused chunk result lists, one per query.
    """
    all_fused = []

    for i, variations in enumerate(variations_per_query):
        rank_lists = []
        for variation in variations:
            emb = embedder.embed_queries([variation])
            results = index_manager.search(emb, top_k=retrieve_k)
            rank_lists.append(results[0])

        fused = rrf_reranker.fuse(rank_lists, top_k=final_k)
        all_fused.append(fused)

    return all_fused


def enrich_chunks(results: list[list[dict]], collection: ChunkCollection) -> list[list[dict]]:
    """Add full chunk content to search results.

    Args:
        results: Search results with chunk_global_index.
        collection: ChunkCollection with full chunk data.

    Returns:
        Results augmented with 'content', 'subtopic', 'embedding_text'.
    """
    enriched = []
    for query_results in results:
        enriched_query = []
        for result in query_results:
            idx = result["chunk_global_index"]
            chunk = collection.chunks[idx]
            augmented = dict(result)
            augmented["content"] = chunk.content
            augmented["subtopic"] = chunk.subtopic
            augmented["embedding_text"] = chunk.embedding_text
            enriched_query.append(augmented)
        enriched.append(enriched_query)
    return enriched


def evaluate(predictions: list[str], gold: list[str]) -> dict:
    """Compute exact match accuracy.

    Args:
        predictions: List of predicted tool calls.
        gold: List of gold standard tool calls.

    Returns:
        Dict with accuracy, correct count, total count, and error examples.
    """
    correct = 0
    errors = []
    for pred, g in zip(predictions, gold):
        if pred.strip() == g.strip():
            correct += 1
        else:
            if len(errors) < 20:
                errors.append({"predicted": pred, "gold": g})

    return {
        "accuracy": round(correct / len(gold), 4) if gold else 0,
        "correct": correct,
        "total": len(gold),
        "sample_errors": errors,
    }


def run_strategy(
    strategy: str,
    queries: list[dict],
    collection: ChunkCollection,
    embedder: ChunkEmbedder,
    index_manager: FaissIndexManager,
    tools_json: str,
    llm: LocalLLM,
    reprompt: LLMQueryReprompt | None = None,
    rrf: RRFReranker | None = None,
    reasoner: LLMToolReasoner | None = None,
) -> dict:
    """Run a single RAG strategy on a set of queries.

    Args:
        strategy: One of 'basic', 'rrf', 'full'.
        queries: List of query dicts with 'id', 'query', 'gold_tool_call'.
        collection: Chunk collection for enrichment.
        embedder: Loaded encoder.
        index_manager: FAISS index.
        tools_json: Tool definitions as JSON string.
        llm: Local LLM for generation.
        reprompt: Query variation generator (for rrf/full strategies).
        rrf: RRF reranker (for rrf/full strategies).
        reasoner: Tool reasoner chain method.

    Returns:
        Dict with strategy name, accuracy metrics, and detailed results.
    """
    logger.info(f"Running strategy: {strategy} on {len(queries)} queries")
    start_time = time.time()

    query_texts = [q["query"] for q in queries]
    gold_calls = [q["gold_tool_call"] for q in queries]

    if strategy == "basic":
        results = retrieve_basic(query_texts, embedder, index_manager, top_k=5)
        results = enrich_chunks(results, collection)
        predictions = []
        for i, (q, chunks) in enumerate(zip(query_texts, results)):
            if (i + 1) % 25 == 0:
                logger.info(f"[basic] Processing {i + 1}/{len(query_texts)}")
            pred = reasoner.reason_direct(q, chunks, tools_json)
            predictions.append(pred)

    elif strategy == "rrf":
        logger.info("Generating query variations...")
        variations = reprompt.generate_variations_batch(query_texts)
        logger.info("Running RRF retrieval...")
        results = retrieve_rrf(
            query_texts, variations, embedder, index_manager, rrf,
            retrieve_k=20, final_k=5,
        )
        results = enrich_chunks(results, collection)
        predictions = []
        for i, (q, chunks) in enumerate(zip(query_texts, results)):
            if (i + 1) % 25 == 0:
                logger.info(f"[rrf] Processing {i + 1}/{len(query_texts)}")
            pred = reasoner.reason_direct(q, chunks, tools_json)
            predictions.append(pred)

    elif strategy == "full":
        logger.info("Generating query variations...")
        variations = reprompt.generate_variations_batch(query_texts)
        logger.info("Running RRF retrieval...")
        results = retrieve_rrf(
            query_texts, variations, embedder, index_manager, rrf,
            retrieve_k=20, final_k=5,
        )
        results = enrich_chunks(results, collection)
        logger.info("Running CoT reasoning...")
        cot_results = reasoner.reason_batch(
            query_texts, results, tools_json, use_cot=True,
        )
        predictions = [r["tool_call"] for r in cot_results]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    elapsed = time.time() - start_time
    metrics = evaluate(predictions, gold_calls)

    logger.info(
        f"[{strategy}] Accuracy: {metrics['accuracy']:.4f} "
        f"({metrics['correct']}/{metrics['total']}) in {elapsed:.1f}s"
    )

    return {
        "strategy": strategy,
        "metrics": metrics,
        "elapsed_seconds": round(elapsed, 1),
        "num_queries": len(queries),
        "detailed_results": [
            {
                "id": q["id"],
                "query": q["query"][:150],
                "gold": q["gold_tool_call"],
                "predicted": pred,
                "correct": pred.strip() == q["gold_tool_call"].strip(),
            }
            for q, pred in zip(queries, predictions)
        ],
    }


def main() -> None:
    """Run the RAG comparison pipeline."""
    args = sys.argv[1:]
    device = "cuda"
    sample_size = None
    test_only = False
    strategies = ["basic", "rrf", "full"]

    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--sample" in args:
        sample_size = int(args[args.index("--sample") + 1])
    if "--test-only" in args:
        test_only = True
    if "--strategy" in args:
        strategies = [args[args.index("--strategy") + 1]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("ARTEMIS RAG Comparison Pipeline")
    logger.info(f"Device: {device}, Sample: {sample_size}, Test only: {test_only}")
    logger.info(f"Strategies: {strategies}")

    train_queries, test_queries = load_data(sample_size, test_only)

    collection, embeddings, index_manager, embedder = load_infrastructure(device)

    tools_json = TOOLS_PATH.read_text(encoding="utf-8")

    logger.info("Loading local LLM (Qwen2.5-7B-Instruct)...")
    llm = LocalLLM()
    llm.load_model(device=device)

    reprompt = LLMQueryReprompt(llm)
    reasoner = LLMToolReasoner(llm)
    rrf = RRFReranker()

    all_results = {}

    for strategy in strategies:
        if train_queries:
            logger.info(f"\n{'='*60}\nTRAIN SET — Strategy: {strategy}\n{'='*60}")
            train_result = run_strategy(
                strategy, train_queries, collection, embedder, index_manager,
                tools_json, llm, reprompt, rrf, reasoner,
            )
            all_results[f"train_{strategy}"] = train_result

        logger.info(f"\n{'='*60}\nTEST SET — Strategy: {strategy}\n{'='*60}")
        test_result = run_strategy(
            strategy, test_queries, collection, embedder, index_manager,
            tools_json, llm, reprompt, rrf, reasoner,
        )
        all_results[f"test_{strategy}"] = test_result

    output_path = RESULTS_DIR / "comparison_results.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Results saved to {output_path}")

    print_summary(all_results)


def print_summary(all_results: dict) -> None:
    """Print comparison summary table.

    Args:
        all_results: Dict mapping strategy keys to result dicts.
    """
    print("\n" + "=" * 70)
    print("RAG STRATEGY COMPARISON")
    print("=" * 70)

    for key, result in all_results.items():
        dataset, strategy = key.split("_", 1)
        m = result["metrics"]
        print(
            f"  [{dataset:5s}] {strategy:8s}: "
            f"accuracy={m['accuracy']:.4f} "
            f"({m['correct']}/{m['total']}) "
            f"time={result['elapsed_seconds']:.0f}s"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
