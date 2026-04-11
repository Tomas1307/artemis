"""Diagnose RAG failures using LLM classification over retrieved context.

For each wrong prediction, retrieves the same top-5 chunks and asks the
local LLM to classify the failure type by reading the actual context.

Usage (on ml-server03):
    export CUDA_VISIBLE_DEVICES=0
    python -m app.scripts.diagnose_failures --device cuda --split test
    python -m app.scripts.diagnose_failures --device cuda --split train --sample 200
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from app.processors.chunk_embedder import ChunkEmbedder
from app.processors.faiss_index_manager import FaissIndexManager
from app.processors.local_llm import LocalLLM
from app.prompts.prompt_loader import prompt_loader
from app.schemas.chunk_schema import ChunkCollection

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_PATH = PROJECT_ROOT / "artifacts" / "data_audit" / "chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "artifacts" / "data_audit" / "chunk_embeddings.npy"
DATA_CSV_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "data.csv"
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_CSV_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_queries.csv"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "rag_comparison"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "rag_diagnosis"


def load_queries(split: str, sample_size: int | None) -> list[dict]:
    """Load queries with gold standard answers.

    Args:
        split: 'test' or 'train'.
        sample_size: If set, randomly sample this many queries.

    Returns:
        List of query dicts with id, query, gold_tool_call, doc_id, seed_type.
    """
    if split == "test":
        gold_data = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))
        gold_map = {g["question_id"]: g for g in gold_data}
        with open(TEST_CSV_PATH, encoding="utf-8") as f:
            queries = [{"id": r["id"], "query": r["query"]} for r in csv.DictReader(f)]
        for q in queries:
            g = gold_map.get(q["id"], {})
            q["gold_tool_call"] = g.get("tool_call", "")
            q["doc_id"] = g.get("doc_id", "")
            q["seed_type"] = g.get("seed_type", "")
    else:
        gold_data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        gold_map = {g["question_id"]: g for g in gold_data}
        with open(DATA_CSV_PATH, encoding="utf-8") as f:
            queries = [{"id": r["id"], "query": r["query"], "gold_tool_call": r["tool_call"]} for r in csv.DictReader(f)]
        for q in queries:
            g = gold_map.get(q["id"], {})
            q["doc_id"] = g.get("doc_id", "")
            q["seed_type"] = g.get("seed_type", "")

    if sample_size and sample_size < len(queries):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(queries), sample_size, replace=False)
        queries = [queries[i] for i in sorted(indices)]

    return queries


def format_chunks_for_prompt(results: list[dict], collection: ChunkCollection) -> str:
    """Format top-5 retrieved chunks as readable text for the LLM.

    Args:
        results: FAISS search results with chunk_global_index.
        collection: ChunkCollection for full content access.

    Returns:
        Formatted string with numbered chunks showing doc_id, section, and content.
    """
    parts = []
    for r in results:
        idx = r["chunk_global_index"]
        chunk = collection.chunks[idx]
        parts.append(
            f"[Chunk {r['rank']}] {chunk.doc_id} > {chunk.subtopic}\n"
            f"{chunk.content[:500]}"
        )
    return "\n\n".join(parts)


def parse_classification(llm_output: str) -> tuple[str, str]:
    """Parse category and evidence from LLM classification output.

    Args:
        llm_output: Raw LLM response.

    Returns:
        Tuple of (category, evidence).
    """
    valid_categories = [
        "RETRIEVAL_FAILURE", "REASONING_FAILURE",
        "NEAR_MISS", "FORMAT_ERROR", "DATA_ISSUE",
    ]

    category = "UNKNOWN"
    evidence = ""

    cat_match = re.search(r'CATEGORY:\s*(\w+)', llm_output)
    if cat_match:
        raw = cat_match.group(1).upper()
        for vc in valid_categories:
            if raw in vc or vc in raw:
                category = vc
                break

    ev_match = re.search(r'EVIDENCE:\s*(.+)', llm_output, re.DOTALL)
    if ev_match:
        evidence = ev_match.group(1).strip()[:300]

    return category, evidence


def main() -> None:
    """Run LLM-based failure diagnosis."""
    args = sys.argv[1:]
    device = "cuda"
    split = "test"
    sample_size = None

    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--split" in args:
        split = args[args.index("--split") + 1]
    if "--sample" in args:
        sample_size = int(args[args.index("--sample") + 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{split}_basic.json"
    if not checkpoint_path.exists():
        logger.error(f"No checkpoint found: {checkpoint_path}")
        return

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    predictions = checkpoint["predictions"]

    queries = load_queries(split, sample_size)

    if sample_size:
        predictions = predictions[:len(queries)]

    logger.info("Loading infrastructure...")
    chunk_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    collection = ChunkCollection(**chunk_data)
    embeddings = np.load(str(EMBEDDINGS_PATH))

    index_manager = FaissIndexManager()
    index_manager.build_index(embeddings, collection)

    embedder = ChunkEmbedder()
    embedder.load_model(device=device)

    logger.info("Loading local LLM for classification...")
    llm = LocalLLM()
    llm.load_model(device=device)

    query_texts = [q["query"] for q in queries]
    query_embeddings = embedder.embed_queries(query_texts, show_progress=True)

    wrong_indices = [
        i for i, (q, pred) in enumerate(zip(queries, predictions))
        if pred.strip() != q["gold_tool_call"].strip()
    ]
    correct_count = len(queries) - len(wrong_indices)

    logger.info(f"Correct: {correct_count}/{len(queries)}, Wrong: {len(wrong_indices)} — classifying failures...")

    categories = {
        "RETRIEVAL_FAILURE": 0, "REASONING_FAILURE": 0,
        "NEAR_MISS": 0, "FORMAT_ERROR": 0,
        "DATA_ISSUE": 0, "UNKNOWN": 0,
    }
    diagnosis_results = []

    for progress, i in enumerate(wrong_indices):
        if (progress + 1) % 25 == 0:
            logger.info(f"Classifying failure {progress + 1}/{len(wrong_indices)}")

        q = queries[i]
        pred = predictions[i]
        results = index_manager.search_single(query_embeddings[i], top_k=5)
        chunks_text = format_chunks_for_prompt(results, collection)

        system_message = prompt_loader.get_system_message_by_type("failure_classifier")
        template = prompt_loader.get_prompt_template_by_type("failure_classifier")
        config = prompt_loader.get_config_by_type("failure_classifier")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": template.format(
                query=q["query"][:300],
                chunks=chunks_text,
                gold=q["gold_tool_call"],
                predicted=pred,
            )},
        ]

        llm_output = llm.generate(messages, **config)
        category, evidence = parse_classification(llm_output)
        categories[category] += 1

        retrieved_docs = [results[j]["doc_id"] for j in range(len(results))]

        diagnosis_results.append({
            "id": q["id"],
            "query": q["query"][:200],
            "gold_tool_call": q["gold_tool_call"],
            "predicted": pred,
            "gold_doc_id": q.get("doc_id", ""),
            "seed_type": q.get("seed_type", ""),
            "retrieved_docs": retrieved_docs,
            "category": category,
            "evidence": evidence,
        })

    total = len(queries)
    wrong = len(wrong_indices)

    print("\n" + "=" * 70)
    print(f"RAG FAILURE DIAGNOSIS — {split} set, basic strategy")
    print("=" * 70)
    print(f"Total queries:       {total}")
    print(f"Correct:             {correct_count} ({correct_count/total:.1%})")
    print(f"Wrong:               {wrong} ({wrong/total:.1%})")
    print()
    print("FAILURE CLASSIFICATION (by LLM):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / wrong if wrong > 0 else 0
            print(f"  {cat:22s}: {count:>4}  ({pct:.1%} of failures)")

    for cat_name in ["RETRIEVAL_FAILURE", "REASONING_FAILURE", "NEAR_MISS", "DATA_ISSUE"]:
        examples = [d for d in diagnosis_results if d["category"] == cat_name][:3]
        if examples:
            print(f"\n{cat_name} examples:")
            for ex in examples:
                print(f"  {ex['id']}: gold={ex['gold_tool_call'][:60]}")
                print(f"    pred={ex['predicted'][:60]}")
                print(f"    evidence={ex['evidence'][:120]}")

    print("=" * 70)

    output_path = OUTPUT_DIR / f"diagnosis_{split}_basic.json"
    output_path.write_text(
        json.dumps(diagnosis_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Full diagnosis saved to {output_path}")


if __name__ == "__main__":
    main()
