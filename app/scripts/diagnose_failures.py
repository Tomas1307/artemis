"""Diagnose RAG failures: bad retrieval vs bad reasoning vs missing data.

For each wrong prediction, retrieves the same top-5 chunks and checks
whether the gold answer's doc appears in the retrieved context.

Usage (on ml-server03):
    python -m app.scripts.diagnose_failures --device cuda
    python -m app.scripts.diagnose_failures --device cuda --split test
    python -m app.scripts.diagnose_failures --device cuda --split train --sample 200
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from app.processors.chunk_embedder import ChunkEmbedder
from app.processors.faiss_index_manager import FaissIndexManager
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


def main() -> None:
    """Run failure diagnosis on basic strategy predictions."""
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
        predictions = [predictions[i] for i in sorted(indices)]

    logger.info(f"Loading infrastructure...")
    chunk_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    collection = ChunkCollection(**chunk_data)
    embeddings = np.load(str(EMBEDDINGS_PATH))

    index_manager = FaissIndexManager()
    index_manager.build_index(embeddings, collection)

    embedder = ChunkEmbedder()
    embedder.load_model(device=device)

    query_texts = [q["query"] for q in queries]
    query_embeddings = embedder.embed_queries(query_texts, show_progress=True)

    retrieval_hit = 0
    retrieval_miss = 0
    reasoning_fail = 0
    no_doc_queries = 0
    correct_count = 0

    diagnosis_results = []

    for i, (q, pred) in enumerate(zip(queries, predictions)):
        gold = q["gold_tool_call"]
        is_correct = pred.strip() == gold.strip()
        doc_id = q.get("doc_id", "")
        seed_type = q.get("seed_type", "")

        results = index_manager.search_single(query_embeddings[i], top_k=5)
        retrieved_docs = [r["doc_id"] for r in results]
        retrieved_content = [
            f"[{r['doc_id']}] {r.get('subtopic', '')}: {r.get('content_preview', '')}"
            for r in results
        ]

        if is_correct:
            correct_count += 1
            failure_type = "correct"
        elif not doc_id or seed_type == "direct":
            no_doc_queries += 1
            failure_type = "direct_query_wrong"
        elif doc_id in retrieved_docs:
            reasoning_fail += 1
            failure_type = "bad_reasoning"
        else:
            retrieval_miss += 1
            failure_type = "bad_retrieval"

        entry = {
            "id": q["id"],
            "query": q["query"][:200],
            "gold_tool_call": gold,
            "predicted": pred,
            "correct": is_correct,
            "failure_type": failure_type,
            "seed_type": seed_type,
            "gold_doc_id": doc_id,
            "retrieved_docs": retrieved_docs,
            "doc_in_top5": doc_id in retrieved_docs if doc_id else None,
            "retrieved_context_preview": retrieved_content[:3],
        }
        diagnosis_results.append(entry)

    total = len(queries)
    wrong = total - correct_count

    print("\n" + "=" * 70)
    print(f"RAG FAILURE DIAGNOSIS — {split} set, basic strategy")
    print("=" * 70)
    print(f"Total queries:     {total}")
    print(f"Correct:           {correct_count} ({correct_count/total:.1%})")
    print(f"Wrong:             {wrong} ({wrong/total:.1%})")
    print(f"")
    print(f"WRONG BREAKDOWN:")
    print(f"  Bad retrieval:   {retrieval_miss:>4}  (gold doc NOT in top-5 chunks)")
    print(f"  Bad reasoning:   {reasoning_fail:>4}  (gold doc IN top-5, LLM got it wrong)")
    print(f"  Direct wrong:    {no_doc_queries:>4}  (no doc needed, LLM still wrong)")
    print(f"")

    if wrong > 0:
        print(f"Of the {wrong} wrong answers:")
        if retrieval_miss:
            print(f"  {retrieval_miss/wrong:.1%} are retrieval failures (encoder problem)")
        if reasoning_fail:
            print(f"  {reasoning_fail/wrong:.1%} are reasoning failures (LLM problem)")
        if no_doc_queries:
            print(f"  {no_doc_queries/wrong:.1%} are direct query failures (no RAG needed)")

    bad_retrieval_examples = [d for d in diagnosis_results if d["failure_type"] == "bad_retrieval"][:10]
    bad_reasoning_examples = [d for d in diagnosis_results if d["failure_type"] == "bad_reasoning"][:10]

    if bad_retrieval_examples:
        print(f"\nBAD RETRIEVAL EXAMPLES (gold doc not in top-5):")
        for ex in bad_retrieval_examples[:5]:
            print(f"  {ex['id']}: gold_doc={ex['gold_doc_id']} | retrieved={ex['retrieved_docs']}")
            print(f"    Q: {ex['query'][:100]}")
            print(f"    Gold: {ex['gold_tool_call']}")
            print(f"    Pred: {ex['predicted']}")

    if bad_reasoning_examples:
        print(f"\nBAD REASONING EXAMPLES (gold doc in top-5 but wrong answer):")
        for ex in bad_reasoning_examples[:5]:
            print(f"  {ex['id']}: gold_doc={ex['gold_doc_id']} | retrieved={ex['retrieved_docs']}")
            print(f"    Q: {ex['query'][:100]}")
            print(f"    Gold: {ex['gold_tool_call']}")
            print(f"    Pred: {ex['predicted']}")

    print("=" * 70)

    output_path = OUTPUT_DIR / f"diagnosis_{split}_basic.json"
    output_path.write_text(
        json.dumps(diagnosis_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Full diagnosis saved to {output_path}")


if __name__ == "__main__":
    main()
