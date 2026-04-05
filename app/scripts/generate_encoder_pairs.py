"""Generate consultas_centro_control.json — (query, doc_id) pairs for encoder training.

Covers all 60 documents in the MASA knowledge base:
- Docs 007-012 (protocol groups): Sample from existing RAG questions in gold_standard.json
- All other docs: Generate queries via LLM from document content

Usage:
    python app/scripts/generate_encoder_pairs.py
    python app/scripts/generate_encoder_pairs.py --per-doc 15
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.llms.llm_nvidia import NvidiaLLM
from app.prompts.prompt_loader import prompt_loader

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "consultas_centro_control.json"
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
DOCS_JSON = PROJECT_ROOT / "proyecto_artemis" / "base_conocimiento" / "documentos_masa.json"

RAG_DOC_IDS = {"MASA-DOC-007", "MASA-DOC-008", "MASA-DOC-009", "MASA-DOC-010", "MASA-DOC-011", "MASA-DOC-012"}


def sample_rag_pairs(per_doc: int) -> list[dict]:
    """Sample (query, doc_id) pairs from existing RAG questions for protocol docs.

    Args:
        per_doc: Number of pairs to sample per document.

    Returns:
        List of dicts with query and doc_id.
    """
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    by_doc: dict[str, list[str]] = {}
    for q in gold:
        doc_id = q.get("doc_id")
        if doc_id and doc_id in RAG_DOC_IDS:
            by_doc.setdefault(doc_id, []).append(q["query"])

    pairs = []
    for doc_id, queries in sorted(by_doc.items()):
        sampled = random.sample(queries, min(per_doc, len(queries)))
        for query in sampled:
            pairs.append({"query": query, "doc_id": doc_id})
        logger.info(f"{doc_id}: sampled {len(sampled)} from {len(queries)} existing RAG queries")

    return pairs


def generate_llm_pairs(llm: NvidiaLLM, doc_id: str, doc_info: dict, per_doc: int) -> list[dict]:
    """Generate (query, doc_id) pairs via LLM for a single document.

    Args:
        llm: LLM provider instance.
        doc_id: Document identifier.
        doc_info: Document metadata from documentos_masa.json.
        per_doc: Number of queries to generate.

    Returns:
        List of dicts with query and doc_id.
    """
    doc_path = Path(doc_info["file_path"])
    if not doc_path.exists():
        logger.warning(f"{doc_id}: file not found at {doc_path}")
        return []

    doc_content = doc_path.read_text(encoding="utf-8")[:2000]

    system_msg = prompt_loader.get_system_message_by_type("encoder_query_generator")
    template = prompt_loader.get_prompt_template_by_type("encoder_query_generator")
    config = prompt_loader.get_config_by_type("encoder_query_generator")

    prompt = template.format(
        doc_id=doc_id,
        doc_title=doc_info.get("title", "Unknown"),
        doc_type=doc_info.get("type", "Unknown"),
        doc_content=doc_content,
        count=per_doc,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        try:
            response = llm.generate(messages, **config)
            lines = [line.strip().strip("-").strip("0123456789.").strip() for line in response.strip().split("\n")]
            queries = [line for line in lines if len(line.split()) >= 6]

            pairs = [{"query": q, "doc_id": doc_id} for q in queries[:per_doc]]
            logger.info(f"{doc_id}: generated {len(pairs)} queries (attempt {attempt + 1})")
            return pairs
        except Exception as exc:
            logger.warning(f"{doc_id}: attempt {attempt + 1} failed — {exc}")

    logger.error(f"{doc_id}: all attempts failed")
    return []


def main() -> None:
    """Generate consultas_centro_control.json covering all 60 documents."""
    args = sys.argv[1:]
    per_doc = 15

    if "--per-doc" in args:
        idx = args.index("--per-doc")
        per_doc = int(args[idx + 1])

    random.seed(42)

    logger.info(f"Generating encoder training pairs: {per_doc} per doc")

    all_pairs: list[dict] = []

    logger.info("Phase 1: Sampling from existing RAG questions (docs 007-012)")
    rag_pairs = sample_rag_pairs(per_doc)
    all_pairs.extend(rag_pairs)
    logger.info(f"RAG pairs: {len(rag_pairs)}")

    logger.info("Phase 2: Generating LLM queries for remaining docs")
    docs = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
    llm = NvidiaLLM()

    non_rag_docs = {did: info for did, info in docs.items() if did not in RAG_DOC_IDS}
    logger.info(f"Non-RAG docs to process: {len(non_rag_docs)}")

    for i, (doc_id, doc_info) in enumerate(sorted(non_rag_docs.items()), 1):
        logger.info(f"[{i}/{len(non_rag_docs)}] Processing {doc_id}: {doc_info.get('title', '?')[:50]}")
        pairs = generate_llm_pairs(llm, doc_id, doc_info, per_doc)
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(all_pairs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    doc_coverage = len({p["doc_id"] for p in all_pairs})
    logger.info(f"Done: {len(all_pairs)} pairs covering {doc_coverage}/60 docs")
    logger.info(f"Saved to {OUTPUT_PATH}")

    print(f"\n=== ENCODER TRAINING PAIRS ===")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Doc coverage: {doc_coverage}/60")
    by_doc = {}
    for p in all_pairs:
        by_doc[p["doc_id"]] = by_doc.get(p["doc_id"], 0) + 1
    for did, count in sorted(by_doc.items()):
        print(f"    {did}: {count}")


if __name__ == "__main__":
    main()
