"""Generate enriched chunks with LLM summaries for winner RAG solution.

Loads all MASA-DOC-* documents, splits them with markdown-aware chunking,
generates LLM summaries (chunk and subchunk) using local Qwen2.5-7B-Instruct,
extracts entity metadata, and saves the result to artifacts/winner/chunks.json.

This script must be run BEFORE finetune_encoder.py and build_index.py, which
both load chunks.json rather than re-running the (expensive) chunking pipeline.

Usage:
    python -m winner_solution.scripts.generate_chunks
    python -m winner_solution.scripts.generate_chunks --device cuda --max-words 300
    python -m winner_solution.scripts.generate_chunks --model Qwen/Qwen2.5-7B-Instruct
"""

import json
import sys
from pathlib import Path

from loguru import logger

from winner_solution.utils.chunker import chunk_all_documents
from winner_solution.utils.llm_manager import llm_manager

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "winner"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_WORDS = 300


def main() -> None:
    """Run the full chunk generation pipeline and save chunks.json."""
    args = sys.argv[1:]
    model_name = DEFAULT_MODEL
    device = DEFAULT_DEVICE
    max_words = DEFAULT_MAX_WORDS

    if "--model" in args:
        model_name = args[args.index("--model") + 1]
    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--max-words" in args:
        max_words = int(args[args.index("--max-words") + 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {model_name} | Device: {device} | Max words: {max_words}")
    llm_manager.load(model_name, device)

    logger.info(f"Chunking documents from {DOCS_DIR}...")
    chunks = chunk_all_documents(DOCS_DIR, max_words=max_words)
    logger.info(f"Total chunks: {len(chunks)}")

    subchunk_count = sum(1 for c in chunks if c.is_subchunk)
    logger.info(f"Regular chunks: {len(chunks) - subchunk_count} | Subchunks: {subchunk_count}")

    chunks_path = OUTPUT_DIR / "chunks.json"
    chunks_data = [c.model_dump() for c in chunks]
    chunks_path.write_text(
        json.dumps(chunks_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Chunks saved to {chunks_path}")


if __name__ == "__main__":
    main()
