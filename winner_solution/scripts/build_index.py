"""Build FAISS index using fine-tuned encoder — winner solution.

Usage:
    python -m app.winner_solution.scripts.build_index --device cuda
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_PATH = PROJECT_ROOT / "artifacts" / "winner" / "chunks.json"
ENCODER_PATH = PROJECT_ROOT / "artifacts" / "winner" / "encoder" / "finetuned_encoder"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "winner"


def main() -> None:
    """Chunk documents, embed with fine-tuned encoder, build FAISS index."""
    args = sys.argv[1:]
    device = "cuda"
    if "--device" in args:
        device = args[args.index("--device") + 1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_PATH}. "
            "Run generate_chunks.py first."
        )
    logger.info(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    logger.info(f"Total chunks: {len(chunks)}")

    logger.info(f"Loading fine-tuned encoder from {ENCODER_PATH}...")
    model = SentenceTransformer(str(ENCODER_PATH), device=device)

    logger.info("Embedding chunks...")
    texts = [c["embedding_text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    np.save(str(OUTPUT_DIR / "embeddings.npy"), embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(OUTPUT_DIR / "faiss_index.bin"))

    logger.info(f"Index built: {index.ntotal} vectors")
    logger.info(f"Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
