"""Build FAISS index from chunked documents — baseline solution.

Usage:
    python -m app.baseline_solution.scripts.build_index --device cuda
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

from baseline_solution.utils.chunker import chunk_all_documents
from baseline_solution.utils.embedder import embed_texts, load_encoder

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "baseline"


def main() -> None:
    """Chunk documents, embed, and build FAISS index."""
    args = sys.argv[1:]
    device = "cuda"
    if "--device" in args:
        device = args[args.index("--device") + 1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Chunking documents...")
    print(f"Total chunks: {len(chunks)}")

    chunks_path = OUTPUT_DIR / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Loading encoder...")
    model = load_encoder(device=device)

    print("Embedding chunks...")
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(model, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(str(OUTPUT_DIR / "embeddings.npy"), embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(OUTPUT_DIR / "faiss_index.bin"))

    print(f"Index built: {index.ntotal} vectors")
    print(f"Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
