"""Simple fixed-size text chunker for baseline RAG solution."""

import json
from pathlib import Path


def chunk_document(doc_id: str, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Split document text into fixed-size overlapping chunks.

    Args:
        doc_id: Document identifier.
        text: Full document text.
        chunk_size: Maximum words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk dicts with doc_id, chunk_index, and content.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "content": chunk_text,
        })
        chunk_index += 1
        start += chunk_size - overlap

    return chunks


def chunk_all_documents(docs_dir: Path, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Chunk all documents in the knowledge base.

    Discovers documents by scanning for MASA-DOC-* directories.

    Args:
        docs_dir: Path to directory containing MASA-DOC-XXX folders.
        chunk_size: Maximum words per chunk.
        overlap: Overlap words between chunks.

    Returns:
        List of all chunk dicts across all documents.
    """
    all_chunks = []

    for doc_path in sorted(docs_dir.glob("MASA-DOC-*/doc.md")):
        doc_id = doc_path.parent.name
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_document(doc_id, text, chunk_size, overlap)
        all_chunks.extend(chunks)

    return all_chunks
