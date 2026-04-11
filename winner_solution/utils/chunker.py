"""Markdown-aware document chunker for winner RAG solution.

Splits documents by H2/H3 headers, preserving topic/subtopic/keypoint
hierarchy. Each chunk includes the document title and section path
for richer embedding signal.
"""

import json
import re
from pathlib import Path


def parse_hierarchy(content: str, doc_title: str) -> list[dict]:
    """Parse markdown into sections with topic/subtopic/keypoint hierarchy.

    Args:
        content: Full markdown text.
        doc_title: Document title for the topic field.

    Returns:
        List of dicts with topic, subtopic, keypoint, body.
    """
    lines = content.split("\n")
    sections = []

    current_topic = doc_title
    current_subtopic = ""
    current_keypoint = None
    current_body_lines = []

    def flush():
        body = "\n".join(current_body_lines).strip()
        if body:
            sections.append({
                "topic": current_topic,
                "subtopic": current_subtopic or current_topic,
                "keypoint": current_keypoint,
                "body": body,
            })

    for line in lines:
        stripped = line.strip()

        h1 = re.match(r'^#\s+(.+)$', stripped)
        if h1:
            current_topic = h1.group(1).strip()
            continue

        h2 = re.match(r'^##\s+(.+)$', stripped)
        if h2:
            flush()
            current_subtopic = h2.group(1).strip()
            current_keypoint = None
            current_body_lines = []
            continue

        h3 = re.match(r'^###\s+(.+)$', stripped)
        if h3:
            flush()
            current_keypoint = h3.group(1).strip()
            current_body_lines = []
            continue

        current_body_lines.append(line)

    flush()
    return sections


def build_embedding_text(topic: str, subtopic: str, keypoint: str | None, content: str) -> str:
    """Build contextualized text for encoder input.

    Format: topic -> subtopic -> keypoint -> content.

    Args:
        topic: Document title.
        subtopic: Section header.
        keypoint: Subsection header or None.
        content: Chunk body text.

    Returns:
        Newline-joined contextualized text.
    """
    parts = [
        topic.strip(),
        subtopic.strip(),
        keypoint.strip() if keypoint else "",
        content.strip(),
    ]
    return "\n".join(p for p in parts if p)


def chunk_document(doc_id: str, content: str, doc_title: str, max_words: int = 300) -> list[dict]:
    """Chunk a single document using markdown hierarchy.

    Args:
        doc_id: Document identifier.
        content: Full markdown content.
        doc_title: Document title from index.
        max_words: Maximum words per chunk before splitting.

    Returns:
        List of chunk dicts with doc_id, chunk_index, topic, subtopic,
        keypoint, content, and embedding_text.
    """
    sections = parse_hierarchy(content, doc_title)
    chunks = []
    chunk_index = 0

    for section in sections:
        body = section["body"]
        words = body.split()

        if len(words) <= max_words:
            embedding_text = build_embedding_text(
                section["topic"], section["subtopic"], section["keypoint"], body,
            )
            chunks.append({
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "topic": section["topic"],
                "subtopic": section["subtopic"],
                "keypoint": section["keypoint"],
                "content": body,
                "embedding_text": embedding_text,
            })
            chunk_index += 1
        else:
            sentences = re.split(r'(?<=[.!?])\s+', body)
            current = []
            current_count = 0
            for sentence in sentences:
                s_count = len(sentence.split())
                if current_count + s_count > max_words and current:
                    sub_text = " ".join(current)
                    embedding_text = build_embedding_text(
                        section["topic"], section["subtopic"], section["keypoint"], sub_text,
                    )
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "topic": section["topic"],
                        "subtopic": section["subtopic"],
                        "keypoint": section["keypoint"],
                        "content": sub_text,
                        "embedding_text": embedding_text,
                    })
                    chunk_index += 1
                    current = []
                    current_count = 0
                current.append(sentence)
                current_count += s_count
            if current:
                sub_text = " ".join(current)
                embedding_text = build_embedding_text(
                    section["topic"], section["subtopic"], section["keypoint"], sub_text,
                )
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "topic": section["topic"],
                    "subtopic": section["subtopic"],
                    "keypoint": section["keypoint"],
                    "content": sub_text,
                    "embedding_text": embedding_text,
                })
                chunk_index += 1

    return chunks


def extract_title(content: str) -> str:
    """Extract H1 title from markdown content.

    Args:
        content: Full markdown text.

    Returns:
        H1 title or empty string.
    """
    match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def chunk_all_documents(docs_dir: Path, max_words: int = 300) -> list[dict]:
    """Chunk all documents with markdown-aware splitting.

    Discovers documents by scanning for MASA-DOC-* directories.
    Extracts the title from the H1 header of each document.

    Args:
        docs_dir: Path to directory containing MASA-DOC-XXX folders.
        max_words: Maximum words per chunk.

    Returns:
        List of all chunk dicts.
    """
    all_chunks = []

    for doc_path in sorted(docs_dir.glob("MASA-DOC-*/doc.md")):
        doc_id = doc_path.parent.name
        content = doc_path.read_text(encoding="utf-8")
        title = extract_title(content)
        chunks = chunk_document(doc_id, content, title, max_words)
        all_chunks.extend(chunks)

    return all_chunks
