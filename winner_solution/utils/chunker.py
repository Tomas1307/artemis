"""Markdown-aware document chunker with on-the-fly LLM summarization.

Splits MASA technical documents by H2/H3 headers. Sections exceeding
max_words are split into subchunks and assigned a shared parent_id UUID.
The parent summary is generated first from the full section body, then
each subchunk summary is generated conditioned on the parent summary.

Regular chunk embedding_text:  topic + subtopic + keypoint + content + summary
Subchunk embedding_text:        topic + subtopic + keypoint + content + summary + parent_summary
"""

import re
import uuid
from pathlib import Path

from loguru import logger

from winner_solution.schemas.chunk import Chunk, ChunkMetadata
from winner_solution.utils.metadata_extractor import metadata_extractor
from winner_solution.utils.summarizer import summary_generator


def _extract_title(content: str) -> str:
    """Extract the H1 title from markdown content.

    Args:
        content: Full markdown document text.

    Returns:
        H1 header text, or empty string if not found.
    """
    match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_hierarchy(content: str, doc_title: str) -> list[dict]:
    """Parse markdown into sections preserving H2/H3 hierarchy.

    Accumulates body lines between headers. H1 updates the topic.
    H2 opens a new section. H3 opens a nested section within H2.

    Args:
        content: Full markdown document text.
        doc_title: Fallback topic if no H1 is found.

    Returns:
        List of dicts with keys: topic, subtopic, keypoint, body.
    """
    lines = content.split("\n")
    sections: list[dict] = []

    current_topic = doc_title
    current_subtopic = ""
    current_keypoint: str | None = None
    current_body_lines: list[str] = []

    def flush() -> None:
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


def _split_body(body: str, max_words: int) -> list[str]:
    """Split a body text into subchunk texts not exceeding max_words.

    Splits on sentence boundaries (after . ! ?) to avoid cutting mid-sentence.
    Any remaining sentences form the last subchunk even if under max_words.

    Args:
        body: Full section body text.
        max_words: Maximum words allowed per subchunk.

    Returns:
        List of subchunk text strings, each within max_words.
    """
    sentences = re.split(r'(?<=[.!?])\s+', body)
    subchunks: list[str] = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        s_count = len(sentence.split())
        if current_count + s_count > max_words and current:
            subchunks.append(" ".join(current))
            current = []
            current_count = 0
        current.append(sentence)
        current_count += s_count

    if current:
        subchunks.append(" ".join(current))

    return subchunks


def _build_embedding_text(
    topic: str,
    subtopic: str,
    keypoint: str | None,
    content: str,
    summary: str,
    parent_summary: str | None = None,
) -> str:
    """Construct the text used for encoder embedding.

    Format for regular chunks:  topic + subtopic + keypoint + content + summary
    Format for subchunks:       topic + subtopic + keypoint + content + summary + parent_summary

    Args:
        topic: Document title (H1).
        subtopic: Section name (H2).
        keypoint: Subsection name (H3), or None.
        content: Chunk body text.
        summary: LLM-generated chunk summary.
        parent_summary: Parent section summary for subchunks, or None.

    Returns:
        Newline-joined embedding text string.
    """
    parts = [
        topic.strip(),
        subtopic.strip(),
        keypoint.strip() if keypoint else "",
        content.strip(),
        summary.strip(),
    ]
    if parent_summary:
        parts.append(parent_summary.strip())
    return "\n".join(p for p in parts if p)


def chunk_document(
    doc_id: str,
    content: str,
    doc_title: str,
    max_words: int = 300,
) -> list[Chunk]:
    """Chunk a single document with LLM summaries generated on the fly.

    For each section:
    - If body <= max_words: generate chunk summary → create one Chunk.
    - If body > max_words: generate parent summary from full body first,
      split into subchunks, then generate each subchunk summary conditioned
      on parent_summary. All subchunks share a parent_id UUID.

    Args:
        doc_id: Document identifier (e.g., MASA-DOC-007).
        content: Full markdown document text.
        doc_title: Document title used as the topic field.
        max_words: Maximum words per chunk before splitting.

    Returns:
        List of Chunk objects for this document.
    """
    sections = _parse_hierarchy(content, doc_title)
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        body: str = section["body"]
        topic: str = section["topic"]
        subtopic: str = section["subtopic"]
        keypoint: str | None = section["keypoint"]
        words = body.split()

        if len(words) <= max_words:
            summary = summary_generator.generate_chunk_summary(
                content=body,
                doc_id=doc_id,
                topic=topic,
                subtopic=subtopic,
                keypoint=keypoint,
            )
            meta: ChunkMetadata = metadata_extractor.extract(body, doc_id)
            embedding_text = _build_embedding_text(topic, subtopic, keypoint, body, summary)
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_index=chunk_index,
                topic=topic,
                subtopic=subtopic,
                keypoint=keypoint,
                content=body,
                summary=summary,
                parent_id=None,
                is_subchunk=False,
                parent_summary=None,
                embedding_text=embedding_text,
                metadata=meta,
            ))
            chunk_index += 1

        else:
            logger.debug(
                f"{doc_id} [{subtopic}]: {len(words)} words — generating parent summary then splitting."
            )
            parent_summary = summary_generator.generate_parent_summary(
                full_section_content=body,
                doc_id=doc_id,
                topic=topic,
                subtopic=subtopic,
                keypoint=keypoint,
            )
            parent_id = str(uuid.uuid4())
            subchunk_texts = _split_body(body, max_words)

            for sub_text in subchunk_texts:
                sub_summary = summary_generator.generate_subchunk_summary(
                    content=sub_text,
                    doc_id=doc_id,
                    topic=topic,
                    subtopic=subtopic,
                    keypoint=keypoint,
                    parent_summary=parent_summary,
                )
                meta = metadata_extractor.extract(sub_text, doc_id)
                embedding_text = _build_embedding_text(
                    topic, subtopic, keypoint, sub_text, sub_summary, parent_summary
                )
                chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    topic=topic,
                    subtopic=subtopic,
                    keypoint=keypoint,
                    content=sub_text,
                    summary=sub_summary,
                    parent_id=parent_id,
                    is_subchunk=True,
                    parent_summary=parent_summary,
                    embedding_text=embedding_text,
                    metadata=meta,
                ))
                chunk_index += 1

    logger.info(f"{doc_id}: {chunk_index} chunks generated.")
    return chunks


def chunk_all_documents(docs_dir: Path, max_words: int = 300) -> list[Chunk]:
    """Chunk all MASA-DOC-* documents in docs_dir with LLM summaries.

    Discovers documents by scanning for MASA-DOC-*/doc.md paths.
    The LLM must be loaded (llm_manager.load()) before calling this.

    Args:
        docs_dir: Directory containing MASA-DOC-XXX subdirectories.
        max_words: Maximum words per chunk before splitting.

    Returns:
        List of all Chunk objects across all documents, sorted by doc_id
        and chunk_index.
    """
    all_chunks: list[Chunk] = []

    doc_paths = sorted(docs_dir.glob("MASA-DOC-*/doc.md"))
    logger.info(f"Found {len(doc_paths)} documents in {docs_dir}")

    for doc_path in doc_paths:
        doc_id = doc_path.parent.name
        content = doc_path.read_text(encoding="utf-8")
        title = _extract_title(content)
        chunks = chunk_document(doc_id, content, title, max_words)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks generated: {len(all_chunks)}")
    return all_chunks
