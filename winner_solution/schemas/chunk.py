"""Pydantic schemas for document chunks in the winner RAG solution."""

from typing import Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Entity metadata extracted from chunk content via regex.

    Attributes:
        modules_mentioned: MASA module names found in content.
        protocols_mentioned: MASA-SEC protocol IDs found in content.
        crew_mentioned: Crew member names found in content.
        thresholds: Numeric threshold values with units found in content.
        document_type: Inferred document category based on doc_id range.
    """

    modules_mentioned: list[str]
    protocols_mentioned: list[str]
    crew_mentioned: list[str]
    thresholds: list[str]
    document_type: str


class Chunk(BaseModel):
    """A single document chunk with summary and metadata.

    Attributes:
        doc_id: Source document identifier (e.g., MASA-DOC-007).
        chunk_index: Zero-based index of this chunk within its document.
        topic: H1 header — document title.
        subtopic: H2 header — section name.
        keypoint: H3 header — subsection name, if present.
        content: Raw body text of the chunk.
        summary: LLM-generated summary of the chunk content.
        parent_id: UUID shared by all subchunks from the same oversized section.
            None for regular chunks.
        is_subchunk: True if this chunk was split from an oversized section.
        parent_summary: LLM-generated summary of the full oversized parent section.
            None for regular chunks.
        embedding_text: Constructed text used for encoder embedding.
            Regular: topic + subtopic + keypoint + content + summary.
            Subchunk: same + parent_summary appended.
        metadata: Extracted entity metadata.
    """

    doc_id: str
    chunk_index: int
    topic: str
    subtopic: str
    keypoint: Optional[str]
    content: str
    summary: str
    parent_id: Optional[str]
    is_subchunk: bool
    parent_summary: Optional[str]
    embedding_text: str
    metadata: ChunkMetadata
