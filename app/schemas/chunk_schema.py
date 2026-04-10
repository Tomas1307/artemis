"""Pydantic schemas for document chunks used across RAG pipeline components.

Adapted from konecto-kb-creator chunk structure: hierarchical
topic/subtopic/keypoint from markdown headers, with MASA-specific
entity metadata for retrieval filtering and validation.
"""

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Structured metadata extracted from chunk content.

    Captures MASA universe entities (modules, protocols, crew) and
    numeric thresholds for retrieval filtering and data validation.

    Attributes:
        modules_mentioned: Canonical module names found in chunk text.
        protocols_mentioned: MASA-SEC/OPS protocol IDs found in text.
        crew_mentioned: Crew member full names found in text.
        thresholds: Numeric values with units extracted from text.
        tools_relevant: Inferred tool names relevant to the chunk content.
        document_type: Classification from documentos_masa.json index.
    """

    modules_mentioned: list[str] = Field(default_factory=list)
    protocols_mentioned: list[str] = Field(default_factory=list)
    crew_mentioned: list[str] = Field(default_factory=list)
    thresholds: list[dict] = Field(default_factory=list)
    tools_relevant: list[str] = Field(default_factory=list)
    document_type: str = Field(default="")


class DocumentChunk(BaseModel):
    """A single chunk extracted from a MASA knowledge base document.

    Follows konecto-kb-creator hierarchy: H1 -> topic, H2 -> subtopic,
    H3 -> keypoint. Each chunk carries enough metadata to trace back
    to its source, enable faceted retrieval, and support validation.

    Attributes:
        doc_id: Parent document identifier (e.g., MASA-DOC-007).
        chunk_id: Globally unique chunk identifier (e.g., MASA-DOC-007-C03).
        topic: H1 header value, typically the document title.
        subtopic: H2 header value, the section name.
        keypoint: H3 header value if present, the subsection name.
        content: Raw text body of the chunk (without headers).
        summary: Auto-extracted summary from first 1-2 sentences.
        parent_summary: For subchunks, LLM summary of the unsplit parent section.
        parent_id: UUID shared by all subchunks from the same parent section.
        embedding_text: Prepared text for encoder input, combining hierarchy
            and content following the semantic search format.
        chunk_index: Zero-based position within parent document.
        is_subchunk: Whether this chunk was split from an oversized section.
        subchunk_part: Part number within subdivision (e.g., 1 of 3).
        subchunk_total: Total parts in the subdivision group.
        token_count: Estimated token count (words / 0.75 heuristic).
        char_count: Character count of raw content.
        word_count: Word count of raw content.
        metadata: Structured entity metadata extracted from content.
    """

    doc_id: str = Field(description="Parent document identifier.")
    chunk_id: str = Field(description="Globally unique chunk identifier.")
    topic: str = Field(description="H1 header: document title.")
    subtopic: str = Field(description="H2 header: section name.")
    keypoint: str | None = Field(default=None, description="H3 header: subsection name.")
    content: str = Field(description="Raw text body without headers.")
    summary: str = Field(default="", description="Auto-extracted 1-2 sentence summary.")
    parent_summary: str | None = Field(default=None, description="LLM parent section summary for subchunks.")
    parent_id: str | None = Field(default=None, description="UUID shared by all subchunks from same parent.")
    embedding_text: str = Field(description="Prepared text for encoder input.")
    chunk_index: int = Field(description="Zero-based index within parent document.")
    is_subchunk: bool = Field(default=False, description="Whether split from oversized section.")
    subchunk_part: int | None = Field(default=None, description="Part N within subdivision.")
    subchunk_total: int | None = Field(default=None, description="Total parts in subdivision.")
    token_count: int = Field(description="Estimated token count.")
    char_count: int = Field(description="Character count of raw content.")
    word_count: int = Field(description="Word count of raw content.")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata, description="Entity metadata.")


class ChunkCollection(BaseModel):
    """Collection of all document chunks with corpus-level metadata.

    Attributes:
        chunks: All chunks across all documents.
        total_documents: Number of documents processed.
        total_chunks: Total number of chunks generated.
    """

    chunks: list[DocumentChunk] = Field(default_factory=list)
    total_documents: int = Field(default=0)
    total_chunks: int = Field(default=0)

    def get_chunks_for_doc(self, doc_id: str) -> list[DocumentChunk]:
        """Return all chunks belonging to a specific document.

        Args:
            doc_id: Document identifier to filter by.

        Returns:
            List of chunks from the specified document.
        """
        return [c for c in self.chunks if c.doc_id == doc_id]

    def get_all_embedding_texts(self) -> list[str]:
        """Return all embedding texts in chunk order.

        Returns:
            List of contextualized texts ready for encoder input.
        """
        return [c.embedding_text for c in self.chunks]

    def get_unique_doc_ids(self) -> list[str]:
        """Return unique doc_ids in sorted order.

        Returns:
            Sorted list of unique document identifiers.
        """
        return sorted(set(c.doc_id for c in self.chunks))

    def get_chunks_by_module(self, module: str) -> list[DocumentChunk]:
        """Return chunks that mention a specific module.

        Args:
            module: Canonical module name (e.g., 'jaguar').

        Returns:
            List of chunks mentioning the module.
        """
        return [c for c in self.chunks if module in c.metadata.modules_mentioned]

    def get_chunks_by_protocol(self, protocol_id: str) -> list[DocumentChunk]:
        """Return chunks that reference a specific protocol.

        Args:
            protocol_id: Protocol ID (e.g., 'MASA-SEC-001').

        Returns:
            List of chunks referencing the protocol.
        """
        return [c for c in self.chunks if protocol_id in c.metadata.protocols_mentioned]
