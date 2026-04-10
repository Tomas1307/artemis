"""Markdown-aware document chunker for MASA knowledge base documents.

Follows konecto-kb-creator hierarchy: H1 -> topic, H2 -> subtopic,
H3 -> keypoint. Uses LLM-generated summaries for each chunk and
parent summaries for subdivided sections. Extracts MASA entities
as structured metadata for retrieval filtering and validation.
"""

import json
import re
import uuid
from pathlib import Path

from loguru import logger

from app.chain_methods.llm_chunk_summarizer import LLMChunkSummarizer
from app.llms.llm import BaseLLM
from app.schemas.chunk_schema import ChunkCollection, ChunkMetadata, DocumentChunk
from app.utils.entity_extractor import (
    extract_crew,
    extract_modules,
    extract_protocols,
    extract_relevant_tools,
    extract_thresholds,
)


class DocumentChunker:
    """Chunks markdown documents by header hierarchy with LLM summaries.

    Adapted from konecto-kb-creator ContentProcessor + ChunkSubdivider:
    splits on markdown headers maintaining topic/subtopic/keypoint hierarchy,
    generates LLM summaries for each chunk (and parent summaries for subchunks),
    extracts MASA entities (modules, protocols, crew, thresholds) as metadata,
    and builds embedding text following the semantic search format:
    parent_summary -> topic -> subtopic -> keypoint -> summary -> content.

    Attributes:
        _max_tokens: Maximum estimated tokens per chunk content.
        _overlap_tokens: Overlap tokens when splitting oversized sections.
        _summarizer: LLM chain method for generating chunk summaries.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_tokens: int = 384,
        overlap_tokens: int = 50,
    ) -> None:
        """Initialize the chunker with an LLM provider and token limits.

        Args:
            llm: LLM provider instance for summary generation.
            max_tokens: Maximum tokens per chunk content. Default 384 leaves
                headroom within bge-small-en-v1.5's 512 token context window
                after prepending the topic/subtopic/keypoint/summary hierarchy.
            overlap_tokens: Overlap tokens between consecutive sub-chunks
                when splitting oversized sections at sentence boundaries.
        """
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._summarizer = LLMChunkSummarizer(llm)

    def chunk_document(
        self,
        doc_id: str,
        content: str,
        doc_title: str = "",
        document_type: str = "",
    ) -> list[DocumentChunk]:
        """Chunk a single markdown document into retrieval-ready segments.

        Parses the markdown header hierarchy (H1/H2/H3), splits into
        sections, generates LLM summaries, extracts entities, and builds
        embedding text. Oversized sections are subdivided with a shared
        parent_id UUID and LLM-generated parent_summary.

        Args:
            doc_id: Document identifier (e.g., MASA-DOC-007).
            content: Full markdown content of the document.
            doc_title: Document title from index. Falls back to H1 header.
            document_type: Classification from documentos_masa.json.

        Returns:
            List of DocumentChunk objects for this document.
        """
        if not doc_title:
            doc_title = self._extract_h1_title(content)

        sections = self._parse_hierarchy(content, doc_title)
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section in sections:
            topic = section["topic"]
            subtopic = section["subtopic"]
            keypoint = section["keypoint"]
            body = section["body"].strip()

            if not body:
                continue

            token_count = self._estimate_tokens(body)

            if token_count <= self._max_tokens:
                summary = self._summarizer.summarize_chunk(topic, subtopic, keypoint, body)
                metadata = self._extract_metadata(body, document_type)
                embedding_text = self._build_embedding_text(
                    topic, subtopic, keypoint, summary, body,
                )
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-C{chunk_index:02d}",
                    topic=topic,
                    subtopic=subtopic,
                    keypoint=keypoint,
                    content=body,
                    summary=summary,
                    embedding_text=embedding_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    char_count=len(body),
                    word_count=len(body.split()),
                    metadata=metadata,
                ))
                chunk_index += 1
            else:
                sub_texts = self._split_oversized(body)
                parent_id = str(uuid.uuid4())
                parent_summary = self._summarizer.summarize_parent(
                    topic, subtopic, keypoint, body, num_parts=len(sub_texts),
                )
                logger.debug(
                    f"{doc_id} section '{subtopic}' oversized ({token_count} tok) "
                    f"-> {len(sub_texts)} subchunks, parent_id={parent_id[:8]}"
                )

                for part_num, sub_text in enumerate(sub_texts, start=1):
                    sub_clean = sub_text.strip()
                    if not sub_clean:
                        continue
                    sub_summary = self._summarizer.summarize_chunk(
                        topic, subtopic, keypoint, sub_clean,
                    )
                    metadata = self._extract_metadata(sub_clean, document_type)
                    embedding_text = self._build_embedding_text(
                        topic, subtopic, keypoint, sub_summary, sub_clean,
                        parent_summary=parent_summary,
                    )
                    chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}-C{chunk_index:02d}",
                        topic=topic,
                        subtopic=subtopic,
                        keypoint=keypoint,
                        content=sub_clean,
                        summary=sub_summary,
                        parent_summary=parent_summary,
                        parent_id=parent_id,
                        embedding_text=embedding_text,
                        chunk_index=chunk_index,
                        is_subchunk=True,
                        subchunk_part=part_num,
                        subchunk_total=len(sub_texts),
                        token_count=self._estimate_tokens(sub_clean),
                        char_count=len(sub_clean),
                        word_count=len(sub_clean.split()),
                        metadata=metadata,
                    ))
                    chunk_index += 1

        return chunks

    def chunk_all_documents(self, docs_dir: Path, docs_index: dict) -> ChunkCollection:
        """Chunk all documents in the MASA knowledge base.

        Args:
            docs_dir: Path to the base_conocimiento directory.
            docs_index: Document index dict from documentos_masa.json,
                mapping doc_id to metadata including 'title' and 'type'.

        Returns:
            ChunkCollection containing all chunks from all documents.
        """
        all_chunks: list[DocumentChunk] = []
        doc_ids = sorted(docs_index.keys())

        for doc_id in doc_ids:
            doc_path = docs_dir / doc_id / "doc.md"
            if not doc_path.exists():
                logger.warning(f"Document file not found: {doc_path}")
                continue

            content = doc_path.read_text(encoding="utf-8")
            doc_title = docs_index[doc_id].get("title", "")
            document_type = docs_index[doc_id].get("type", "")
            chunks = self.chunk_document(doc_id, content, doc_title, document_type)
            all_chunks.extend(chunks)

            modules_in_doc = set()
            for c in chunks:
                modules_in_doc.update(c.metadata.modules_mentioned)
            logger.info(
                f"{doc_id}: {len(chunks)} chunks, "
                f"{sum(c.token_count for c in chunks)} est. tokens, "
                f"modules={sorted(modules_in_doc)}"
            )

        collection = ChunkCollection(
            chunks=all_chunks,
            total_documents=len(doc_ids),
            total_chunks=len(all_chunks),
        )
        logger.info(f"Total: {collection.total_documents} docs -> {collection.total_chunks} chunks")
        return collection

    def save_chunks(self, collection: ChunkCollection, output_path: Path) -> None:
        """Persist the chunk collection to a JSON file.

        Args:
            collection: ChunkCollection to save.
            output_path: Destination file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(collection.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Chunks saved to {output_path}")

    def load_chunks(self, input_path: Path) -> ChunkCollection:
        """Load a previously saved chunk collection from JSON.

        Args:
            input_path: Path to the chunks JSON file.

        Returns:
            Deserialized ChunkCollection.
        """
        data = json.loads(input_path.read_text(encoding="utf-8"))
        return ChunkCollection(**data)

    def _parse_hierarchy(self, content: str, doc_title: str) -> list[dict]:
        """Parse markdown into sections with topic/subtopic/keypoint hierarchy.

        Maps markdown headers following konecto-kb-creator convention:
        H1 -> topic (document title), H2 -> subtopic, H3 -> keypoint.

        Args:
            content: Full markdown text.
            doc_title: Document title (used as topic for all sections).

        Returns:
            List of dicts with keys: topic, subtopic, keypoint, body.
        """
        lines = content.split("\n")
        sections: list[dict] = []

        current_topic = doc_title
        current_subtopic = ""
        current_keypoint: str | None = None
        current_body_lines: list[str] = []

        def flush_section() -> None:
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

            h1_match = re.match(r'^#\s+(.+)$', stripped)
            if h1_match:
                current_topic = h1_match.group(1).strip()
                continue

            h2_match = re.match(r'^##\s+(.+)$', stripped)
            if h2_match:
                flush_section()
                current_subtopic = h2_match.group(1).strip()
                current_keypoint = None
                current_body_lines = []
                continue

            h3_match = re.match(r'^###\s+(.+)$', stripped)
            if h3_match:
                flush_section()
                current_keypoint = h3_match.group(1).strip()
                current_body_lines = []
                continue

            current_body_lines.append(line)

        flush_section()
        return sections

    def _extract_metadata(self, text: str, document_type: str) -> ChunkMetadata:
        """Extract structured MASA entity metadata from chunk text.

        Args:
            text: Chunk body text.
            document_type: Document classification from index.

        Returns:
            Populated ChunkMetadata instance.
        """
        return ChunkMetadata(
            modules_mentioned=extract_modules(text),
            protocols_mentioned=extract_protocols(text),
            crew_mentioned=extract_crew(text),
            thresholds=extract_thresholds(text),
            tools_relevant=extract_relevant_tools(text),
            document_type=document_type,
        )

    def _build_embedding_text(
        self,
        topic: str,
        subtopic: str,
        keypoint: str | None,
        summary: str,
        content: str,
        parent_summary: str | None = None,
    ) -> str:
        """Build contextualized text for encoder input.

        Follows konecto-kb-creator semantic search format:
        parent_summary -> topic -> subtopic -> keypoint -> summary -> content.

        Args:
            topic: H1 document title.
            subtopic: H2 section header.
            keypoint: H3 subsection header (optional).
            summary: LLM-generated chunk summary.
            content: Raw chunk text.
            parent_summary: LLM-generated parent summary for subchunks.

        Returns:
            Newline-joined contextualized text.
        """
        parts = [
            parent_summary.strip() if parent_summary else "",
            topic.strip() if topic else "",
            subtopic.strip() if subtopic else "",
            keypoint.strip() if keypoint else "",
            summary.strip() if summary else "",
            content.strip() if content else "",
        ]
        return "\n".join(part for part in parts if part)

    def _split_oversized(self, text: str) -> list[str]:
        """Split oversized text at sentence boundaries with overlap.

        Args:
            text: Text exceeding max_tokens.

        Returns:
            List of sub-chunk texts, each within max_tokens.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self._max_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens > self._overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _extract_h1_title(self, content: str) -> str:
        """Extract the H1 title from markdown content.

        Args:
            content: Full markdown text.

        Returns:
            H1 title text, or empty string if not found.
        """
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length.

        Uses words / 0.75 heuristic, which is conservative for English text
        with BERT-style tokenizers (~1.3 tokens per word).

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        return int(len(text.split()) / 0.75)
