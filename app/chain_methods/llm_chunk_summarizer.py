"""Chain method for generating chunk and parent summaries via LLM."""

from loguru import logger

from app.llms.llm import BaseLLM
from app.prompts.prompt_loader import prompt_loader


class LLMChunkSummarizer:
    """Generates concise summaries for document chunks using an LLM.

    Produces two types of summaries following konecto-kb-creator patterns:
    - Chunk summary: 1-3 sentences capturing key facts of a single chunk.
    - Parent summary: 2-3 sentences providing broader context for sub-chunks
      split from an oversized section.

    Summaries are optimized for semantic search embedding — they prioritize
    specific values, protocol IDs, module names, and crew references over
    general descriptions.

    Args:
        llm: LLM provider instance for summary generation.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def summarize_chunk(
        self,
        topic: str,
        subtopic: str,
        keypoint: str | None,
        content: str,
    ) -> str:
        """Generate a summary for a single document chunk.

        Args:
            topic: H1 document title.
            subtopic: H2 section header.
            keypoint: H3 subsection header, or None.
            content: Raw chunk text body.

        Returns:
            1-3 sentence summary string.
        """
        system_message = prompt_loader.get_system_message_by_type("chunk_summarizer")
        template = prompt_loader.get_prompt_template_by_type("chunk_summarizer")
        config = prompt_loader.get_config_by_type("chunk_summarizer")

        keypoint_line = f"Subsection: {keypoint}" if keypoint else ""

        user_message = template.format(
            topic=topic,
            subtopic=subtopic,
            keypoint_line=keypoint_line,
            content=content[:3000],
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        summary = self._llm.generate(messages, **config)
        return summary.strip()

    def summarize_parent(
        self,
        topic: str,
        subtopic: str,
        keypoint: str | None,
        content: str,
        num_parts: int,
    ) -> str:
        """Generate an overview summary for an oversized section before splitting.

        This parent summary is prepended to each sub-chunk's embedding text,
        giving the encoder broader context about the section's purpose.

        Args:
            topic: H1 document title.
            subtopic: H2 section header.
            keypoint: H3 subsection header, or None.
            content: Full oversized section text.
            num_parts: Number of sub-chunks the section will be split into.

        Returns:
            2-3 sentence overview summary string.
        """
        system_message = prompt_loader.get_system_message_by_type("parent_summarizer")
        template = prompt_loader.get_prompt_template_by_type("parent_summarizer")
        config = prompt_loader.get_config_by_type("parent_summarizer")

        keypoint_line = f"Subsection: {keypoint}" if keypoint else ""

        user_message = template.format(
            topic=topic,
            subtopic=subtopic,
            keypoint_line=keypoint_line,
            content=content[:6000],
            num_parts=num_parts,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        summary = self._llm.generate(messages, **config)
        return summary.strip()

    def summarize_chunks_batch(
        self,
        chunks_data: list[dict],
    ) -> list[str]:
        """Generate summaries for multiple chunks sequentially.

        Args:
            chunks_data: List of dicts with keys: topic, subtopic, keypoint, content.

        Returns:
            List of summary strings in the same order as input.
        """
        summaries: list[str] = []
        total = len(chunks_data)

        for i, chunk in enumerate(chunks_data):
            logger.info(f"Summarizing chunk {i + 1}/{total}")
            summary = self.summarize_chunk(
                topic=chunk["topic"],
                subtopic=chunk["subtopic"],
                keypoint=chunk.get("keypoint"),
                content=chunk["content"],
            )
            summaries.append(summary)

        logger.info(f"Generated {len(summaries)} chunk summaries")
        return summaries
