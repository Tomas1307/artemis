"""LLM-based summary generator for MASA document chunks."""

from loguru import logger

from winner_solution.prompts.prompt_loader import prompt_loader
from winner_solution.utils.llm_manager import llm_manager


class SummaryGenerator:
    """Generates LLM summaries for document chunks and subchunks.

    Uses the local Qwen2.5-7B-Instruct model via LocalLLMManager and
    YAML prompts via PromptLoader. Two prompt types are used:
    - 'chunk_summarization': for regular chunks
    - 'subchunk_summarization': for oversized section subchunks,
      conditioned on the pre-generated parent summary

    The LLM must be loaded (llm_manager.load()) before calling any
    generate method.
    """

    def _build_keypoint_line(self, keypoint: str | None) -> str:
        """Format keypoint as a section path fragment.

        Args:
            keypoint: H3 header string, or None.

        Returns:
            ' > keypoint' string, or empty string if None.
        """
        if keypoint:
            return f" > {keypoint}"
        return ""

    def generate_chunk_summary(
        self,
        content: str,
        doc_id: str,
        topic: str,
        subtopic: str,
        keypoint: str | None,
    ) -> str:
        """Generate a summary for a regular (non-subchunk) chunk.

        Args:
            content: Raw body text of the chunk.
            doc_id: Source document identifier (e.g., MASA-DOC-007).
            topic: H1 document title.
            subtopic: H2 section name.
            keypoint: H3 subsection name, or None.

        Returns:
            LLM-generated summary string.

        Raises:
            RuntimeError: If the LLM is not loaded.
            Exception: On any LLM generation failure.
        """
        system_msg = prompt_loader.get_system_message_by_type("chunk_summarization")
        template = prompt_loader.get_prompt_template_by_type("chunk_summarization")
        if not template:
            raise ValueError("No prompt template found for 'chunk_summarization'.")

        user_prompt = template.format(
            doc_id=doc_id,
            topic=topic,
            subtopic=subtopic,
            keypoint_line=self._build_keypoint_line(keypoint),
            content=content,
        )
        logger.debug(f"Generating chunk summary for {doc_id} [{subtopic}]")
        return llm_manager.generate(system_msg, user_prompt, max_new_tokens=256)

    def generate_parent_summary(
        self,
        full_section_content: str,
        doc_id: str,
        topic: str,
        subtopic: str,
        keypoint: str | None,
    ) -> str:
        """Generate a summary for an oversized section before splitting.

        This summary is generated from the full section body and later
        passed to generate_subchunk_summary() to condition each subchunk.

        Args:
            full_section_content: Complete body text of the oversized section.
            doc_id: Source document identifier.
            topic: H1 document title.
            subtopic: H2 section name.
            keypoint: H3 subsection name, or None.

        Returns:
            LLM-generated parent summary string.

        Raises:
            RuntimeError: If the LLM is not loaded.
            Exception: On any LLM generation failure.
        """
        system_msg = prompt_loader.get_system_message_by_type("chunk_summarization")
        template = prompt_loader.get_prompt_template_by_type("chunk_summarization")
        if not template:
            raise ValueError("No prompt template found for 'chunk_summarization'.")

        user_prompt = template.format(
            doc_id=doc_id,
            topic=topic,
            subtopic=subtopic,
            keypoint_line=self._build_keypoint_line(keypoint),
            content=full_section_content,
        )
        logger.debug(f"Generating parent summary for {doc_id} [{subtopic}]")
        return llm_manager.generate(system_msg, user_prompt, max_new_tokens=256)

    def generate_subchunk_summary(
        self,
        content: str,
        doc_id: str,
        topic: str,
        subtopic: str,
        keypoint: str | None,
        parent_summary: str,
    ) -> str:
        """Generate a summary for a subchunk conditioned on its parent summary.

        The subchunk prompt references the parent_summary so the model can
        anchor the subchunk within the broader section context, maintaining
        semantic coherence with sibling subchunks.

        Args:
            content: Raw body text of the subchunk.
            doc_id: Source document identifier.
            topic: H1 document title.
            subtopic: H2 section name.
            keypoint: H3 subsection name, or None.
            parent_summary: Pre-generated summary of the full oversized section.

        Returns:
            LLM-generated subchunk summary string.

        Raises:
            RuntimeError: If the LLM is not loaded.
            Exception: On any LLM generation failure.
        """
        system_msg = prompt_loader.get_system_message_by_type("subchunk_summarization")
        template = prompt_loader.get_prompt_template_by_type("subchunk_summarization")
        if not template:
            raise ValueError("No prompt template found for 'subchunk_summarization'.")

        user_prompt = template.format(
            doc_id=doc_id,
            topic=topic,
            subtopic=subtopic,
            keypoint_line=self._build_keypoint_line(keypoint),
            parent_summary=parent_summary,
            content=content,
        )
        logger.debug(f"Generating subchunk summary for {doc_id} [{subtopic}]")
        return llm_manager.generate(system_msg, user_prompt, max_new_tokens=256)


summary_generator = SummaryGenerator()
