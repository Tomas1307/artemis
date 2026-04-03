from loguru import logger

from app.llms.llm import BaseLLM
from app.prompts.prompt_loader import prompt_loader


class LLMDocumentWriter:
    """Chain method that generates MASA technical documents via LLM.

    Takes a document specification and extracted skeleton context, builds a
    prompt using the appropriate template, and returns generated markdown.

    Args:
        llm: LLM provider instance for document generation.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def generate_document(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        target_words: int,
        sections: list[str],
        skeleton_context: str,
    ) -> str:
        """Generate a complete MASA technical document.

        Args:
            doc_id: Document identifier (e.g., MASA-DOC-001).
            title: Document title.
            doc_type: Document category (module_manual, protocol_group, etc.).
            target_words: Target word count for the document.
            sections: Ordered list of section headings to include.
            skeleton_context: Formatted string of skeleton data to inject.

        Returns:
            Generated markdown document as a string.
        """
        is_noise = doc_type == "noise"
        prompt_type = "noise_writer" if is_noise else "document_writer"

        system_message = prompt_loader.get_system_message_by_type(prompt_type)
        template = prompt_loader.get_prompt_template_by_type(prompt_type)
        config = prompt_loader.get_config_by_type(prompt_type)

        sections_formatted = "\n".join(f"  - {s}" for s in sections)

        user_message = template.format(
            doc_id=doc_id,
            title=title,
            doc_type=doc_type,
            target_words=target_words,
            sections_formatted=sections_formatted,
            skeleton_context=skeleton_context if not is_noise else "",
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        logger.info(f"Generating {doc_id}: {title} ({doc_type}, ~{target_words} words)")
        result = self._llm.generate(messages, **config)
        logger.info(f"Generated {doc_id}: {len(result.split())} words")

        return result
