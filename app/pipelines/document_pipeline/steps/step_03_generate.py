from loguru import logger

from app.chain_methods.llm_document_writer import LLMDocumentWriter
from app.pipelines.document_pipeline.schemas.document_spec import DocumentSpec
from app.utils.retry import retry_on_api_error


class GenerateDocumentStep:
    """Calls the LLM to generate a document from its spec and skeleton context.

    Delegates to the LLMDocumentWriter chain method. Handles both API-level
    retries (transient errors) and content-level retries (empty output).

    Args:
        writer: LLMDocumentWriter chain method instance.
        max_retries: Maximum content-level retries per document (empty output).
        max_api_retries: Maximum API-level retries per call (transient errors).
    """

    def __init__(
        self, writer: LLMDocumentWriter, max_retries: int = 2, max_api_retries: int = 3
    ) -> None:
        self._writer = writer
        self._max_retries = max_retries
        self._max_api_retries = max_api_retries

    def execute(self, spec: DocumentSpec, skeleton_context: str) -> tuple[str, int]:
        """Generate a document, retrying on API errors and empty output.

        Args:
            spec: Document specification defining what to generate.
            skeleton_context: Formatted skeleton data to inject into the prompt.

        Returns:
            Tuple of (generated_markdown_string, number_of_attempts).

        Raises:
            RuntimeError: If all retry attempts produce empty output or fail.
        """
        for attempt in range(1, self._max_retries + 1):
            logger.info(f"{spec.doc_id}: generation attempt {attempt}/{self._max_retries}")

            try:
                result = retry_on_api_error(
                    lambda: self._writer.generate_document(
                        doc_id=spec.doc_id,
                        title=spec.title,
                        doc_type=spec.type,
                        target_words=spec.target_words,
                        sections=spec.sections,
                        skeleton_context=skeleton_context,
                    ),
                    max_retries=self._max_api_retries,
                )
            except Exception as exc:
                logger.error(f"{spec.doc_id}: attempt {attempt} API error: {exc}")
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"{spec.doc_id}: generation failed after {self._max_retries} attempts — {exc}"
                    ) from exc
                continue

            if result and len(result.strip()) > 100:
                return result, attempt

            logger.warning(f"{spec.doc_id}: attempt {attempt} produced empty/short output")

        raise RuntimeError(
            f"{spec.doc_id}: all {self._max_retries} generation attempts produced empty output"
        )
