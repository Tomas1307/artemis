from loguru import logger

from app.chain_methods.llm_document_reviewer import LLMDocumentReviewer
from app.pipelines.document_pipeline.schemas.generation_result import ValidationVerdict


class ValidateDocumentStep:
    """Validates and corrects a generated document using parallel LangChain review chains.

    Runs factual accuracy and hallucination detection in parallel, then applies
    corrections if needed. Loops up to max_correction_cycles for persistent issues.
    Falls back to saving with needs_review flag if issues cannot be resolved.

    Args:
        reviewer: LLMDocumentReviewer chain method instance.
    """

    def __init__(self, reviewer: LLMDocumentReviewer) -> None:
        self._reviewer = reviewer

    def execute(
        self,
        doc_id: str,
        document_text: str,
        skeleton_context: str,
        reference_index: str,
        is_noise: bool = False,
    ) -> tuple[str, ValidationVerdict]:
        """Validate and potentially correct a document.

        Args:
            doc_id: Document identifier for logging.
            document_text: Generated markdown text to validate.
            skeleton_context: Formatted skeleton data for comparison.
            reference_index: Protocol/crew reference index for corrections.
            is_noise: If True, skip validation (noise docs have no skeleton data).

        Returns:
            Tuple of (final_document_text, ValidationVerdict).
        """
        if is_noise:
            logger.info(f"{doc_id}: noise document — skipping validation")
            return document_text, ValidationVerdict(
                passed=True,
                facts_checked=0,
                facts_present=0,
                facts_missing=[],
                judge_reasoning="Noise document — no validation required.",
            )

        logger.info(f"{doc_id}: starting review-correct cycle")

        corrected_text, summary = self._reviewer.review_and_correct(
            document_text=document_text,
            skeleton_context=skeleton_context,
            reference_index=reference_index,
        )

        status = summary["status"]
        passed = status in ("clean", "corrected")
        remaining = summary.get("remaining_issues", [])

        logger.info(
            f"{doc_id}: review complete — status={status}, "
            f"cycles={summary['cycles_used']}, "
            f"total_issues={summary['total_issues_found']}, "
            f"remaining={len(remaining)}"
        )

        verdict = ValidationVerdict(
            passed=passed,
            facts_checked=summary["total_issues_found"],
            facts_present=summary["total_issues_found"] - len(remaining),
            facts_missing=remaining,
            judge_reasoning=f"status={status}, cycles={summary['cycles_used']}",
        )

        return corrected_text, verdict
