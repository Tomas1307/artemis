import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import settings
from app.prompts.prompt_loader import prompt_loader
from app.utils.retry import retry_on_api_error


class LLMDocumentReviewer:
    """Chain method that reviews and corrects generated MASA documents.

    Uses LangChain's RunnableParallel to run factual accuracy and hallucination
    detection chains concurrently, then applies corrections if issues are found.
    Includes retry logic for API errors and a review-correct loop for persistent
    hallucinations.

    Args:
        model: NVIDIA model identifier for review calls.
        api_key: API key for the review model.
        temperature: Sampling temperature for review calls.
        max_correction_cycles: Maximum review-correct iterations before fallback.
        max_api_retries: Maximum retries per API call on transient errors.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_correction_cycles: int = 2,
        max_api_retries: int = 3,
    ) -> None:
        self._max_correction_cycles = max_correction_cycles
        self._max_api_retries = max_api_retries

        self._llm = ChatOpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=api_key or settings.NVIDIA_API_KEY,
            model=model or settings.NVIDIA_MODEL,
            temperature=temperature,
            max_tokens=2048,
        )
        self._correction_llm = ChatOpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=api_key or settings.NVIDIA_API_KEY,
            model=model or settings.NVIDIA_MODEL,
            temperature=0.1,
            max_tokens=16384,
        )
        self._parser = StrOutputParser()
        self._review_chain = self._build_review_chain()
        self._correction_chain = self._build_correction_chain()

    def _build_review_chain(self) -> RunnableParallel:
        """Build the parallel review chain with factual accuracy and hallucination detection.

        Returns:
            RunnableParallel that executes both review chains concurrently.
        """
        factual_system = prompt_loader.get_system_message_by_type("factual_accuracy")
        factual_template = prompt_loader.get_prompt_template_by_type("factual_accuracy")
        factual_prompt = ChatPromptTemplate.from_messages([
            ("system", factual_system),
            ("human", factual_template),
        ])

        hallucination_system = prompt_loader.get_system_message_by_type("hallucination_detection")
        hallucination_template = prompt_loader.get_prompt_template_by_type("hallucination_detection")
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", hallucination_system),
            ("human", hallucination_template),
        ])

        return RunnableParallel(
            factual=factual_prompt | self._llm | self._parser,
            hallucination=hallucination_prompt | self._llm | self._parser,
        )

    def _build_correction_chain(self):
        """Build the correction chain for fixing identified issues.

        Returns:
            LangChain chain that takes issues + document and returns corrected text.
        """
        correction_system = prompt_loader.get_system_message_by_type("correction")
        correction_template = prompt_loader.get_prompt_template_by_type("correction")
        correction_prompt = ChatPromptTemplate.from_messages([
            ("system", correction_system),
            ("human", correction_template),
        ])

        return correction_prompt | self._correction_llm | self._parser

    def review_and_correct(
        self,
        document_text: str,
        skeleton_context: str,
        reference_index: str,
    ) -> tuple[str, dict]:
        """Review a document and correct issues, with retry loop for persistent problems.

        Runs factual accuracy and hallucination detection in parallel. If issues
        are found, applies correction and re-reviews. Loops up to max_correction_cycles.
        If issues persist after all cycles, returns the best version with a needs_review flag.

        Args:
            document_text: Generated markdown document to review.
            skeleton_context: Formatted skeleton data the document should contain.
            reference_index: Protocol/crew/procedure reference index.

        Returns:
            Tuple of (final_document_text, review_summary_dict).
            summary contains: issues_found, issues, corrected, cycles_used, status.
            status is 'clean', 'corrected', or 'needs_review'.
        """
        current_text = document_text
        all_issues_history = []

        for cycle in range(1, self._max_correction_cycles + 1):
            logger.info(f"Review cycle {cycle}/{self._max_correction_cycles}")

            issues = self._run_review(current_text, skeleton_context)
            all_issues_history.append(issues)

            if not issues:
                logger.info(f"Cycle {cycle}: no issues found — document is clean")
                return current_text, self._build_summary(
                    all_issues_history, cycles_used=cycle, status="clean"
                )

            logger.info(f"Cycle {cycle}: found {len(issues)} issues — correcting")

            corrected = self._run_correction(current_text, issues, reference_index)
            if corrected:
                current_text = corrected
            else:
                logger.warning(f"Cycle {cycle}: correction failed — keeping previous version")

        final_issues = self._run_review(current_text, skeleton_context)
        all_issues_history.append(final_issues)

        if not final_issues:
            logger.info("Final review after corrections: clean")
            return current_text, self._build_summary(
                all_issues_history, cycles_used=self._max_correction_cycles, status="corrected"
            )

        logger.warning(
            f"Issues persist after {self._max_correction_cycles} correction cycles. "
            f"Remaining: {final_issues[:3]}... Saving with needs_review flag."
        )
        return current_text, self._build_summary(
            all_issues_history,
            cycles_used=self._max_correction_cycles,
            status="needs_review",
            remaining_issues=final_issues,
        )

    def _run_review(self, document_text: str, skeleton_context: str) -> list[str]:
        """Run the parallel review chain with retry on API errors.

        Args:
            document_text: Document to review.
            skeleton_context: Skeleton data for comparison.

        Returns:
            List of issue strings. Empty if no issues found.
        """
        review_input = {
            "skeleton_context": skeleton_context,
            "document_text": document_text,
        }

        try:
            review_results = retry_on_api_error(
                lambda: self._review_chain.invoke(review_input),
                max_retries=self._max_api_retries,
            )
        except Exception as exc:
            logger.error(f"Review chain failed after retries: {exc}")
            return []

        factual_result = review_results.get("factual", "")
        hallucination_result = review_results.get("hallucination", "")

        logger.debug(f"Factual: {factual_result[:150]}")
        logger.debug(f"Hallucination: {hallucination_result[:150]}")

        return self._extract_issues(factual_result, hallucination_result)

    def _run_correction(
        self, document_text: str, issues: list[str], reference_index: str
    ) -> str | None:
        """Run the correction chain with retry on API errors.

        Args:
            document_text: Document to correct.
            issues: List of issues to fix.
            reference_index: Correct protocol/crew names for reference.

        Returns:
            Corrected document text, or None if correction failed.
        """
        issues_formatted = "\n".join(f"- {issue}" for issue in issues)

        try:
            corrected = retry_on_api_error(
                lambda: self._correction_chain.invoke({
                    "issues": issues_formatted,
                    "reference_index": reference_index,
                    "document_text": document_text,
                }),
                max_retries=self._max_api_retries,
            )
            if corrected and len(corrected.strip()) > 100:
                return corrected
            logger.warning("Correction returned empty/short output")
            return None
        except Exception as exc:
            logger.error(f"Correction chain failed after retries: {exc}")
            return None

    def _build_summary(
        self,
        issues_history: list[list[str]],
        cycles_used: int,
        status: str,
        remaining_issues: list[str] | None = None,
    ) -> dict:
        """Build a structured review summary.

        Args:
            issues_history: List of issue lists from each review cycle.
            cycles_used: Number of review-correct cycles executed.
            status: Final status ('clean', 'corrected', 'needs_review').
            remaining_issues: Issues still present after all cycles.

        Returns:
            Summary dictionary with review metadata.
        """
        return {
            "status": status,
            "cycles_used": cycles_used,
            "issues_history": issues_history,
            "remaining_issues": remaining_issues or [],
            "total_issues_found": sum(len(i) for i in issues_history),
        }

    def _extract_issues(self, factual_result: str, hallucination_result: str) -> list[str]:
        """Parse review results to extract a flat list of issues.

        Args:
            factual_result: Raw output from the factual accuracy reviewer.
            hallucination_result: Raw output from the hallucination detector.

        Returns:
            List of issue description strings. Empty if no issues found.
        """
        issues = []

        numbers_ok = "numbers_ok: true" in factual_result.lower()
        protocols_ok = "protocols_ok: true" in factual_result.lower()
        crew_ok = "crew_ok: true" in factual_result.lower()

        if not numbers_ok:
            issues.extend(self._parse_section(factual_result, "NUMBERS_ISSUES", "PROTOCOLS", "WRONG NUMBER"))

        if not protocols_ok:
            issues.extend(self._parse_section(factual_result, "PROTOCOLS_ISSUES", "CREW", "WRONG PROTOCOL"))

        if not crew_ok:
            issues.extend(self._parse_section(factual_result, "CREW_ISSUES", None, "INVENTED CREW"))

        hallucination_found = "hallucinations_found: true" in hallucination_result.lower()
        if hallucination_found:
            issues.extend(self._parse_section(hallucination_result, "HALLUCINATIONS:", None, "HALLUCINATION"))

        return issues

    def _parse_section(
        self, text: str, start_marker: str, end_marker: str | None, prefix: str
    ) -> list[str]:
        """Parse a section of review output to extract individual issues.

        Args:
            text: Full review output text.
            start_marker: Marker indicating the start of the issues section.
            end_marker: Marker indicating the end. None means end of text.
            prefix: Label prefix for each extracted issue.

        Returns:
            List of prefixed issue strings.
        """
        pattern = rf"{start_marker}\s*\[?(.*?)\]?"
        if end_marker:
            pattern += rf"\s*(?:{end_marker}|$)"
        else:
            pattern += r"\s*$"

        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        issues = []
        for line in match.group(1).strip().split("\n"):
            cleaned = line.strip().lstrip("- ").strip()
            if cleaned and cleaned.lower() != "none" and len(cleaned) > 3:
                issues.append(f"{prefix}: {cleaned}")

        return issues
