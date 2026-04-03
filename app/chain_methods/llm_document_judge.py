import re

from loguru import logger

from app.llms.llm import BaseLLM
from app.pipelines.document_pipeline.schemas.generation_result import ValidationVerdict
from app.prompts.prompt_loader import prompt_loader


class LLMDocumentJudge:
    """Chain method that validates generated documents against skeleton facts.

    Uses a reasoning LLM (QwQ-32B) to verify that all required factual data
    from the skeleton appears in the generated document text.

    Args:
        llm: LLM provider instance for validation (should be the judge model).
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def validate(self, required_facts: list[str], document_text: str) -> ValidationVerdict:
        """Validate a generated document against a list of required facts.

        Args:
            required_facts: List of factual statements that must appear in the document.
                Each fact is a plain-language description of a skeleton value.
            document_text: The full generated document markdown text.

        Returns:
            ValidationVerdict with pass/fail status and details on missing facts.
        """
        system_message = prompt_loader.get_system_message_by_type("document_judge")
        template = prompt_loader.get_prompt_template_by_type("document_judge")
        config = prompt_loader.get_config_by_type("document_judge")

        facts_formatted = "\n".join(f"  - {fact}" for fact in required_facts)

        user_message = template.format(
            required_facts=facts_formatted,
            document_text=document_text,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        logger.info(f"Validating document with {len(required_facts)} required facts")
        raw_response = self._llm.generate(messages, **config)

        return self._parse_verdict(raw_response, len(required_facts))

    def _parse_verdict(self, raw_response: str, total_facts: int) -> ValidationVerdict:
        """Parse the judge model's response into a structured ValidationVerdict.

        Args:
            raw_response: Raw text output from the judge model.
            total_facts: Total number of facts that were checked.

        Returns:
            Parsed ValidationVerdict instance.
        """
        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", raw_response, re.IGNORECASE)
        passed = verdict_match.group(1).upper() == "PASS" if verdict_match else False

        present_match = re.search(r"FACTS PRESENT:\s*(\d+)", raw_response)
        facts_present = int(present_match.group(1)) if present_match else 0

        missing_section = re.search(
            r"MISSING FACTS.*?:\s*\n((?:\s*-\s*.+\n?)*)", raw_response, re.IGNORECASE
        )
        facts_missing = []
        if missing_section:
            facts_missing = [
                line.strip().lstrip("- ").strip()
                for line in missing_section.group(1).strip().split("\n")
                if line.strip() and line.strip() != "-"
            ]

        return ValidationVerdict(
            passed=passed,
            facts_checked=total_facts,
            facts_present=facts_present,
            facts_missing=facts_missing,
            judge_reasoning=raw_response,
        )
