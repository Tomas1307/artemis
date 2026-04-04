import re

from loguru import logger

from app.pipelines.question_pipeline.schemas.question_schema import QuestionSpec


_CANONICAL_PATTERN = re.compile(
    r"^(get_telemetry|get_crew_status|get_module_status|send_alert|send_message"
    r"|schedule_maintenance|activate_protocol|control_system|calculate_trajectory"
    r"|request_supply|no_action)"
    r"(\(.*\))?$"
)

_MIN_QUERY_WORDS = 5
_MAX_QUERY_WORDS = 60


class ValidateQuestionStep:
    """Step 4 — Validate generated QuestionSpec instances for correctness.

    Runs three checks per question:
    1. Tool call matches canonical format (regex + no spaces after commas).
    2. Query text meets length bounds (5–60 words).
    3. Query does not accidentally contain the tool call string (leakage).

    Questions failing any check are excluded from the final dataset and
    logged for inspection.

    No LLM is called in this step.
    """

    def execute(self, questions: list[QuestionSpec]) -> tuple[list[QuestionSpec], list[str]]:
        """Filter questions by validity rules.

        Args:
            questions: Generated questions from Step 3.

        Returns:
            Tuple of (valid QuestionSpec list, list of rejected question_ids with reasons).
        """
        valid: list[QuestionSpec] = []
        rejected: list[str] = []

        for q in questions:
            reason = self._validate(q)
            if reason:
                logger.warning(f"{q.question_id}: rejected — {reason}")
                rejected.append(f"{q.question_id}: {reason}")
            else:
                valid.append(q)

        logger.info(f"Validation complete: {len(valid)} valid, {len(rejected)} rejected")
        return valid, rejected

    def _validate(self, q: QuestionSpec) -> str:
        """Run all validation checks on a single question.

        Args:
            q: QuestionSpec to validate.

        Returns:
            Non-empty reason string if invalid, empty string if valid.
        """
        if not _CANONICAL_PATTERN.match(q.tool_call):
            return f"tool_call format invalid: {q.tool_call!r}"

        if ", " in q.tool_call:
            return f"tool_call contains space after comma: {q.tool_call!r}"

        word_count = len(q.query.split())
        if word_count < _MIN_QUERY_WORDS:
            return f"query too short ({word_count} words): {q.query!r}"
        if word_count > _MAX_QUERY_WORDS:
            return f"query too long ({word_count} words)"

        if any(token in q.query for token in ["get_telemetry(", "send_alert(", "no_action", "activate_protocol("]):
            return f"query contains tool call leakage: {q.query!r}"

        return ""
