"""Validator that detects questions with similar scenarios but conflicting gold tool_calls."""

import json
import re
from pathlib import Path

from app.validators.base_validator import BaseValidator
from app.validators.validation_report import ValidationFinding, ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"


class CrossQuestionConsistencyValidator(BaseValidator):
    """Detect scenario conflicts: same canonical query → different tool_calls.

    Builds a normalized signature from each query (lowercased, punctuation
    stripped, numbers bucketized) and reports any signature with more than
    one distinct gold tool_call.
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "CrossQuestionConsistencyValidator"

    def validate(self) -> ValidationReport:
        """Group questions by signature and flag conflicts."""
        gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        test_gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))

        all_questions = gold + test_gold
        by_signature: dict[str, list[dict]] = {}

        for question in all_questions:
            sig = self._signature(question["query"])
            by_signature.setdefault(sig, []).append(question)

        findings: list[ValidationFinding] = []
        passed = 0
        groups_with_multiple = 0

        for sig, group in by_signature.items():
            if len(group) < 2:
                continue
            groups_with_multiple += 1
            unique_calls = set(q["tool_call"] for q in group)
            if len(unique_calls) > 1:
                findings.append(ValidationFinding(
                    severity="error",
                    rule="conflicting_gold_for_same_signature",
                    message=f"{len(group)} questions share signature but have {len(unique_calls)} distinct tool_calls.",
                    context={
                        "signature": sig[:120],
                        "question_ids": [q["question_id"] for q in group],
                        "tool_calls": sorted(unique_calls),
                    },
                ))
            else:
                passed += 1

        return ValidationReport(
            validator_name=self.name,
            total_checked=groups_with_multiple,
            passed=passed,
            findings=findings,
        )

    def _signature(self, query: str) -> str:
        """Build a normalized signature for grouping similar queries.

        Preserves numbers verbatim — only collapses whitespace, casing, and
        punctuation. Two queries with the same normalized text but different
        gold tool_calls indicate a real conflict.
        """
        s = query.lower()
        s = re.sub(r"[^\w\s.]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
