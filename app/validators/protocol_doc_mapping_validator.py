"""Validator that checks every MASA-SEC protocol_id used in the dataset is grep-able in some document."""

import json
import re
from pathlib import Path

from app.validators.base_validator import BaseValidator
from app.validators.validation_report import ValidationFinding, ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"
DOCS_INDEX_PATH = PROJECT_ROOT / "proyecto_artemis" / "base_conocimiento" / "documentos_masa.json"


class ProtocolDocMappingValidator(BaseValidator):
    """Verify every protocol_id used in tool_calls is found in the corpus.

    For each unique protocol_id used in the dataset, this validator confirms
    that the protocol is mentioned in at least one document, and that the
    questions citing it consistently point to the same set of documents.
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "ProtocolDocMappingValidator"

    def validate(self) -> ValidationReport:
        """Audit protocol_id usage across all questions and documents."""
        gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        test_gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))
        docs_index = json.loads(DOCS_INDEX_PATH.read_text(encoding="utf-8"))

        doc_cache: dict[str, str] = {}
        for doc_id, info in docs_index.items():
            doc_path = Path(info["file_path"])
            if doc_path.exists():
                doc_cache[doc_id] = doc_path.read_text(encoding="utf-8").lower()

        protocol_to_docs: dict[str, set[str]] = {}
        protocol_questions: dict[str, list[str]] = {}

        for question in gold + test_gold:
            tool_call = question.get("tool_call", "")
            doc_id = question.get("doc_id")
            m = re.search(r"protocol_id='(MASA-SEC-\d+)'", tool_call)
            if not m:
                continue
            protocol_id = m.group(1)
            protocol_questions.setdefault(protocol_id, []).append(question["question_id"])
            if doc_id:
                protocol_to_docs.setdefault(protocol_id, set()).add(doc_id)

        findings: list[ValidationFinding] = []
        passed = 0

        for protocol_id, citing_docs in sorted(protocol_to_docs.items()):
            protocol_lower = protocol_id.lower()
            docs_containing = [d for d, text in doc_cache.items() if protocol_lower in text]

            if not docs_containing:
                findings.append(ValidationFinding(
                    severity="error",
                    rule="protocol_in_no_doc",
                    message=f"{protocol_id} is used in {len(protocol_questions[protocol_id])} questions but appears in zero documents.",
                    context={"protocol": protocol_id, "question_count": len(protocol_questions[protocol_id])},
                ))
                continue

            inconsistent = citing_docs - set(docs_containing)
            if inconsistent:
                findings.append(ValidationFinding(
                    severity="error",
                    rule="protocol_cited_to_wrong_doc",
                    message=f"{protocol_id} cited to docs {sorted(inconsistent)} but the protocol does not appear in those docs (only in {sorted(docs_containing)}).",
                    context={"protocol": protocol_id, "wrong_docs": sorted(inconsistent), "actual_docs": sorted(docs_containing)},
                ))
            else:
                passed += 1

        return ValidationReport(
            validator_name=self.name,
            total_checked=len(protocol_to_docs),
            passed=passed,
            findings=findings,
        )
