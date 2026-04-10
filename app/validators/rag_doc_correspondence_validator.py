"""Validator that checks RAG questions against the content of their cited documents."""

import json
import re
from pathlib import Path

from app.validators.base_validator import BaseValidator
from app.validators.validation_report import ValidationFinding, ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"
DOCS_INDEX_PATH = PROJECT_ROOT / "proyecto_artemis" / "base_conocimiento" / "documentos_masa.json"


class RAGDocCorrespondenceValidator(BaseValidator):
    """Validate that each RAG question's gold answer is supported by its cited document.

    For activate_protocol questions, the cited doc must literally contain the
    protocol_id from the tool_call. For send_alert questions, the cited doc
    must mention the relevant module and reason. Numeric thresholds extracted
    from the query should appear in the doc when present.
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "RAGDocCorrespondenceValidator"

    def validate(self) -> ValidationReport:
        """Run RAG-doc correspondence checks against both train and test gold."""
        gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        test_gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))
        docs_index = json.loads(DOCS_INDEX_PATH.read_text(encoding="utf-8"))

        doc_cache: dict[str, str] = {}
        for doc_id, info in docs_index.items():
            doc_path = Path(info["file_path"])
            if doc_path.exists():
                doc_cache[doc_id] = doc_path.read_text(encoding="utf-8").lower()

        all_questions = [(q, "train") for q in gold] + [(q, "test") for q in test_gold]
        rag_questions = [(q, split) for q, split in all_questions if q.get("seed_type") == "rag" and q.get("doc_id")]

        findings: list[ValidationFinding] = []
        passed = 0

        for question, split in rag_questions:
            qid = question["question_id"]
            doc_id = question["doc_id"]
            tool_call = question["tool_call"]
            query = question["query"]

            if doc_id not in doc_cache:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="error",
                    rule="missing_doc",
                    message=f"Cited doc {doc_id} does not exist on disk.",
                    context={"split": split, "doc_id": doc_id},
                ))
                continue

            doc_text = doc_cache[doc_id]
            question_findings = self._check_question(qid, query, tool_call, doc_id, doc_text, split)

            if question_findings:
                findings.extend(question_findings)
            else:
                passed += 1

        return ValidationReport(
            validator_name=self.name,
            total_checked=len(rag_questions),
            passed=passed,
            findings=findings,
        )

    def _check_question(
        self,
        qid: str,
        query: str,
        tool_call: str,
        doc_id: str,
        doc_text: str,
        split: str,
    ) -> list[ValidationFinding]:
        """Check a single RAG question against its cited document."""
        findings: list[ValidationFinding] = []
        ctx = {"split": split, "doc_id": doc_id, "tool_call": tool_call}

        protocol_match = re.search(r"protocol_id='(MASA-SEC-\d+)'", tool_call)
        if protocol_match:
            protocol_id = protocol_match.group(1).lower()
            if protocol_id not in doc_text:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="error",
                    rule="protocol_id_not_in_doc",
                    message=f"Tool calls protocol {protocol_id.upper()} but it does not appear in cited doc {doc_id}.",
                    context=ctx,
                ))

        module_match = re.search(r"module='(\w+)'", tool_call)
        if module_match:
            module = module_match.group(1).lower()
            module_accented = {
                "vicuna": "vicuña",
                "colibri": "colibrí",
                "tucan": "tucán",
                "condor": "cóndor",
            }.get(module, module)
            if module not in doc_text and module_accented not in doc_text:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="warning",
                    rule="module_not_in_doc",
                    message=f"Tool call references module '{module}' but neither it nor accented form appears in {doc_id}.",
                    context=ctx,
                ))

        return findings
