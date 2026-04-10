"""Validator that checks direct (non-RAG) questions are deterministic from query text alone."""

import json
import re
from pathlib import Path

from app.validators.base_validator import BaseValidator
from app.validators.validation_report import ValidationFinding, ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"

MODULE_SYNONYMS = {
    "condor": ["condor", "cóndor", "command", "control"],
    "quetzal": ["quetzal", "lab", "science"],
    "jaguar": ["jaguar", "life support"],
    "colibri": ["colibri", "colibrí", "comms", "communication", "navigation", "antenna"],
    "vicuna": ["vicuna", "vicuña", "storage", "cargo", "docking"],
    "tucan": ["tucan", "tucán", "crew quarters", "habitat", "dormitor"],
}

METRIC_SYNONYMS = {
    "temperature": ["temperature", "temp", "thermal", "°c", "degrees"],
    "pressure": ["pressure", "kpa", "psi"],
    "oxygen": ["oxygen", "o2", "o₂"],
    "radiation": ["radiation", "msv", "rad"],
    "humidity": ["humidity", "moisture", "humid"],
    "power": ["power", "voltage", "kw", "watt", "consumption", "load"],
}

INFO_SYNONYMS = {
    "health": ["health", "vital", "biometric", "medical"],
    "location": ["location", "where", "position", "located"],
    "current_activity": ["activity", "doing", "task", "shift"],
    "schedule": ["schedule", "timeline", "agenda", "rotation"],
}


class DirectDeterminismValidator(BaseValidator):
    """Validate direct questions can have all params inferred from the query.

    For each direct (non-RAG) question, this validator checks that every
    parameter value in the gold tool_call has at least a synonym mentioned
    in the query text. Failures indicate ambiguous questions.
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "DirectDeterminismValidator"

    def validate(self) -> ValidationReport:
        """Audit direct questions for parameter inferability."""
        gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        test_gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))

        all_questions = gold + test_gold
        direct_questions = [
            q for q in all_questions
            if q.get("seed_type") == "direct" and q.get("tool_call") != "no_action"
        ]

        findings: list[ValidationFinding] = []
        passed = 0

        for question in direct_questions:
            qid = question["question_id"]
            query_lower = question["query"].lower()
            tool_call = question["tool_call"]

            params = re.findall(r"(\w+)=(?:'([^']*)'|(\d+))", tool_call)
            missing = []
            for pname, str_val, int_val in params:
                val = str_val if str_val else int_val
                if not self._param_inferable(pname, val, query_lower):
                    missing.append(f"{pname}={val}")

            if missing:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="warning",
                    rule="param_not_in_query",
                    message=f"Params not clearly inferable from query: {', '.join(missing)}",
                    context={"tool_call": tool_call, "missing": missing},
                ))
            else:
                passed += 1

        return ValidationReport(
            validator_name=self.name,
            total_checked=len(direct_questions),
            passed=passed,
            findings=findings,
        )

    def _param_inferable(self, pname: str, value: str, query: str) -> bool:
        """Check if a param value can be inferred from the query text."""
        if pname == "module":
            return any(syn in query for syn in MODULE_SYNONYMS.get(value, [value]))
        if pname == "metric":
            return any(syn in query for syn in METRIC_SYNONYMS.get(value, [value]))
        if pname == "info":
            return any(syn in query for syn in INFO_SYNONYMS.get(value, [value]))
        if pname == "timeframe_hours":
            number_words = {"1": ["1 hour", "one hour", "last hour"],
                            "6": ["6 hour", "six hour"],
                            "12": ["12 hour", "twelve hour"],
                            "24": ["24 hour", "day", "twenty-four"]}
            return str(value) in query or any(w in query for w in number_words.get(str(value), []))
        return value.lower() in query
