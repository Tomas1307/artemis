"""Validator that checks RAG question numeric readings match the gold protocol_id per skeleton thresholds."""

import json
import re
from pathlib import Path

import yaml

from app.validators.base_validator import BaseValidator
from app.validators.validation_report import ValidationFinding, ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"
SKELETON_PATH = PROJECT_ROOT / "app" / "skeleton" / "skeleton.yaml"

METRIC_EXTRACTORS: list[tuple[str, str, re.Pattern]] = [
    ("airlock_equalization_rate", "kPa_per_minute", re.compile(
        r"(?:airlock|equalization)[^.]{0,60}?(\d+\.?\d*)\s*kPa\s*(?:per|/)\s*min", re.IGNORECASE
    )),
    ("docking_pressure_differential", "kPa", re.compile(
        r"(?:docking|differential)[^.]{0,40}?(\d+\.?\d*)\s*kPa", re.IGNORECASE
    )),
    ("scrubber_efficiency", "percent", re.compile(
        r"scrubber[^.]{0,40}?(\d+\.?\d*)\s*%", re.IGNORECASE
    )),
    ("water_recycling_output", "percent", re.compile(
        r"water\s*recycling[^.]{0,40}?(\d+\.?\d*)\s*%", re.IGNORECASE
    )),
    ("radiation", "mSv_per_hour", re.compile(r"(\d+\.?\d*)\s*mSv", re.IGNORECASE)),
    ("temperature", "celsius_per_minute", re.compile(
        r"(\d+\.?\d*)\s*(?:degrees?|°C|celsius)\s*(?:per|/)\s*minute", re.IGNORECASE
    )),
    ("co2_concentration", "percent", re.compile(
        r"CO2[^.]{0,40}?(\d+\.?\d*)\s*%", re.IGNORECASE
    )),
    ("hull_stress", "percent_rated_capacity", re.compile(
        r"(?:hull|stress)[^.]{0,40}?(\d+\.?\d*)\s*%", re.IGNORECASE
    )),
    ("voltage_fluctuation_percent", "V", re.compile(
        r"(?:voltage|power\s*bus)[^.]{0,60}?(\d+\.?\d*)\s*V\b", re.IGNORECASE
    )),
    ("power", "percent_rated_capacity", re.compile(
        r"power[^.]{0,60}?(\d+\.?\d*)\s*%", re.IGNORECASE
    )),
    ("oxygen", "percent", re.compile(
        r"oxygen[^.]{0,60}?(\d+\.?\d*)\s*%", re.IGNORECASE
    )),
    ("communication_uptime", "minutes", re.compile(
        r"(?:lost|contact|blackout|no\s*comm)[^.]{0,60}?(\d+\.?\d*)\s*minute", re.IGNORECASE
    )),
    ("pressure", "kPa", re.compile(r"(\d+\.?\d*)\s*kPa", re.IGNORECASE)),
]


class ThresholdProtocolCorrectnessValidator(BaseValidator):
    """Verify that the numeric reading in each RAG query triggers the gold protocol_id.

    Parses the skeleton security_protocols to build a metric-to-protocol lookup,
    then regex-extracts numeric values from queries and checks the gold protocol_id
    and scope match the expected protocol for that metric and value.
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "ThresholdProtocolCorrectnessValidator"

    def validate(self) -> ValidationReport:
        """Run threshold-to-protocol correctness checks on all RAG questions."""
        protocols = self._load_protocols()
        gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        test_gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))

        rag_questions = [
            q for q in gold + test_gold
            if q.get("seed_type") == "rag" and q.get("protocol_id")
        ]

        findings: list[ValidationFinding] = []
        passed = 0
        skipped = 0

        for question in rag_questions:
            qid = question["question_id"]
            query = question["query"]
            gold_protocol = question["protocol_id"]
            gold_tool_call = question["tool_call"]

            extracted = self._extract_metric_value(query)
            if extracted is None:
                skipped += 1
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="info",
                    rule="no_numeric_extracted",
                    message="Could not extract numeric reading from query.",
                    context={"query": query[:200]},
                ))
                continue

            metric, value = extracted
            expected_protocol = self._find_matching_protocol(protocols, metric, value)

            if expected_protocol is None:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="warning",
                    rule="no_protocol_match",
                    message=f"Extracted {metric}={value} but no protocol threshold matched.",
                    context={"metric": metric, "value": value, "gold_protocol": gold_protocol},
                ))
                continue

            expected_pid = expected_protocol["pid"]
            expected_scope = expected_protocol["scope"]

            if expected_pid != gold_protocol:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="error",
                    rule="protocol_mismatch",
                    message=f"Value {metric}={value} should trigger {expected_pid} but gold says {gold_protocol}.",
                    context={
                        "metric": metric,
                        "value": value,
                        "expected_protocol": expected_pid,
                        "gold_protocol": gold_protocol,
                        "query": query[:200],
                    },
                ))
                continue

            gold_scope = self._extract_scope(gold_tool_call)
            if gold_scope and gold_scope != expected_scope:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="error",
                    rule="scope_mismatch",
                    message=f"Protocol {gold_protocol} expects scope={expected_scope} but gold has scope={gold_scope}.",
                    context={"gold_scope": gold_scope, "expected_scope": expected_scope},
                ))
                continue

            passed += 1

        return ValidationReport(
            validator_name=self.name,
            total_checked=len(rag_questions),
            passed=passed,
            findings=findings,
        )

    def _load_protocols(self) -> list[dict]:
        """Load security protocols from skeleton and build threshold lookup."""
        with open(SKELETON_PATH, encoding="utf-8") as f:
            skeleton = yaml.safe_load(f)

        result = []
        for pid, proto in skeleton["security_protocols"].items():
            thresholds = proto.get("trigger_thresholds", {})
            result.append({
                "pid": pid,
                "metric": thresholds.get("metric"),
                "operator": thresholds.get("operator"),
                "value": thresholds.get("value"),
                "value_min": thresholds.get("value_min"),
                "value_max": thresholds.get("value_max"),
                "severity": proto["severity"],
                "scope": proto["scope"],
            })
        return result

    def _extract_metric_value(self, query: str) -> tuple[str, float] | None:
        """Regex-extract the first numeric sensor reading from a query.

        Returns:
            Tuple of (metric_name, numeric_value) or None if nothing matched.
        """
        for metric, _unit, pattern in METRIC_EXTRACTORS:
            match = pattern.search(query)
            if match:
                return metric, float(match.group(1))
        return None

    def _find_matching_protocol(self, protocols: list[dict], metric: str, value: float) -> dict | None:
        """Find the protocol whose threshold range contains the given value.

        For metrics with both a critical and a warning-band protocol (e.g., radiation
        has SEC-004 for >5.0 and SEC-012 for 1.1-5.0), the more specific match wins.
        """
        candidates = [p for p in protocols if p["metric"] == metric]
        for proto in candidates:
            if self._value_matches_threshold(proto, value):
                return proto
        return None

    def _value_matches_threshold(self, proto: dict, value: float) -> bool:
        """Check if a numeric value falls within a protocol's threshold range."""
        op = proto["operator"]
        if op == "less_than":
            return value < proto["value"]
        if op == "greater_than":
            return value > proto["value"]
        if op == "greater_than_or_equal":
            return value >= proto["value"]
        if op == "between":
            return proto["value_min"] <= value <= proto["value_max"]
        if op == "rate_of_change_greater_than":
            return value > proto["value"]
        return False

    def _extract_scope(self, tool_call: str) -> str | None:
        """Extract the scope parameter from a tool_call string."""
        match = re.search(r"scope='(\w+)'", tool_call)
        return match.group(1) if match else None
