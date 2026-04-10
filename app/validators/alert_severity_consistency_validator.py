"""Validator that checks send_alert severity matches telemetry alert bands from the skeleton."""

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

METRIC_REGEX = {
    "radiation": re.compile(r"(\d+\.?\d*)\s*mSv", re.IGNORECASE),
    "pressure": re.compile(r"(\d+\.?\d*)\s*kPa", re.IGNORECASE),
    "oxygen": re.compile(r"oxygen[^.]{0,60}?(\d+\.?\d*)\s*%", re.IGNORECASE),
    "power": re.compile(r"power[^.]{0,60}?(\d+\.?\d*)\s*%", re.IGNORECASE),
    "hull_stress": re.compile(r"(?:hull|stress)[^.]{0,40}?(\d+\.?\d*)\s*%", re.IGNORECASE),
}

REASON_TO_METRIC = {
    "radiation_spike": "radiation",
    "pressure_drop": "pressure",
    "oxygen_leak": "oxygen",
    "power_fluctuation": "power",
    "structural_damage": "hull_stress",
    "abnormal_temperature": "temperature",
    "system_failure": None,
    "communication_loss": None,
}


class AlertSeverityConsistencyValidator(BaseValidator):
    """Verify send_alert severity matches the telemetry alert bands from the skeleton.

    Loads per-module telemetry bands from skeleton.yaml and checks that
    each send_alert question's severity parameter is consistent with the
    numeric reading extracted from the query text.
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "AlertSeverityConsistencyValidator"

    def validate(self) -> ValidationReport:
        """Check severity consistency for all send_alert RAG questions."""
        telemetry_bands = self._load_telemetry_bands()
        protocol_severities = self._load_protocol_severities()

        gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        test_gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))

        alert_questions = [
            q for q in gold + test_gold
            if q.get("tool_name") == "send_alert" and q.get("seed_type") == "rag"
        ]

        findings: list[ValidationFinding] = []
        passed = 0
        skipped = 0

        for question in alert_questions:
            qid = question["question_id"]
            query = question["query"]
            tool_call = question["tool_call"]
            protocol_id = question.get("protocol_id")

            gold_severity = self._extract_param(tool_call, "severity")
            gold_reason = self._extract_param(tool_call, "reason")
            gold_module = self._extract_param(tool_call, "module")

            if not gold_severity or not gold_reason:
                skipped += 1
                continue

            metric = REASON_TO_METRIC.get(gold_reason)
            if not metric or metric not in METRIC_REGEX:
                skipped += 1
                continue

            match = METRIC_REGEX[metric].search(query)
            if not match:
                skipped += 1
                continue

            value = float(match.group(1))
            expected_severity = self._get_telemetry_severity(
                telemetry_bands, gold_module or "condor", metric, value
            )

            if expected_severity is None:
                skipped += 1
                continue

            proto_severity = protocol_severities.get(protocol_id) if protocol_id else None

            if gold_severity == expected_severity:
                passed += 1
            elif proto_severity and gold_severity == proto_severity:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="warning",
                    rule="severity_uses_protocol_not_telemetry",
                    message=(
                        f"Gold severity='{gold_severity}' matches protocol {protocol_id} "
                        f"severity, but telemetry band says '{expected_severity}'."
                    ),
                    context={
                        "metric": metric,
                        "value": value,
                        "gold_severity": gold_severity,
                        "telemetry_severity": expected_severity,
                        "protocol_severity": proto_severity,
                        "protocol_id": protocol_id,
                    },
                ))
            else:
                findings.append(ValidationFinding(
                    question_id=qid,
                    severity="error",
                    rule="severity_mismatch",
                    message=(
                        f"Gold severity='{gold_severity}' matches neither telemetry "
                        f"band ('{expected_severity}') nor protocol severity "
                        f"('{proto_severity}')."
                    ),
                    context={
                        "metric": metric,
                        "value": value,
                        "gold_severity": gold_severity,
                        "telemetry_severity": expected_severity,
                        "protocol_severity": proto_severity,
                    },
                ))

        return ValidationReport(
            validator_name=self.name,
            total_checked=len(alert_questions) - skipped,
            passed=passed,
            findings=findings,
        )

    def _load_telemetry_bands(self) -> dict:
        """Load per-module telemetry alert bands from skeleton."""
        with open(SKELETON_PATH, encoding="utf-8") as f:
            skeleton = yaml.safe_load(f)
        return {
            module_name: module_data["telemetry"]
            for module_name, module_data in skeleton["modules"].items()
            if "telemetry" in module_data
        }

    def _load_protocol_severities(self) -> dict[str, str]:
        """Load protocol_id to severity mapping from skeleton."""
        with open(SKELETON_PATH, encoding="utf-8") as f:
            skeleton = yaml.safe_load(f)
        return {
            pid: proto["severity"]
            for pid, proto in skeleton["security_protocols"].items()
        }

    def _get_telemetry_severity(
        self, bands: dict, module: str, metric: str, value: float
    ) -> str | None:
        """Determine the severity level for a metric value using telemetry bands.

        Returns:
            One of 'low', 'medium', 'high', 'critical', or None if not determinable.
        """
        module_bands = bands.get(module, {}).get(metric)
        if not module_bands:
            return None

        critical_above = module_bands.get("critical_above")
        critical_below = module_bands.get("critical_below")
        high_max = module_bands.get("high_alert_max")
        high_min = module_bands.get("high_alert_min")
        medium_max = module_bands.get("medium_alert_max")
        medium_min = module_bands.get("medium_alert_min")
        low_max = module_bands.get("low_alert_max")
        low_min = module_bands.get("low_alert_min")

        if critical_above is not None and value >= critical_above:
            return "critical"
        if critical_below is not None and value <= critical_below:
            return "critical"

        if high_max is not None and value > medium_max and value <= high_max:
            return "high"
        if high_min is not None and value >= high_min and value < medium_min:
            return "high"

        if medium_max is not None and value > low_max and value <= medium_max:
            return "medium"
        if medium_min is not None and value >= medium_min and value < low_min:
            return "medium"

        if low_max is not None and value > module_bands.get("normal_max", 0) and value <= low_max:
            return "low"
        if low_min is not None and value >= low_min and value < module_bands.get("normal_min", 999):
            return "low"

        return None

    def _extract_param(self, tool_call: str, param: str) -> str | None:
        """Extract a named parameter value from a tool_call string."""
        match = re.search(rf"{param}='(\w+)'", tool_call)
        return match.group(1) if match else None
