"""Validator that checks structural consistency between train and test datasets."""

import json
import re
from collections import Counter
from pathlib import Path

from app.validators.base_validator import BaseValidator
from app.validators.validation_report import ValidationFinding, ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_GOLD_PATH = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"


class TestTrainConsistencyValidator(BaseValidator):
    """Verify structural parity and no data leakage between train and test sets.

    Checks:
    - Schema parity (same keys in each question object)
    - Tool distribution coverage (every tool in train appears in test)
    - Seed type distribution proportionality
    - No near-duplicate queries across train/test (leakage detection)
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "TestTrainConsistencyValidator"

    def validate(self) -> ValidationReport:
        """Run all train-test consistency checks."""
        train = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        test = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))

        findings: list[ValidationFinding] = []
        checks_passed = 0
        total_checks = 0

        total_checks += 1
        schema_findings = self._check_schema_parity(train, test)
        if schema_findings:
            findings.extend(schema_findings)
        else:
            checks_passed += 1

        total_checks += 1
        tool_findings = self._check_tool_coverage(train, test)
        if tool_findings:
            findings.extend(tool_findings)
        else:
            checks_passed += 1

        total_checks += 1
        seed_findings = self._check_seed_type_distribution(train, test)
        if seed_findings:
            findings.extend(seed_findings)
        else:
            checks_passed += 1

        total_checks += 1
        leakage_findings = self._check_leakage(train, test)
        if leakage_findings:
            findings.extend(leakage_findings)
        else:
            checks_passed += 1

        return ValidationReport(
            validator_name=self.name,
            total_checked=total_checks,
            passed=checks_passed,
            findings=findings,
        )

    def _check_schema_parity(self, train: list[dict], test: list[dict]) -> list[ValidationFinding]:
        """Verify train and test questions have the same keys."""
        train_keys = set(train[0].keys()) if train else set()
        test_keys = set(test[0].keys()) if test else set()

        if train_keys != test_keys:
            return [ValidationFinding(
                severity="error",
                rule="schema_mismatch",
                message=f"Key difference: train-only={train_keys - test_keys}, test-only={test_keys - train_keys}",
                context={"train_keys": sorted(train_keys), "test_keys": sorted(test_keys)},
            )]
        return []

    def _check_tool_coverage(self, train: list[dict], test: list[dict]) -> list[ValidationFinding]:
        """Check that every tool_name in train also appears in test and vice versa."""
        train_tools = set(q["tool_name"] for q in train)
        test_tools = set(q["tool_name"] for q in test)

        findings = []
        train_only = train_tools - test_tools
        test_only = test_tools - train_tools

        if train_only:
            findings.append(ValidationFinding(
                severity="warning",
                rule="tool_missing_from_test",
                message=f"Tools in train but not test: {sorted(train_only)}",
                context={"tools": sorted(train_only)},
            ))
        if test_only:
            findings.append(ValidationFinding(
                severity="error",
                rule="tool_missing_from_train",
                message=f"Tools in test but not train: {sorted(test_only)}",
                context={"tools": sorted(test_only)},
            ))
        return findings

    def _check_seed_type_distribution(self, train: list[dict], test: list[dict]) -> list[ValidationFinding]:
        """Check that seed_type proportions are roughly similar between splits."""
        train_dist = Counter(q.get("seed_type", "unknown") for q in train)
        test_dist = Counter(q.get("seed_type", "unknown") for q in test)

        train_total = len(train)
        test_total = len(test)

        findings = []
        for seed_type in set(list(train_dist.keys()) + list(test_dist.keys())):
            train_pct = train_dist[seed_type] / train_total * 100 if train_total else 0
            test_pct = test_dist[seed_type] / test_total * 100 if test_total else 0
            diff = abs(train_pct - test_pct)
            if diff > 15:
                findings.append(ValidationFinding(
                    severity="warning",
                    rule="seed_type_skew",
                    message=(
                        f"'{seed_type}' is {train_pct:.1f}% of train but "
                        f"{test_pct:.1f}% of test (diff={diff:.1f}pp)."
                    ),
                    context={
                        "seed_type": seed_type,
                        "train_pct": round(train_pct, 1),
                        "test_pct": round(test_pct, 1),
                    },
                ))
        return findings

    def _check_leakage(self, train: list[dict], test: list[dict]) -> list[ValidationFinding]:
        """Detect near-duplicate queries between train and test sets."""
        train_sigs = {}
        for q in train:
            sig = self._signature(q["query"])
            train_sigs.setdefault(sig, []).append(q["question_id"])

        findings = []
        for q in test:
            sig = self._signature(q["query"])
            if sig in train_sigs:
                findings.append(ValidationFinding(
                    question_id=q["question_id"],
                    severity="error",
                    rule="train_test_leakage",
                    message=(
                        f"Test query near-duplicates train query "
                        f"{train_sigs[sig][0]}: '{sig[:80]}...'"
                    ),
                    context={
                        "test_id": q["question_id"],
                        "train_ids": train_sigs[sig],
                        "signature": sig[:120],
                    },
                ))
        return findings

    def _signature(self, query: str) -> str:
        """Normalize query for near-duplicate detection (same as CrossQuestionConsistencyValidator)."""
        s = query.lower()
        s = re.sub(r"[^\w\s.]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
