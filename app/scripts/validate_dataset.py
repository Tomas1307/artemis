"""Run all dataset validators and produce a consolidated report.

Usage:
    python app/scripts/validate_dataset.py
    python app/scripts/validate_dataset.py --verbose
    python app/scripts/validate_dataset.py --only RAGDocCorrespondenceValidator
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.validators.base_validator import BaseValidator
from app.validators.cross_question_consistency_validator import CrossQuestionConsistencyValidator
from app.validators.direct_determinism_validator import DirectDeterminismValidator
from app.validators.protocol_doc_mapping_validator import ProtocolDocMappingValidator
from app.validators.rag_doc_correspondence_validator import RAGDocCorrespondenceValidator
from app.validators.validation_report import ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORT_PATH = PROJECT_ROOT / "proyecto_artemis" / "validation_report.json"


def get_validators() -> list[BaseValidator]:
    """Build the full list of dataset validators to run."""
    return [
        RAGDocCorrespondenceValidator(),
        ProtocolDocMappingValidator(),
        DirectDeterminismValidator(),
        CrossQuestionConsistencyValidator(),
    ]


def print_summary(reports: list[ValidationReport], verbose: bool) -> None:
    """Print a console summary of all validator reports."""
    print("\n" + "=" * 78)
    print("ARTEMIS DATASET VALIDATION REPORT")
    print("=" * 78)

    total_errors = 0
    total_warnings = 0

    for report in reports:
        errors = report.error_count
        warnings = report.warning_count
        total_errors += errors
        total_warnings += warnings

        status = "PASS" if errors == 0 else "FAIL"
        print(f"\n[{status}] {report.validator_name}")
        print(f"  Checked: {report.total_checked} | Passed: {report.passed} | Errors: {errors} | Warnings: {warnings}")

        if verbose or errors > 0:
            shown = 0
            for finding in report.findings:
                if not verbose and finding.severity != "error":
                    continue
                if shown >= 10 and not verbose:
                    print(f"  ... and {len([f for f in report.findings if f.severity == 'error']) - shown} more errors")
                    break
                qid = f"[{finding.question_id}] " if finding.question_id else ""
                print(f"  {finding.severity.upper():8s} {qid}{finding.rule}: {finding.message}")
                shown += 1

    print("\n" + "=" * 78)
    print(f"TOTAL: {total_errors} errors, {total_warnings} warnings across {len(reports)} validators")
    print("=" * 78)


def main() -> None:
    """Run all validators and write a JSON report."""
    args = sys.argv[1:]
    verbose = "--verbose" in args
    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1]

    validators = get_validators()
    if only:
        validators = [v for v in validators if v.name == only]

    reports: list[ValidationReport] = []
    for validator in validators:
        logger.info(f"Running {validator.name}...")
        report = validator.validate()
        reports.append(report)

    print_summary(reports, verbose)

    REPORT_PATH.write_text(
        json.dumps([r.model_dump() for r in reports], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Full report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
