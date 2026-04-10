"""Pydantic models representing validation findings."""

from pydantic import BaseModel, Field


class ValidationFinding(BaseModel):
    """Single validation finding for one question or one rule."""

    question_id: str | None = Field(default=None, description="Question ID, if applicable.")
    severity: str = Field(description="Severity level: error, warning, or info.")
    rule: str = Field(description="Short identifier of the rule that was violated.")
    message: str = Field(description="Human-readable explanation of the finding.")
    context: dict = Field(default_factory=dict, description="Extra structured context.")


class ValidationReport(BaseModel):
    """Aggregated report from a single validator run."""

    validator_name: str = Field(description="Name of the validator that produced this report.")
    total_checked: int = Field(description="Number of items inspected.")
    passed: int = Field(description="Number of items that passed all rules.")
    findings: list[ValidationFinding] = Field(default_factory=list, description="All findings.")

    @property
    def error_count(self) -> int:
        """Return the number of error-severity findings."""
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def warning_count(self) -> int:
        """Return the number of warning-severity findings."""
        return sum(1 for f in self.findings if f.severity == "warning")
