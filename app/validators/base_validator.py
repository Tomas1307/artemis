"""Abstract base class for dataset validators (Strategy pattern)."""

from abc import ABC, abstractmethod

from app.validators.validation_report import ValidationReport


class BaseValidator(ABC):
    """Abstract base class for all dataset validators.

    Each validator implements one specific rule or audit check, returning a
    ValidationReport that aggregates findings.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this validator."""

    @abstractmethod
    def validate(self) -> ValidationReport:
        """Run the validation and return a structured report.

        Returns:
            ValidationReport with all findings from this validator run.
        """
