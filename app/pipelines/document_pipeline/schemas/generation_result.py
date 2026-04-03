from pydantic import BaseModel


class ValidationVerdict(BaseModel):
    """Result from the LLM-as-judge validation of a generated document.

    Attributes:
        passed: Whether the document passed validation.
        facts_checked: Total number of skeleton facts verified.
        facts_present: Number of facts confirmed present in the document.
        facts_missing: List of skeleton facts not found in the document.
        judge_reasoning: Raw reasoning output from the judge model.
    """

    passed: bool
    facts_checked: int
    facts_present: int
    facts_missing: list[str]
    judge_reasoning: str


class GenerationResult(BaseModel):
    """Tracks the outcome of generating a single document.

    Attributes:
        doc_id: Document identifier from the registry.
        title: Document title.
        type: Document category.
        status: Generation status (success, failed, validation_failed).
        word_count: Actual word count of the generated document.
        file_path: Path where the .md file was saved. None if generation failed.
        attempts: Number of generation attempts before success or final failure.
        validation: Judge validation result. None if validation was skipped.
        error: Error message if generation failed. None on success.
    """

    doc_id: str
    title: str
    type: str
    status: str
    word_count: int = 0
    file_path: str | None = None
    attempts: int = 1
    validation: ValidationVerdict | None = None
    error: str | None = None
