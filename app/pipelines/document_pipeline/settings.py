from pathlib import Path

from pydantic import BaseModel


class DocumentPipelineSettings(BaseModel):
    """Configuration for the document generation pipeline.

    Attributes:
        registry_path: Path to the document_registry.yaml file.
        output_dir: Directory for generated .md files.
        max_retries: Maximum generation attempts per document on empty output.
        max_correction_cycles: Maximum review-correct iterations per document.
        max_api_retries: Maximum retries per API call on transient errors.
        skip_validation: If True, skip the review-correct validation step.
        doc_filter: If set, only generate documents matching these IDs.
    """

    registry_path: str = str(
        Path(__file__).parent.parent.parent / "skeleton" / "document_registry.yaml"
    )
    output_dir: str = str(
        Path(__file__).parent.parent.parent.parent / "proyecto_artemis" / "base_conocimiento"
    )
    max_retries: int = 2
    max_correction_cycles: int = 2
    max_api_retries: int = 3
    skip_validation: bool = False
    doc_filter: list[str] | None = None
