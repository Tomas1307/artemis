from pathlib import Path

from pydantic import BaseModel


class QuestionPipelineSettings(BaseModel):
    """Configuration for the question generation pipeline.

    Attributes:
        output_base_dir: Base directory for proyecto_artemis deliverables.
        valid_doc_ids_path: Path to documentos_masa.json for doc_id validation.
        target_per_tool: Number of questions to generate per tool. Default 200.
        random_seed: Seed for reproducible sampling in seed generation. Default 42.
        max_retries: Maximum LLM call attempts per seed on failure. Default 3.
        tool_filter: If set, only generate questions for these tool names.
    """

    output_base_dir: str = str(
        Path(__file__).parent.parent.parent.parent / "proyecto_artemis"
    )
    valid_doc_ids_path: str = str(
        Path(__file__).parent.parent.parent.parent
        / "proyecto_artemis"
        / "base_conocimiento"
        / "documentos_masa.json"
    )
    target_per_tool: int = 200
    random_seed: int = 42
    max_retries: int = 3
    tool_filter: list[str] | None = None
