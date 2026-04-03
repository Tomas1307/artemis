from pathlib import Path

import yaml
from loguru import logger

from app.pipelines.document_pipeline.schemas.document_spec import DocumentSpec


class LoadRegistryStep:
    """Loads the document registry YAML and returns validated DocumentSpec list.

    Reads the document_registry.yaml file, parses each entry into a
    DocumentSpec model, and returns the full list for downstream processing.

    Args:
        registry_path: Path to the document_registry.yaml file.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self._path = Path(registry_path)

    def execute(self) -> list[DocumentSpec]:
        """Load and validate all document specs from the registry.

        Returns:
            List of validated DocumentSpec instances.

        Raises:
            FileNotFoundError: If the registry file does not exist.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Document registry not found: {self._path}")

        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))

        specs = []
        for doc_id, entry in raw.items():
            spec = DocumentSpec(
                doc_id=doc_id,
                title=entry["title"],
                type=entry["type"],
                target_words=entry["target_words"],
                skeleton_refs=entry.get("skeleton_refs", []),
                sections=entry["sections"],
            )
            specs.append(spec)

        logger.info(f"Loaded {len(specs)} document specs from registry")
        return specs
