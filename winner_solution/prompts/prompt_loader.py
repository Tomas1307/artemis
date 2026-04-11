"""Singleton prompt loader for winner solution YAML prompts."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class PromptLoader:
    """Singleton loader for YAML prompt configurations.

    Maps logical prompt type names to physical YAML files and keys.
    Caches loaded files in memory to avoid repeated disk reads.

    Attributes:
        PROMPT_TYPE_MAPPING: Dict mapping type names to file+key pairs.
    """

    _instance = None

    def __new__(cls) -> "PromptLoader":
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize prompts directory, cache, and type mapping."""
        if self._initialized:
            return
        self._initialized = True
        self._prompts_dir = Path(__file__).parent / "template"
        self._cache: Dict[str, Any] = {}
        self.PROMPT_TYPE_MAPPING: Dict[str, Dict[str, str]] = {
            "chunk_summarization": {
                "file": "summarization.yaml",
                "key": "chunk_summarization",
            },
            "subchunk_summarization": {
                "file": "summarization.yaml",
                "key": "subchunk_summarization",
            },
            "decoder_system": {
                "file": "decoder.yaml",
                "key": "decoder_system",
            },
            "decoder_user": {
                "file": "decoder.yaml",
                "key": "decoder_user",
            },
        }

    def _load_file(self, filename: str) -> Dict[str, Any]:
        """Load and cache a YAML prompt file.

        Args:
            filename: YAML filename relative to the template directory.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        if filename not in self._cache:
            file_path = self._prompts_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
            self._cache[filename] = yaml.safe_load(
                file_path.read_text(encoding="utf-8")
            )
        return self._cache[filename]

    def _get_mapping(self, prompt_type: str) -> Dict[str, str]:
        """Return the file and key mapping for a given prompt type.

        Args:
            prompt_type: Logical prompt type name.

        Returns:
            Dict with 'file' and 'key' entries.

        Raises:
            ValueError: If the prompt type is not registered.
        """
        if prompt_type not in self.PROMPT_TYPE_MAPPING:
            valid = list(self.PROMPT_TYPE_MAPPING)
            raise ValueError(
                f"Unknown prompt type: '{prompt_type}'. Available: {valid}"
            )
        return self.PROMPT_TYPE_MAPPING[prompt_type]

    def get_system_message_by_type(self, prompt_type: str) -> str:
        """Return the system message for the given prompt type.

        Args:
            prompt_type: Logical prompt type name.

        Returns:
            System message string. Empty string if not defined.
        """
        mapping = self._get_mapping(prompt_type)
        data = self._load_file(mapping["file"])
        return data.get(mapping["key"], {}).get("system_message", "")

    def get_prompt_template_by_type(self, prompt_type: str) -> Optional[str]:
        """Return the prompt template for the given prompt type.

        Args:
            prompt_type: Logical prompt type name.

        Returns:
            Prompt template string, or None if not defined.
        """
        mapping = self._get_mapping(prompt_type)
        data = self._load_file(mapping["file"])
        return data.get(mapping["key"], {}).get("prompt_template")


prompt_loader = PromptLoader()
