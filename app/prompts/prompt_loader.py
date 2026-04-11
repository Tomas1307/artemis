from pathlib import Path
from typing import Any, Optional

import yaml


class PromptLoader:
    """Singleton-like loader for YAML prompt configurations.

    Manages caching and mapping between logical prompt types and physical files.
    All prompt templates are stored as YAML files in the template/ directory.

    Usage:
        from app.prompts.prompt_loader import prompt_loader
        system_msg = prompt_loader.get_system_message_by_type("document_writer")
        template = prompt_loader.get_prompt_template_by_type("document_writer")
    """

    PROMPT_TYPE_MAPPING: dict[str, dict[str, str]] = {
        "document_writer": {
            "file": "template/document_generation.yaml",
            "key": "document_writer",
        },
        "noise_writer": {
            "file": "template/document_generation.yaml",
            "key": "noise_writer",
        },
        "document_judge": {
            "file": "template/document_validation.yaml",
            "key": "document_judge",
        },
        "factual_accuracy": {
            "file": "template/document_review.yaml",
            "key": "factual_accuracy",
        },
        "hallucination_detection": {
            "file": "template/document_review.yaml",
            "key": "hallucination_detection",
        },
        "correction": {
            "file": "template/document_review.yaml",
            "key": "correction",
        },
        "query_generator": {
            "file": "template/question_generation.yaml",
            "key": "query_generator",
        },
        "rag_query_generator": {
            "file": "template/rag_question_generation.yaml",
            "key": "rag_query_generator",
        },
        "encoder_query_generator": {
            "file": "template/encoder_training.yaml",
            "key": "encoder_query_generator",
        },
        "chunk_summarizer": {
            "file": "template/chunk_summarization.yaml",
            "key": "chunk_summarizer",
        },
        "parent_summarizer": {
            "file": "template/chunk_summarization.yaml",
            "key": "parent_summarizer",
        },
        "query_reprompt": {
            "file": "template/rag_validation.yaml",
            "key": "query_reprompt",
        },
        "tool_reason_direct": {
            "file": "template/rag_validation.yaml",
            "key": "tool_reason_direct",
        },
        "tool_reason_cot": {
            "file": "template/rag_validation.yaml",
            "key": "tool_reason_cot",
        },
        "failure_classifier": {
            "file": "template/failure_diagnosis.yaml",
            "key": "failure_classifier",
        },
    }

    def __init__(self, prompts_dir: str = "app/prompts") -> None:
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, Any] = {}

    def load_prompts(self, filename: str) -> dict[str, Any]:
        """Load prompts from a YAML file with in-memory caching.

        Args:
            filename: Relative path to the YAML file within the prompts directory.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """
        if filename not in self._cache:
            file_path = self.prompts_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
            self._cache[filename] = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        return self._cache[filename]

    def get_system_message_by_type(self, prompt_type: str) -> str:
        """Retrieve the system message for a specific prompt type.

        Args:
            prompt_type: Logical prompt type name from PROMPT_TYPE_MAPPING.

        Returns:
            System message string for the given prompt type.
        """
        mapping = self._get_mapping(prompt_type)
        return self.get_system_message(mapping["file"], mapping["key"])

    def get_prompt_template_by_type(self, prompt_type: str) -> Optional[str]:
        """Retrieve the prompt template for a specific prompt type.

        Args:
            prompt_type: Logical prompt type name from PROMPT_TYPE_MAPPING.

        Returns:
            Prompt template string, or None if not defined.
        """
        mapping = self._get_mapping(prompt_type)
        return self.get_prompt_template(mapping["file"], mapping["key"])

    def get_config_by_type(self, prompt_type: str) -> dict[str, Any]:
        """Retrieve the configuration dict for a specific prompt type.

        Args:
            prompt_type: Logical prompt type name from PROMPT_TYPE_MAPPING.

        Returns:
            Configuration dictionary for the given prompt type.
        """
        mapping = self._get_mapping(prompt_type)
        return self.get_config(mapping["file"], mapping["key"])

    def get_full_prompt_by_type(self, prompt_type: str) -> dict[str, Any]:
        """Retrieve the complete prompt object for a specific prompt type.

        Args:
            prompt_type: Logical prompt type name from PROMPT_TYPE_MAPPING.

        Returns:
            Full prompt dictionary including system_message, prompt_template, and config.
        """
        mapping = self._get_mapping(prompt_type)
        return self.get_full_prompt(mapping["file"], mapping["key"])

    def _get_mapping(self, prompt_type: str) -> dict[str, str]:
        """Resolve a logical prompt type to its file and key mapping.

        Args:
            prompt_type: Logical prompt type name.

        Returns:
            Dictionary with 'file' and 'key' entries.

        Raises:
            ValueError: If the prompt type is not registered in PROMPT_TYPE_MAPPING.
        """
        if prompt_type not in self.PROMPT_TYPE_MAPPING:
            valid_types = list(self.PROMPT_TYPE_MAPPING.keys())
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {valid_types}")
        return self.PROMPT_TYPE_MAPPING[prompt_type]

    def get_system_message(self, filename: str, prompt_name: str) -> str:
        """Extract system message from a loaded YAML file.

        Args:
            filename: Relative path to the YAML file.
            prompt_name: Key within the YAML file.

        Returns:
            System message string, or empty string if not found.
        """
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {}).get("system_message", "")

    def get_prompt_template(self, filename: str, prompt_name: str) -> Optional[str]:
        """Extract prompt template from a loaded YAML file.

        Args:
            filename: Relative path to the YAML file.
            prompt_name: Key within the YAML file.

        Returns:
            Prompt template string, or None if not found.
        """
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {}).get("prompt_template")

    def get_config(self, filename: str, prompt_name: str) -> dict[str, Any]:
        """Extract configuration dictionary from a loaded YAML file.

        Args:
            filename: Relative path to the YAML file.
            prompt_name: Key within the YAML file.

        Returns:
            Configuration dictionary, or empty dict if not found.
        """
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {}).get("config", {})

    def get_full_prompt(self, filename: str, prompt_name: str) -> dict[str, Any]:
        """Extract the raw dictionary for a prompt entry.

        Args:
            filename: Relative path to the YAML file.
            prompt_name: Key within the YAML file.

        Returns:
            Full prompt dictionary as stored in YAML.
        """
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {})


prompt_loader = PromptLoader()
