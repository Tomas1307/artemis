from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM provider implementations.

    All LLM providers must implement the `generate` method. This ensures
    chain methods and generators remain decoupled from any specific provider.

    Usage:
        Instantiate a concrete provider (e.g., NvidiaLLM) and inject it into
        chain methods or generators that require LLM capabilities.
    """

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate a completion from a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Supported roles: 'system', 'user', 'assistant'.
            **kwargs: Provider-specific generation parameters (temperature,
                max_tokens, etc.).

        Returns:
            The generated text content as a string.
        """

    @abstractmethod
    def generate_stream(self, messages: list[dict], **kwargs):
        """Generate a streaming completion from a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Provider-specific generation parameters.

        Yields:
            String chunks of the generated content as they arrive.
        """
