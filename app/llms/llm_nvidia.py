from typing import Generator

from openai import OpenAI
from loguru import logger

from app.config import settings
from app.llms.llm import BaseLLM


class NvidiaLLM(BaseLLM):
    """NVIDIA API LLM provider using OpenAI-compatible client.

    Connects to the NVIDIA API endpoint via the OpenAI SDK. Configured
    through environment variables: NVIDIA_BASE_URL, NVIDIA_MODEL, NVIDIA_API_KEY.

    Args:
        model: Override the default model from settings.
        api_key: Override the default API key from settings.
        temperature: Sampling temperature. Lower values are more deterministic.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling probability cutoff.
        seed: Random seed for reproducible outputs.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> None:
        self._client = OpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=api_key or settings.NVIDIA_API_KEY,
        )
        self._model = model or settings.NVIDIA_MODEL
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._seed = seed
        logger.info(f"NvidiaLLM initialized with model={self._model}")

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate a non-streaming completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Override default generation parameters (temperature,
                max_tokens, top_p, seed).

        Returns:
            The full generated text content as a string.
        """
        response = self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            top_p=kwargs.get("top_p", self._top_p),
            seed=kwargs.get("seed", self._seed),
            stream=False,
        )
        return response.choices[0].message.content

    def generate_stream(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """Generate a streaming completion, yielding text chunks.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Override default generation parameters.

        Yields:
            String chunks of generated content as they arrive from the API.
        """
        completion = self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            top_p=kwargs.get("top_p", self._top_p),
            seed=kwargs.get("seed", self._seed),
            stream=True,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
