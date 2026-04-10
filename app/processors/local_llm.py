"""Local LLM loader and generator for on-device inference."""

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LocalLLM:
    """Loads and runs a local causal LLM for tool-call reasoning.

    Designed to run on a single GPU (e.g., NVIDIA A40 with 20GB+).
    Uses HuggingFace transformers for model loading and generation.

    Attributes:
        _model_name: HuggingFace model identifier.
        _model: Loaded causal LM instance.
        _tokenizer: Loaded tokenizer instance.
        _device: Target device string.
    """

    def __init__(self, model_name: str = LOCAL_MODEL) -> None:
        """Initialize the local LLM.

        Args:
            model_name: HuggingFace model identifier.
                Defaults to Qwen2.5-7B-Instruct (~14GB fp16).
        """
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def load_model(self, device: str = "cuda") -> None:
        """Load the model and tokenizer onto the specified device.

        Args:
            device: Target device ('cuda', 'cpu', or 'cuda:N').
        """
        self._device = device
        logger.info(f"Loading local LLM: {self._model_name} on {device}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()

        logger.info(f"Local LLM loaded: {self._model_name}")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> str:
        """Generate a completion from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Supported roles: 'system', 'user', 'assistant'.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            top_p: Nucleus sampling cutoff.

        Returns:
            Generated text content as a string.

        Raises:
            RuntimeError: If model has not been loaded via load_model().
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model_name
