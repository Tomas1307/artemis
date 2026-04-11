"""Singleton manager for local Qwen2.5-7B-Instruct inference."""

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLMManager:
    """Singleton manager for local Qwen2.5-7B-Instruct LLM inference.

    Wraps a HuggingFace causal LM loaded once and reused across all
    summarization calls. The model is NOT loaded on instantiation — call
    load() explicitly before generate().

    Attributes:
        _model: Loaded causal LM, or None before load().
        _tokenizer: Loaded tokenizer, or None before load().
        _device: Target device string (e.g., 'cuda', 'cpu').
    """

    _instance = None

    def __new__(cls) -> "LocalLLMManager":
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize state. Model loading is deferred to load()."""
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._tokenizer = None
        self._device: str = "cuda"

    def load(self, model_name_or_path: str, device: str = "cuda") -> None:
        """Load model and tokenizer onto the target device.

        Idempotent: subsequent calls are no-ops if model is already loaded.

        Args:
            model_name_or_path: HuggingFace model ID or local filesystem path.
            device: Target device string (e.g., 'cuda', 'cuda:0', 'cpu').
        """
        if self._model is not None:
            logger.debug("LLM already loaded — skipping reload.")
            return
        logger.info(f"Loading LLM '{model_name_or_path}' on device '{device}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._device = device
        logger.info("LLM loaded successfully.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a response given system and user prompts.

        Args:
            system_prompt: System message defining the model role and task.
            user_prompt: User message with the specific input to process.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Generated text string with special tokens stripped.

        Raises:
            RuntimeError: If the model has not been loaded via load().
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call llm_manager.load() before generate()."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


llm_manager = LocalLLMManager()
