"""Fine-tune Llama-3.2-1B on (query + context + tools → tool_call) with LoRA.

For each training query, retrieves top-5 chunks with the fine-tuned
encoder, formats rich context including tool definitions, and trains
the decoder to generate the correct tool call.

Uses GISTEmbedLoss-fine-tuned encoder for retrieval. Requires:
  - artifacts/winner/chunks.json (from generate_chunks.py)
  - artifacts/winner/faiss_index.bin (from build_index.py)
  - artifacts/winner/embeddings.npy (from build_index.py)
  - artifacts/winner/encoder/finetuned_encoder/ (from finetune_encoder.py)
  - artifacts/winner/train_data.csv (from clean_data.py)
  - artifacts/winner/val_data.csv (from clean_data.py)

Usage:
    python -m winner_solution.scripts.finetune_decoder --device cuda
    python -m winner_solution.scripts.finetune_decoder --device cuda --epochs 5 --batch-size 8
"""

import csv
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from winner_solution.prompts.prompt_loader import prompt_loader
from winner_solution.utils.formatter import build_rich_context

PROJECT_ROOT = Path(__file__).parent.parent
TOOLS_PATH = PROJECT_ROOT / "data" / "tools_definition.json"
WINNER_DIR = PROJECT_ROOT / "artifacts" / "winner"
TRAIN_CSV_PATH = WINNER_DIR / "train_data.csv"
VAL_CSV_PATH = WINNER_DIR / "val_data.csv"
ENCODER_PATH = WINNER_DIR / "encoder" / "finetuned_encoder"
CHUNKS_PATH = WINNER_DIR / "chunks.json"
EMBEDDINGS_PATH = WINNER_DIR / "embeddings.npy"
INDEX_PATH = WINNER_DIR / "faiss_index.bin"
OUTPUT_DIR = WINNER_DIR / "decoder"

DECODER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


class ToolCallDataset(Dataset):
    """Dataset of (prompt, tool_call) pairs for decoder fine-tuning.

    Each example is tokenized as a full chat sequence (system + user + assistant).
    Labels are masked so loss is computed only on the assistant response (tool_call).

    Attributes:
        examples: List of dicts with 'input' and 'output' keys.
        tokenizer: Loaded tokenizer for encoding.
        max_length: Auto-detected from the longest example plus buffer.
        system_prompt: System message for the chat template.
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        system_prompt: str,
    ) -> None:
        """Initialize and auto-detect max_length from the longest example.

        Tokenizes every example once to find the actual maximum token count,
        then sets max_length to that value plus a 64-token buffer. Guarantees
        zero truncation while minimizing padding waste.

        Args:
            examples: List of dicts with 'input' (user content) and 'output' (tool_call).
            tokenizer: HuggingFace tokenizer.
            system_prompt: System message from YAML prompt.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = self._detect_max_length()

    def _detect_max_length(self) -> int:
        """Scan all examples and return the required max_length.

        Tokenizes each example's full chat sequence (system + user + assistant)
        without truncation to measure actual token counts. Returns the maximum
        found plus a 64-token safety buffer.

        Returns:
            Computed max_length (max token count + 64).
        """
        max_tokens = 0
        for example in self.examples:
            full_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]},
            ]
            full_text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False,
            )
            n_tokens = len(
                self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            )
            max_tokens = max(max_tokens, n_tokens)
        result = max_tokens + 64
        logger.info(f"Auto-detected max_length: {result} (longest example: {max_tokens} tokens)")
        return result

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Tokenize one example with label masking on the prompt portion.

        Args:
            idx: Example index.

        Returns:
            Dict with input_ids, attention_mask, and labels tensors.
        """
        example = self.examples[idx]

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": example["input"]},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_len = len(
            self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )

        full_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        full_text = self.tokenizer.apply_chat_template(full_messages, tokenize=False)
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _load_csv_rows(csv_path: Path) -> list[dict]:
    """Load rows from a CSV file.

    Args:
        csv_path: Path to CSV with id, query, tool_call columns.

    Returns:
        List of row dicts.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run clean_data.py first."
        )
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_examples(
    rows: list[dict],
    encoder: SentenceTransformer,
    chunks: list[dict],
    index: faiss.Index,
    tools_json: str,
    user_template: str,
) -> list[dict]:
    """Build training examples by retrieving context for each query.

    Embeds all queries with the fine-tuned encoder, retrieves top-5
    chunks from FAISS, formats the user prompt with tool definitions
    and rich context.

    Args:
        rows: Data rows with 'query' and 'tool_call' fields.
        encoder: Fine-tuned SentenceTransformer for query embedding.
        chunks: All document chunks from chunks.json.
        index: FAISS index over chunk embeddings.
        tools_json: Tool definitions as JSON string.
        user_template: User prompt template with {tools}, {context}, {query} placeholders.

    Returns:
        List of dicts with 'input' (formatted user prompt) and 'output' (tool_call).
    """
    query_texts = [row["query"] for row in rows]
    logger.info(f"Embedding {len(query_texts)} queries...")
    query_embeddings = encoder.encode(
        query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True,
    )

    _, indices_all = index.search(query_embeddings.astype(np.float32), 5)

    examples = []
    for i, row in enumerate(rows):
        retrieved = [chunks[idx] for idx in indices_all[i] if idx >= 0]
        context = build_rich_context(retrieved)
        user_content = user_template.format(
            tools=tools_json,
            context=context,
            query=row["query"],
        )
        examples.append({
            "input": user_content,
            "output": row["tool_call"],
        })

    return examples


def main() -> None:
    """Fine-tune Llama-3.2-1B with LoRA on retrieved context + tool calls."""
    args = sys.argv[1:]
    device = "cuda"
    epochs = 5
    batch_size = 2
    lr = 1e-4

    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--epochs" in args:
        epochs = int(args[args.index("--epochs") + 1])
    if "--batch-size" in args:
        batch_size = int(args[args.index("--batch-size") + 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system_prompt = prompt_loader.get_system_message_by_type("decoder_system")
    user_template = prompt_loader.get_prompt_template_by_type("decoder_user")
    if not user_template:
        raise ValueError("No prompt template found for 'decoder_user'.")

    tools_json = TOOLS_PATH.read_text(encoding="utf-8")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))

    logger.info(f"Loading fine-tuned encoder from {ENCODER_PATH}...")
    encoder = SentenceTransformer(str(ENCODER_PATH), device=device)

    train_rows = _load_csv_rows(TRAIN_CSV_PATH)
    val_rows = _load_csv_rows(VAL_CSV_PATH)
    logger.info(f"Train rows: {len(train_rows)}, Val rows: {len(val_rows)}")

    logger.info("Building training examples...")
    train_examples = build_examples(
        train_rows, encoder, chunks, index, tools_json, user_template,
    )
    logger.info("Building validation examples...")
    val_examples = build_examples(
        val_rows, encoder, chunks, index, tools_json, user_template,
    )
    logger.info(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")

    logger.info(f"Loading decoder: {DECODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ToolCallDataset(train_examples, tokenizer, system_prompt)
    val_dataset = ToolCallDataset(val_examples, tokenizer, system_prompt)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info(
        f"Training: {epochs} epochs, batch_size={batch_size}, "
        f"grad_accum=16, effective_batch={batch_size * 16}, lr={lr}"
    )
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR / "finetuned_decoder"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "finetuned_decoder"))
    logger.info(f"Fine-tuned decoder saved to {OUTPUT_DIR / 'finetuned_decoder'}")


if __name__ == "__main__":
    main()
