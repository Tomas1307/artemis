"""Fine-tune Llama-3.2-1B on (query + context → tool_call) with LoRA.

For each training query, retrieves top-5 chunks with the fine-tuned
encoder, formats rich context, and trains the decoder to generate
the correct tool call.

Usage:
    python -m app.winner_solution.scripts.finetune_decoder --device cuda
    python -m app.winner_solution.scripts.finetune_decoder --device cuda --epochs 3 --batch-size 4
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from winner_solution.utils.formatter import build_rich_context

PROJECT_ROOT = Path(__file__).parent.parent
DATA_CSV_PATH = PROJECT_ROOT / "data" / "data.csv"
TOOLS_PATH = PROJECT_ROOT / "data" / "tools_definition.json"
WINNER_DIR = PROJECT_ROOT / "artifacts" / "winner"
ENCODER_PATH = WINNER_DIR / "encoder" / "finetuned_encoder"
CHUNKS_PATH = WINNER_DIR / "chunks.json"
EMBEDDINGS_PATH = WINNER_DIR / "embeddings.npy"
INDEX_PATH = WINNER_DIR / "faiss_index.bin"
OUTPUT_DIR = WINNER_DIR / "decoder"

DECODER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = """You are the MASA Kuntur Station operations assistant. Given an operator query and retrieved document context, output the single correct tool call in canonical format.

Canonical format: no spaces after commas, parameters in defined order, single quotes for strings, all lowercase, numeric values unquoted."""

USER_TEMPLATE = """CONTEXT:
{context}

QUERY: {query}

TOOL CALL:"""


class ToolCallDataset(Dataset):
    """Dataset of (prompt, tool_call) pairs for decoder fine-tuning.

    Attributes:
        examples: List of dicts with 'input' (formatted prompt) and 'output' (tool_call).
        tokenizer: Loaded tokenizer for encoding.
        max_length: Maximum sequence length.
    """

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 1024) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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


def build_training_data(
    encoder: SentenceTransformer,
    chunks: list[dict],
    index: faiss.Index,
    tools_json: str,
    device: str,
) -> list[dict]:
    """Build training examples by retrieving context for each query.

    Args:
        encoder: Fine-tuned encoder for query embedding.
        chunks: All document chunks.
        index: FAISS index over chunk embeddings.
        tools_json: Tool definitions as JSON string.
        device: Torch device.

    Returns:
        List of dicts with 'input' (formatted prompt) and 'output' (tool_call).
    """
    with open(DATA_CSV_PATH, encoding="utf-8") as f:
        train_data = list(csv.DictReader(f))

    query_texts = [row["query"] for row in train_data]
    logger.info(f"Embedding {len(query_texts)} training queries...")
    query_embeddings = encoder.encode(
        query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True,
    )

    scores, indices = index.search(query_embeddings.astype(np.float32), 5)

    examples = []
    for i, row in enumerate(train_data):
        retrieved = [chunks[idx] for idx in indices[i] if idx >= 0]
        context = build_rich_context(retrieved)

        user_content = USER_TEMPLATE.format(context=context, query=row["query"])
        examples.append({
            "input": user_content,
            "output": row["tool_call"],
        })

    return examples


def main() -> None:
    """Fine-tune Llama-3.2-1B with LoRA on retrieved context + tool calls."""
    args = sys.argv[1:]
    device = "cuda"
    epochs = 3
    batch_size = 4
    lr = 1e-4

    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--epochs" in args:
        epochs = int(args[args.index("--epochs") + 1])
    if "--batch-size" in args:
        batch_size = int(args[args.index("--batch-size") + 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tools_json = TOOLS_PATH.read_text(encoding="utf-8")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(str(EMBEDDINGS_PATH))
    index = faiss.read_index(str(INDEX_PATH))

    logger.info(f"Loading fine-tuned encoder from {ENCODER_PATH}...")
    encoder = SentenceTransformer(str(ENCODER_PATH), device=device)

    logger.info("Building training data with retrieved context...")
    examples = build_training_data(encoder, chunks, index, tools_json, device)
    logger.info(f"Training examples: {len(examples)}")

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

    dataset = ToolCallDataset(examples, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR / "finetuned_decoder"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "finetuned_decoder"))
    logger.info(f"Fine-tuned decoder saved to {OUTPUT_DIR / 'finetuned_decoder'}")


if __name__ == "__main__":
    main()
