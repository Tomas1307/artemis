"""Run zero-shot inference on test queries — baseline solution.

Retrieves top-5 chunks, formats a simple prompt, and generates
tool calls with Llama-3.2-1B (no fine-tuning).

Usage:
    python -m app.baseline_solution.scripts.run_inference --device cuda
    python -m app.baseline_solution.scripts.run_inference --device cuda --split train --sample 100
"""

import csv
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_solution.utils.embedder import embed_texts, load_encoder
from baseline_solution.utils.formatter import extract_tool_call, format_context

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "baseline"
TEST_CSV_PATH = PROJECT_ROOT / "data" / "test_queries.csv"
PUBLIC_TEST_CSV_PATH = PROJECT_ROOT / "data" / "test.csv"
PUBLIC_TEST_GOLD_PATH = PROJECT_ROOT / "data" / "test_gold.json"
TRAIN_CSV_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_GOLD_PATH = PROJECT_ROOT / "data" / "test_gold_standard.json"
TOOLS_PATH = PROJECT_ROOT / "data" / "tools_definition.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "baseline"

DECODER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = """You are the MASA Kuntur Station operations assistant. Given an operator query and retrieved document context, determine the single correct tool call to execute.

Output ONLY the tool call in canonical format. Nothing else.

Canonical format rules:
- No spaces after commas
- Parameters in defined order
- String values in single quotes
- All lowercase for enum values
- Numeric values unquoted"""

USER_TEMPLATE = """TOOLS:
{tools}

CONTEXT:
{context}

QUERY: {query}

TOOL CALL:"""


def load_decoder(model_name: str, device: str) -> tuple:
    """Load Llama-3.2-1B for inference.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading decoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_tool_call(model, tokenizer, messages: list[dict], device: str) -> str:
    """Generate a tool call from chat messages.

    Args:
        model: Loaded causal LM.
        tokenizer: Loaded tokenizer.
        messages: Chat messages with system and user roles.
        device: Target device.

    Returns:
        Generated text string.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()



def main() -> None:
    """Run baseline inference pipeline."""
    args = sys.argv[1:]
    device = "cuda"
    split = "test"
    sample_size = None

    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--split" in args:
        split = args[args.index("--split") + 1]
    if "--sample" in args:
        sample_size = int(args[args.index("--sample") + 1])

    chunks = json.loads((ARTIFACTS_DIR / "chunks.json").read_text(encoding="utf-8"))
    embeddings = np.load(str(ARTIFACTS_DIR / "embeddings.npy"))
    index = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.bin"))
    tools_json = TOOLS_PATH.read_text(encoding="utf-8")

    if split == "test":
        with open(TEST_CSV_PATH, encoding="utf-8") as f:
            queries = [{"id": r["id"], "query": r["query"]} for r in csv.DictReader(f)]
        gold_map = {}
        if TEST_GOLD_PATH.exists():
            gold = json.loads(TEST_GOLD_PATH.read_text(encoding="utf-8"))
            gold_map = {g["question_id"]: g["tool_call"] for g in gold}
            logger.info("Gold standard loaded — accuracy will be computed.")
        else:
            logger.warning("test_gold_standard.json not found — skipping accuracy computation.")
    elif split == "public_test":
        with open(PUBLIC_TEST_CSV_PATH, encoding="utf-8") as f:
            queries = [{"id": r["id"], "query": r["query"]} for r in csv.DictReader(f)]
        gold_map = {}
        if PUBLIC_TEST_GOLD_PATH.exists():
            gold = json.loads(PUBLIC_TEST_GOLD_PATH.read_text(encoding="utf-8"))
            gold_map = {g["question_id"]: g["tool_call"] for g in gold}
            logger.info("Public test gold loaded — accuracy will be computed.")
        else:
            logger.info("No public test gold found — predictions only.")
    else:
        with open(TRAIN_CSV_PATH, encoding="utf-8") as f:
            queries = [{"id": r["id"], "query": r["query"], "gold": r["tool_call"]} for r in csv.DictReader(f)]
        gold_map = {q["id"]: q["gold"] for q in queries}

    if sample_size and sample_size < len(queries):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(queries), sample_size, replace=False)
        queries = [queries[i] for i in sorted(indices)]

    logger.info(f"Queries: {len(queries)} ({split})")

    encoder = load_encoder(device=device)
    model, tokenizer = load_decoder(DECODER_MODEL, device)

    query_texts = [q["query"] for q in queries]
    query_embeddings = embed_texts(encoder, query_texts)

    scores_all, indices_all = index.search(query_embeddings.astype(np.float32), 5)

    predictions = []
    start_time = time.time()

    for i, q in enumerate(queries):
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Processing {i+1}/{len(queries)} ({elapsed:.0f}s)")

        context = format_context(chunks, indices_all[i], scores_all[i])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                tools=tools_json,
                context=context,
                query=q["query"],
            )},
        ]

        raw_output = generate_tool_call(model, tokenizer, messages, device)
        tool_call = extract_tool_call(raw_output)
        predictions.append(tool_call)

    if gold_map:
        correct = sum(1 for q, p in zip(queries, predictions) if p.strip() == gold_map.get(q["id"], "").strip())
        total = len(queries)
        logger.info(f"Accuracy: {correct}/{total} = {correct/total:.4f}")

    results = [{"id": q["id"], "tool_call": p} for q, p in zip(queries, predictions)]
    output_path = OUTPUT_DIR / f"predictions_{split}.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    submission_path = OUTPUT_DIR / f"submission_{split}.csv"
    with open(submission_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tool_call"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
