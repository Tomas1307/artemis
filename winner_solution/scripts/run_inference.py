"""Run inference with fine-tuned encoder + decoder — winner solution.

Usage:
    python -m app.winner_solution.scripts.run_inference --device cuda
    python -m app.winner_solution.scripts.run_inference --device cuda --split train --sample 100
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
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from winner_solution.utils.formatter import build_rich_context, extract_tool_call

PROJECT_ROOT = Path(__file__).parent.parent
TEST_CSV_PATH = PROJECT_ROOT / "data" / "test_queries.csv"
DATA_CSV_PATH = PROJECT_ROOT / "data" / "data.csv"
TEST_GOLD_PATH = PROJECT_ROOT / "data" / "test_gold_standard.json"
TOOLS_PATH = PROJECT_ROOT / "data" / "tools_definition.json"
WINNER_DIR = PROJECT_ROOT / "artifacts" / "winner"
ENCODER_PATH = WINNER_DIR / "encoder" / "finetuned_encoder"
DECODER_PATH = WINNER_DIR / "decoder" / "finetuned_decoder"
CHUNKS_PATH = WINNER_DIR / "chunks.json"
INDEX_PATH = WINNER_DIR / "faiss_index.bin"
OUTPUT_DIR = WINNER_DIR

DECODER_BASE = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = """You are the MASA Kuntur Station operations assistant. Given an operator query and retrieved document context, output the single correct tool call in canonical format.

Canonical format: no spaces after commas, parameters in defined order, single quotes for strings, all lowercase, numeric values unquoted."""

USER_TEMPLATE = """CONTEXT:
{context}

QUERY: {query}

TOOL CALL:"""


def main() -> None:
    """Run winner solution inference pipeline."""
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

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))

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
    else:
        with open(DATA_CSV_PATH, encoding="utf-8") as f:
            queries = [{"id": r["id"], "query": r["query"], "gold": r["tool_call"]} for r in csv.DictReader(f)]
        gold_map = {q["id"]: q["gold"] for q in queries}

    if sample_size and sample_size < len(queries):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(queries), sample_size, replace=False)
        queries = [queries[i] for i in sorted(indices)]

    logger.info(f"Loading fine-tuned encoder from {ENCODER_PATH}...")
    encoder = SentenceTransformer(str(ENCODER_PATH), device=device)

    logger.info(f"Loading fine-tuned decoder from {DECODER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(DECODER_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        DECODER_BASE,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(DECODER_PATH))
    model.eval()

    logger.info(f"Embedding {len(queries)} queries...")
    query_texts = [q["query"] for q in queries]
    query_embeddings = encoder.encode(
        query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True,
    )

    scores_all, indices_all = index.search(query_embeddings.astype(np.float32), 5)

    predictions = []
    start_time = time.time()

    for i, q in enumerate(queries):
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Processing {i+1}/{len(queries)} ({elapsed:.0f}s)")

        retrieved = [chunks[idx] for idx in indices_all[i] if idx >= 0]
        context = build_rich_context(retrieved)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, query=q["query"])},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0, do_sample=False)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        tool_call = extract_tool_call(raw_output)
        predictions.append(tool_call)

    if gold_map:
        correct = sum(1 for q, p in zip(queries, predictions) if p.strip() == gold_map.get(q["id"], "").strip())
        total = len(queries)
        logger.info(f"Accuracy: {correct}/{total} = {correct/total:.4f}")

    results = [{"id": q["id"], "tool_call": p} for q, p in zip(queries, predictions)]

    predictions_path = OUTPUT_DIR / f"predictions_{split}.json"
    predictions_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    submission_path = OUTPUT_DIR / f"submission_{split}.csv"
    with open(submission_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tool_call"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Predictions saved to {predictions_path}")
    logger.info(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
