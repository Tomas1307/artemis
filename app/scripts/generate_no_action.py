"""Generate additional no_action questions to bring percentage from ~4% to ~11%.

Generates diverse informational/historical queries that don't map to any tool.
Adds them to data.csv, gold_standard.json, test_queries.csv, test_gold_standard.json,
and sample_submission.csv.

Usage:
    python app/scripts/generate_no_action.py
    python app/scripts/generate_no_action.py --dry-run
"""

import csv
import io
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.llms.llm_nvidia import NvidiaLLM

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_CSV = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "data.csv"
GOLD_JSON = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_CSV = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_queries.csv"
TEST_GOLD = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"
SAMPLE_SUB = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "sample_submission.csv"

SYSTEM_PROMPT = """You are a MASA (Mision Aeroespacial Sudamericana Avanzada) control center operator simulator.
Generate realistic queries that a ground control operator would ask that are PURELY INFORMATIONAL.
These queries should NOT require any system action — no alerts, no telemetry pulls, no maintenance scheduling,
no protocol activation, no trajectory calculations, no supply requests, no messages to crew, no system control.

They should be questions about:
- Station history, past missions (e.g., Cóndor-5, Cóndor-7), mission timelines
- Crew backgrounds, training records, qualifications, biographies
- Module design specs, purposes, capacity, construction history
- MASA organizational structure, partnerships, founding history
- Scientific research programs, experiment history, results
- General knowledge about Kuntur Station architecture and operations
- Standard operating procedures (asking ABOUT them, not activating them)
- Cultural significance of module names (Cóndor, Quetzal, Jaguar, Colibrí, Vicuña, Tucán)
- Past incidents and their resolutions (historical, not current)
- Training protocols and certification requirements

Rules:
- Each query must be a natural, conversational request from a control center operator
- Queries must be diverse in phrasing, topic, and specificity
- Use crew names naturally: Commander Santiago Reyes, Pilot Ana Valdivia, Specialist Kai Nakamura, Specialist Fatima Al-Hassan, Engineer Pavel Kozlov, Medical Officer Lucía Mendoza
- Use accented module names in query text: Cóndor, Quetzal, Jaguar, Colibrí, Vicuña, Tucán
- Each query on its own line, no numbering, no bullets, no quotes
- Minimum 10 words per query, maximum 40 words
- NEVER generate queries that could be answered by pulling telemetry, sending alerts, scheduling maintenance, etc.
- Generate exactly {count} queries"""

TOPICS = [
    "station history and past missions",
    "crew backgrounds and qualifications",
    "module design and architecture",
    "MASA organization and partnerships",
    "scientific research programs",
    "standard operating procedures (informational only)",
    "cultural significance of module names",
    "past incidents and resolutions",
    "training and certification requirements",
    "station construction timeline and milestones",
    "crew rotation schedules and policies",
    "international cooperation and agreements",
    "life support system design philosophy",
    "communication protocols with Earth",
    "EVA history and procedures",
]


def generate_batch(llm: NvidiaLLM, topic: str, count: int, existing: set) -> list[str]:
    """Generate a batch of no_action queries for a given topic.

    Args:
        llm: LLM provider instance.
        topic: Topic focus for this batch.
        count: Number of queries to generate.
        existing: Set of existing query texts to avoid duplicates.

    Returns:
        List of unique query strings.
    """
    prompt = f"Focus on: {topic}\n\nGenerate exactly {count} diverse control center operator queries."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(count=count)},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        try:
            response = llm.generate(messages, temperature=0.95, max_tokens=3000)
            lines = [line.strip().strip("-").strip("0123456789.").strip() for line in response.strip().split("\n")]
            queries = [line for line in lines if len(line.split()) >= 8 and line not in existing]
            logger.info(f"  [{topic[:30]}] attempt {attempt+1}: {len(queries)} unique queries")
            return queries
        except Exception as exc:
            logger.warning(f"  [{topic[:30]}] attempt {attempt+1} failed: {exc}")
            time.sleep(3)

    return []


def main() -> None:
    """Generate no_action questions and add to all data files."""
    dry_run = "--dry-run" in sys.argv
    random.seed(42)

    with open(DATA_CSV, encoding="utf-8") as f:
        data_rows = list(csv.DictReader(f))
    with open(GOLD_JSON, encoding="utf-8") as f:
        gold = json.load(f)
    with open(TEST_CSV, encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    with open(TEST_GOLD, encoding="utf-8") as f:
        test_gold = json.load(f)

    existing_queries = set(r["query"] for r in data_rows)
    existing_queries.update(r["query"] for r in test_rows)

    target_data = 213
    target_test = 25
    target_total = target_data + target_test

    logger.info(f"Generating {target_total} no_action queries ({target_data} data + {target_test} test)")

    llm = NvidiaLLM()
    all_queries: list[str] = []
    queries_per_topic = (target_total // len(TOPICS)) + 5

    for topic in TOPICS:
        logger.info(f"Topic: {topic}")
        batch = generate_batch(llm, topic, queries_per_topic, existing_queries)
        for q in batch:
            existing_queries.add(q)
        all_queries.extend(batch)
        logger.info(f"  Running total: {len(all_queries)}")
        if len(all_queries) >= target_total + 20:
            break
        time.sleep(1)

    random.shuffle(all_queries)
    all_queries = all_queries[:target_total]
    logger.info(f"Final pool: {len(all_queries)} queries")

    if len(all_queries) < target_total:
        logger.warning(f"Only generated {len(all_queries)}/{target_total} — will use what we have")
        target_test = min(target_test, len(all_queries) // 10)
        target_data = len(all_queries) - target_test

    test_queries = all_queries[:target_test]
    data_queries = all_queries[target_test:]

    max_q_num = max(int(r["id"].split("-")[1]) for r in data_rows)
    max_t_num = max(int(r["id"].split("-")[1]) for r in test_rows)

    logger.info(f"Adding {len(data_queries)} to data.csv (IDs from Q-{max_q_num+1:05d})")
    logger.info(f"Adding {len(test_queries)} to test (IDs from T-{max_t_num+1:05d})")

    if dry_run:
        logger.info("DRY RUN — showing first 10 queries:")
        for q in all_queries[:10]:
            logger.info(f"  {q[:100]}")
        return

    new_data_rows = []
    new_gold_entries = []
    for i, query in enumerate(data_queries):
        qid = f"Q-{max_q_num + 1 + i:05d}"
        new_data_rows.append({"id": qid, "query": query, "tool_call": "no_action"})
        new_gold_entries.append({
            "question_id": qid,
            "query": query,
            "tool_call": "no_action",
            "tool_name": "no_action",
            "seed_type": "direct",
            "doc_id": None,
            "protocol_id": None,
        })

    new_test_rows = []
    new_test_gold = []
    new_sample_sub = []
    for i, query in enumerate(test_queries):
        tid = f"T-{max_t_num + 1 + i:05d}"
        new_test_rows.append({"id": tid, "query": query})
        new_test_gold.append({
            "question_id": tid,
            "query": query,
            "tool_call": "no_action",
            "tool_name": "no_action",
            "seed_type": "direct",
            "doc_id": None,
            "protocol_id": None,
        })
        new_sample_sub.append({"id": tid, "tool_call": "no_action"})

    all_data = data_rows + new_data_rows
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["id", "query", "tool_call"])
    writer.writeheader()
    writer.writerows(all_data)
    DATA_CSV.write_text(out.getvalue(), encoding="utf-8", newline="")
    logger.info(f"data.csv: {len(data_rows)} -> {len(all_data)}")

    all_gold = gold + new_gold_entries
    GOLD_JSON.write_text(json.dumps(all_gold, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"gold_standard.json: {len(gold)} -> {len(all_gold)}")

    all_test = test_rows + new_test_rows
    out2 = io.StringIO()
    writer2 = csv.DictWriter(out2, fieldnames=["id", "query"])
    writer2.writeheader()
    writer2.writerows(all_test)
    TEST_CSV.write_text(out2.getvalue(), encoding="utf-8", newline="")
    logger.info(f"test_queries.csv: {len(test_rows)} -> {len(all_test)}")

    all_test_gold = test_gold + new_test_gold
    TEST_GOLD.write_text(json.dumps(all_test_gold, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"test_gold_standard.json: {len(test_gold)} -> {len(all_test_gold)}")

    with open(SAMPLE_SUB, encoding="utf-8") as f:
        existing_sub = list(csv.DictReader(f))
    all_sub = existing_sub + new_sample_sub
    out3 = io.StringIO()
    writer3 = csv.DictWriter(out3, fieldnames=["id", "tool_call"])
    writer3.writeheader()
    writer3.writerows(all_sub)
    SAMPLE_SUB.write_text(out3.getvalue(), encoding="utf-8", newline="")
    logger.info(f"sample_submission.csv: {len(existing_sub)} -> {len(all_sub)}")

    na_final = sum(1 for r in all_data if r["tool_call"] == "no_action")
    na_test_final = sum(1 for g in all_test_gold if g["tool_call"] == "no_action")
    print(f"\n=== RESULTS ===")
    print(f"data.csv: {len(all_data)} rows, {na_final} no_action ({na_final/len(all_data)*100:.1f}%)")
    print(f"test: {len(all_test)} rows, {na_test_final} no_action ({na_test_final/len(all_test)*100:.1f}%)")


if __name__ == "__main__":
    main()
