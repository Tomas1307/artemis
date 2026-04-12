"""Prepare student-facing data files for the ARTEMIS competition.

Takes the original data.csv (3186 rows) and produces:
  - train.csv: ~2739 rows WITH duplicates (students must clean). Has tool_call.
  - test.csv: 447 queries, id + query ONLY (no answers). Public Kaggle eval.
  - test_gold.json: 447 answers for Kaggle scoring (never given to students).
  - sample_submission_public.csv: 447 IDs with no_action placeholder.

The split uses the SAME seed (42) and logic as winner_solution/scripts/clean_data.py
so the 447 test IDs are identical to the internal test partition.

Usage:
    python -m proyecto_artemis.scripts.prepare_student_data
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
DATA_CSV_PATH = PROJECT_ROOT / "datos_entrenamiento" / "data.csv"
TRAIN_OUTPUT = PROJECT_ROOT / "datos_entrenamiento" / "train.csv"
TEST_OUTPUT = PROJECT_ROOT / "evaluacion" / "test.csv"
TEST_GOLD_OUTPUT = PROJECT_ROOT / "evaluacion" / "test_gold.json"
SAMPLE_SUB_OUTPUT = PROJECT_ROOT / "evaluacion" / "sample_submission_public.csv"

TEST_RATIO = 0.15
SEED = 42


def _extract_tool_name(tool_call: str) -> str:
    """Extract the tool name from a tool call string.

    Args:
        tool_call: Full canonical tool call string.

    Returns:
        Tool name, or 'unknown' if unparseable.
    """
    match = re.match(r'(\w+)', tool_call.strip())
    return match.group(1) if match else "unknown"


def main() -> None:
    """Produce student-facing train.csv, test.csv, test_gold.json, and sample_submission_public.csv."""
    with open(DATA_CSV_PATH, encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    logger.info(f"Raw rows from data.csv: {len(raw_rows)}")

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for row in raw_rows:
        key = (row["query"], row["tool_call"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    logger.info(f"After dedup: {len(deduped)} unique rows")

    by_tool: dict[str, list[dict]] = defaultdict(list)
    for row in deduped:
        by_tool[_extract_tool_name(row["tool_call"])].append(row)

    rng = np.random.default_rng(SEED)
    test_ids: set[str] = set()
    test_rows_clean: list[dict] = []

    for tool_name, tool_rows in sorted(by_tool.items()):
        n_test = max(1, int(len(tool_rows) * TEST_RATIO))
        indices = rng.permutation(len(tool_rows))
        test_indices = set(indices[:n_test].tolist())
        for i, row in enumerate(tool_rows):
            if i in test_indices:
                test_ids.add(row["id"])
                test_rows_clean.append(row)

    rng.shuffle(test_rows_clean)
    logger.info(f"Test rows (public eval): {len(test_rows_clean)}")

    train_rows: list[dict] = []
    removed_dupes = 0
    for row in raw_rows:
        if row["id"] in test_ids:
            continue
        dup_key = (row["query"], row["tool_call"])
        dup_id_in_test = any(
            r["id"] in test_ids
            for r in raw_rows
            if (r["query"], r["tool_call"]) == dup_key and r["id"] != row["id"]
        )
        if dup_id_in_test:
            removed_dupes += 1
            continue
        train_rows.append(row)

    logger.info(f"Train rows (with dupes): {len(train_rows)}")
    if removed_dupes:
        logger.info(f"Removed {removed_dupes} train rows that were duplicates of test queries")

    train_tool_dist = Counter(_extract_tool_name(r["tool_call"]) for r in train_rows)
    test_tool_dist = Counter(_extract_tool_name(r["tool_call"]) for r in test_rows_clean)
    logger.info("Train tool distribution:")
    for tool, count in train_tool_dist.most_common():
        logger.info(f"  {tool}: {count}")
    logger.info("Test tool distribution:")
    for tool, count in test_tool_dist.most_common():
        logger.info(f"  {tool}: {count}")

    fieldnames_train = ["id", "query", "tool_call"]
    with open(TRAIN_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_train, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(train_rows)
    logger.info(f"Train saved: {TRAIN_OUTPUT}")

    fieldnames_test = ["id", "query"]
    with open(TEST_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_test, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(test_rows_clean)
    logger.info(f"Test saved (no answers): {TEST_OUTPUT}")

    gold_entries = [
        {"question_id": r["id"], "query": r["query"], "tool_call": r["tool_call"]}
        for r in test_rows_clean
    ]
    TEST_GOLD_OUTPUT.write_text(
        json.dumps(gold_entries, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(f"Test gold saved: {TEST_GOLD_OUTPUT}")

    with open(SAMPLE_SUB_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tool_call"])
        writer.writeheader()
        for r in test_rows_clean:
            writer.writerow({"id": r["id"], "tool_call": "no_action"})
    logger.info(f"Sample submission saved: {SAMPLE_SUB_OUTPUT}")


if __name__ == "__main__":
    main()
