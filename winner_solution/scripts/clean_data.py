"""Clean and split data.csv for decoder fine-tuning.

Removes exact duplicate (query, tool_call) rows, validates tool call
format, and creates a stratified train/val/test split by tool type.

Stratified split (70/15/15) per tool type:
  - Test:  15% allocated first
  - Val:   15% allocated second
  - Train: remaining ~70% (used for LoRA fine-tuning)

Usage:
    python -m winner_solution.scripts.clean_data
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "winner"

VALID_TOOLS = {
    "get_telemetry", "get_crew_status", "get_module_status",
    "send_alert", "send_message", "schedule_maintenance",
    "activate_protocol", "control_system", "calculate_trajectory",
    "request_supply", "no_action",
}

VALID_MODULES = {"condor", "quetzal", "jaguar", "colibri", "vicuna", "tucan"}

VAL_RATIO = 0.15
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


def _validate_row(row: dict) -> list[str]:
    """Validate a single data row and return a list of issues (empty if clean).

    Args:
        row: Dict with 'id', 'query', 'tool_call' keys.

    Returns:
        List of issue description strings. Empty list means the row is valid.
    """
    issues = []
    tc = row["tool_call"].strip()
    tool_name = _extract_tool_name(tc)

    if tool_name not in VALID_TOOLS:
        issues.append(f"unrecognized_tool: {tool_name}")

    if tc != "no_action":
        params_match = re.match(r'\w+\((.+)\)', tc)
        if params_match:
            params = re.findall(r"(\w+)=(?:'([^']*)'|(\d+))", params_match.group(1))
            param_dict = {p[0]: p[1] or p[2] for p in params}
            if "module" in param_dict and param_dict["module"] not in VALID_MODULES:
                issues.append(f"invalid_module: {param_dict['module']}")
            if ", " in tc:
                issues.append("non_canonical: space after comma")
        elif tool_name in VALID_TOOLS and tool_name != "no_action":
            issues.append("no_params_parsed")

    if len(row["query"].split()) < 3:
        issues.append("query_too_short")

    return issues


def main() -> None:
    """Clean data.csv and produce train/val splits."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"Raw rows: {len(rows)}")

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    dup_count = 0
    for row in rows:
        key = (row["query"], row["tool_call"])
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(row)
    logger.info(f"Removed {dup_count} exact duplicates — {len(deduped)} remaining")

    clean: list[dict] = []
    rejected: list[dict] = []
    for row in deduped:
        issues = _validate_row(row)
        if issues:
            rejected.append({"id": row["id"], "issues": issues, "tool_call": row["tool_call"]})
        else:
            clean.append(row)

    if rejected:
        logger.warning(f"Rejected {len(rejected)} rows with validation issues:")
        for r in rejected[:10]:
            logger.warning(f"  [{r['id']}] {r['issues']} — {r['tool_call'][:60]}")
    else:
        logger.info("All rows passed validation.")

    logger.info(f"Clean rows: {len(clean)}")

    tool_dist = Counter(_extract_tool_name(r["tool_call"]) for r in clean)
    logger.info("Tool distribution:")
    for tool, count in tool_dist.most_common():
        logger.info(f"  {tool}: {count}")

    by_tool: dict[str, list[dict]] = defaultdict(list)
    for row in clean:
        by_tool[_extract_tool_name(row["tool_call"])].append(row)

    rng = np.random.default_rng(SEED)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for tool_name, tool_rows in sorted(by_tool.items()):
        n_test = max(1, int(len(tool_rows) * TEST_RATIO))
        n_val = max(1, int(len(tool_rows) * VAL_RATIO))
        indices = rng.permutation(len(tool_rows))
        test_indices = set(indices[:n_test].tolist())
        val_indices = set(indices[n_test:n_test + n_val].tolist())
        for i, row in enumerate(tool_rows):
            if i in test_indices:
                test_rows.append(row)
            elif i in val_indices:
                val_rows.append(row)
            else:
                train_rows.append(row)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    logger.info(f"Split — train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")

    for label, split_rows in [("Train", train_rows), ("Val", val_rows), ("Test", test_rows)]:
        dist = Counter(_extract_tool_name(r["tool_call"]) for r in split_rows)
        logger.info(f"{label} tool distribution:")
        for tool, count in dist.most_common():
            logger.info(f"  {tool}: {count}")

    fieldnames = ["id", "query", "tool_call"]

    train_path = OUTPUT_DIR / "train_data.csv"
    with open(train_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(train_rows)

    val_path = OUTPUT_DIR / "val_data.csv"
    with open(val_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(val_rows)

    test_path = OUTPUT_DIR / "test_data.csv"
    with open(test_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(test_rows)

    split_info = {
        "raw_rows": len(rows),
        "duplicates_removed": dup_count,
        "rejected_rows": len(rejected),
        "clean_rows": len(clean),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "seed": SEED,
    }
    split_path = OUTPUT_DIR / "data_split.json"
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    logger.info(f"Train saved: {train_path}")
    logger.info(f"Val saved: {val_path}")
    logger.info(f"Test saved: {test_path}")
    logger.info(f"Split info saved: {split_path}")


if __name__ == "__main__":
    main()
