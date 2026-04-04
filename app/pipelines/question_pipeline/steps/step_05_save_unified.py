import csv
import json
from pathlib import Path

from loguru import logger


class SaveUnifiedStep:
    """Step 5 — Save all generated questions to a single train.csv.

    Output format: id, query, tool_call — no difficulty, no doc_ids, no splits.
    Master Tomas handles splitting into train/test/hidden manually.

    Also saves an internal gold_standard.json with all metadata for instructor use.

    Args:
        output_dir: Directory to write output files.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)

    def execute(
        self,
        questions: list[dict],
    ) -> dict:
        """Save all questions to output files.

        Args:
            questions: List of dicts with at minimum 'question_id', 'query', 'tool_call'.
                May also contain 'tool_name', 'seed_type' (rag/direct) for internal use.

        Returns:
            Summary dict with counts and file paths.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self._output_dir / "train.csv"
        self._save_csv(questions, csv_path)

        gold_path = self._output_dir / "gold_standard.json"
        self._save_gold(questions, gold_path)

        distribution: dict[str, int] = {}
        seed_type_counts: dict[str, int] = {}
        for q in questions:
            tn = q.get("tool_name", "unknown")
            distribution[tn] = distribution.get(tn, 0) + 1
            st = q.get("seed_type", "unknown")
            seed_type_counts[st] = seed_type_counts.get(st, 0) + 1

        logger.info(
            f"Saved {len(questions)} questions to {csv_path}. "
            f"Seed types: {seed_type_counts}"
        )

        return {
            "total": len(questions),
            "csv_path": str(csv_path),
            "gold_path": str(gold_path),
            "distribution": distribution,
            "seed_type_counts": seed_type_counts,
        }

    def _save_csv(self, questions: list[dict], path: Path) -> None:
        """Save student-facing CSV with id, query, tool_call only.

        Args:
            questions: All generated questions.
            path: Output CSV path.
        """
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "query", "tool_call"])
            for q in questions:
                writer.writerow([q["question_id"], q["query"], q["tool_call"]])
        logger.info(f"Saved train CSV: {path} ({len(questions)} rows)")

    def _save_gold(self, questions: list[dict], path: Path) -> None:
        """Save full internal gold standard with all metadata.

        Args:
            questions: All generated questions with metadata.
            path: Output JSON path.
        """
        path.write_text(
            json.dumps(questions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved gold standard: {path}")
