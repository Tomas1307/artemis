import json
from pathlib import Path

from loguru import logger


class CheckpointWriter:
    """Incremental JSONL checkpoint writer for question generation.

    Appends each generated question as a single JSON line to a JSONL file
    immediately after generation. Supports loading existing checkpoints to
    resume interrupted runs without re-generating already completed items.

    Args:
        checkpoint_path: Path to the JSONL checkpoint file.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self._path = checkpoint_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        """Append a single question record to the checkpoint file.

        Args:
            record: Dict with at minimum question_id, query, tool_call.
        """
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_existing(self) -> list[dict]:
        """Load all records from an existing checkpoint file.

        Returns:
            List of dicts from the checkpoint, or empty list if file missing.
        """
        if not self._path.exists():
            return []

        records = []
        with self._path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Checkpoint line {line_num} corrupt, skipping")

        logger.info(f"Loaded {len(records)} existing records from {self._path}")
        return records

    def get_completed_seed_ids(self) -> set[str]:
        """Get set of seed_ids already in checkpoint for skip-on-resume.

        Returns:
            Set of seed_id strings from existing checkpoint records.
        """
        records = self.load_existing()
        return {r["seed_id"] for r in records if "seed_id" in r}

    def clear(self) -> None:
        """Delete the checkpoint file to start fresh.

        Use when a full run completes successfully and the checkpoint
        is no longer needed.
        """
        if self._path.exists():
            self._path.unlink()
            logger.info(f"Cleared checkpoint: {self._path}")
