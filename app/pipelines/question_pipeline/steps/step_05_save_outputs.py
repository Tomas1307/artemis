import csv
import json
from pathlib import Path

from loguru import logger

from app.pipelines.question_pipeline.schemas.question_schema import GenerationBatchResult, QuestionSpec, QuestionSplit


class SaveOutputsStep:
    """Step 5 — Persist generated questions to the proyecto_artemis deliverable structure.

    Writes three output files:
    - datos_entrenamiento/train_tool_calls.json — train split questions
    - evaluacion/test_queries.csv — test split questions (Kaggle format)
    - evaluacion/sample_submission.csv — format reference with first 5 answers

    Also writes a hidden_reserve.json alongside train for instructor use.
    All output directories are created if absent.

    Args:
        output_base_dir: Base path for proyecto_artemis deliverables.
            Defaults to proyecto_artemis/ relative to project root.
    """

    def __init__(self, output_base_dir: str) -> None:
        self._base = Path(output_base_dir)
        self._train_dir = self._base / "datos_entrenamiento"
        self._eval_dir = self._base / "evaluacion"

    def execute(self, questions: list[QuestionSpec]) -> GenerationBatchResult:
        """Split and save all questions to their respective output files.

        Args:
            questions: Validated questions from Step 4.

        Returns:
            GenerationBatchResult with counts and the full question list.
        """
        self._train_dir.mkdir(parents=True, exist_ok=True)
        self._eval_dir.mkdir(parents=True, exist_ok=True)

        train = [q for q in questions if q.split == QuestionSplit.train]
        test = [q for q in questions if q.split == QuestionSplit.test]
        hidden = [q for q in questions if q.split == QuestionSplit.hidden]

        self._save_train(train)
        self._save_test_csv(test)
        self._save_sample_submission(test[:5])
        self._save_hidden(hidden)

        distribution: dict[str, int] = {}
        for q in questions:
            distribution[q.tool_name] = distribution.get(q.tool_name, 0) + 1

        result = GenerationBatchResult(
            total_generated=len(questions),
            total_failed=0,
            distribution=distribution,
            split_counts={
                "train": len(train),
                "test": len(test),
                "hidden": len(hidden),
            },
            questions=questions,
        )

        logger.info(
            f"Saved: {len(train)} train, {len(test)} test, {len(hidden)} hidden. "
            f"Total: {len(questions)}"
        )
        return result

    def _save_train(self, questions: list[QuestionSpec]) -> None:
        """Save train questions as JSON array.

        Format per entry: query, tool_call, tool_name, doc_ids, difficulty.

        Args:
            questions: Train split questions.
        """
        records = [
            {
                "query": q.query,
                "tool_call": q.tool_call,
                "tool_name": q.tool_name,
                "doc_ids": q.doc_ids,
                "difficulty": q.difficulty.value,
            }
            for q in questions
        ]
        path = self._train_dir / "train_tool_calls.json"
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved train: {path} ({len(records)} examples)")

    def _save_test_csv(self, questions: list[QuestionSpec]) -> None:
        """Save test questions as CSV in Kaggle submission format (ID, query only).

        Students do not receive the answers. The gold standard is kept separately.

        Args:
            questions: Test split questions.
        """
        path = self._eval_dir / "test_queries.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "query"])
            for q in questions:
                writer.writerow([q.question_id, q.query])
        logger.info(f"Saved test queries: {path} ({len(questions)} rows)")

        gold_path = self._eval_dir / "test_gold_standard.json"
        gold = [
            {
                "question_id": q.question_id,
                "tool_call": q.tool_call,
                "tool_name": q.tool_name,
                "doc_ids": q.doc_ids,
                "difficulty": q.difficulty.value,
            }
            for q in questions
        ]
        gold_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved gold standard: {gold_path}")

    def _save_sample_submission(self, sample_questions: list[QuestionSpec]) -> None:
        """Save a sample_submission.csv showing the expected format.

        Uses the first 5 test questions with their real answers visible,
        so students understand the output format.

        Args:
            sample_questions: A small subset of test questions (typically 5).
        """
        path = self._eval_dir / "sample_submission.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "answer"])
            for q in sample_questions:
                writer.writerow([q.question_id, q.tool_call])
        logger.info(f"Saved sample submission: {path}")

    def _save_hidden(self, questions: list[QuestionSpec]) -> None:
        """Save hidden reserve questions as JSON for instructor evaluation.

        Args:
            questions: Hidden split questions.
        """
        path = self._eval_dir / "hidden_reserve.json"
        records = [
            {
                "question_id": q.question_id,
                "query": q.query,
                "tool_call": q.tool_call,
                "tool_name": q.tool_name,
                "doc_ids": q.doc_ids,
                "difficulty": q.difficulty.value,
            }
            for q in questions
        ]
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved hidden reserve: {path} ({len(records)} questions)")
