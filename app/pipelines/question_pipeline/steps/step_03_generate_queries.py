from pathlib import Path
from typing import Optional

from loguru import logger

from app.llms.llm import BaseLLM
from app.pipelines.question_pipeline.schemas.question_schema import QuestionSpec
from app.pipelines.question_pipeline.schemas.seed_schema import QuestionSeed
from app.pipelines.question_pipeline.utils.checkpoint_writer import CheckpointWriter
from app.prompts.prompt_loader import prompt_loader


class GenerateQueriesStep:
    """Step 3 — Generate natural language operator queries from seeds via LLM.

    For each seed, builds a prompt describing the scenario, difficulty, and
    context facts, then calls the LLM to produce a realistic operator query.
    The canonical tool_call from the seed is passed through unchanged. Supports
    incremental checkpointing to resume interrupted runs.

    Args:
        llm: LLM provider instance for query generation.
        max_retries: Maximum generation attempts per seed on empty or invalid output.
        checkpoint_path: Optional path for JSONL checkpoint file. When set,
            each generated question is appended immediately and existing
            checkpoint records are loaded to skip already-completed seeds.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_retries: int = 3,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._checkpoint = CheckpointWriter(checkpoint_path) if checkpoint_path else None
        self._system_message = prompt_loader.get_system_message_by_type("query_generator")
        self._template = prompt_loader.get_prompt_template_by_type("query_generator")

    def execute(
        self,
        seeds: list[QuestionSeed],
        question_id_offset: int = 0,
    ) -> tuple[list[QuestionSpec], list[str]]:
        """Generate one QuestionSpec per seed with incremental checkpointing.

        If a checkpoint file exists, already-completed seeds are loaded and
        skipped. New generations are appended to the checkpoint immediately.

        Args:
            seeds: Validated seeds from Step 2.
            question_id_offset: Starting index for question IDs.

        Returns:
            Tuple of (generated QuestionSpec list, list of failed seed_ids).
        """
        generated: list[QuestionSpec] = []
        failed: list[str] = []
        total = len(seeds)

        completed_ids: set[str] = set()
        if self._checkpoint:
            existing = self._checkpoint.load_existing()
            for rec in existing:
                completed_ids.add(rec["seed_id"])
                generated.append(QuestionSpec(
                    question_id=rec["question_id"],
                    query=rec["query"],
                    tool_call=rec["tool_call"],
                    tool_name=rec["tool_name"],
                    doc_ids=rec.get("doc_ids", []),
                    difficulty=rec.get("difficulty", "easy"),
                    split=rec.get("split", "train"),
                ))
            if completed_ids:
                logger.info(f"Resumed from checkpoint: {len(completed_ids)} seeds already completed")

        for idx, seed in enumerate(seeds, 1):
            question_id = f"Q-{question_id_offset + idx:05d}"

            if seed.seed_id in completed_ids:
                continue

            logger.info(f"[{idx}/{total}] Generating query for {seed.seed_id} ({seed.tool_name}/{seed.difficulty.value})")

            query = self._generate_single(seed)
            if not query:
                logger.warning(f"{seed.seed_id}: query generation failed after {self._max_retries} attempts")
                failed.append(seed.seed_id)
                continue

            question = QuestionSpec(
                question_id=question_id,
                query=query,
                tool_call=seed.tool_call,
                tool_name=seed.tool_name,
                doc_ids=seed.doc_ids,
                difficulty=seed.difficulty,
                split=seed.split,
            )
            generated.append(question)

            if self._checkpoint:
                self._checkpoint.append({
                    "seed_id": seed.seed_id,
                    "question_id": question_id,
                    "query": query,
                    "tool_call": seed.tool_call,
                    "tool_name": seed.tool_name,
                    "doc_ids": seed.doc_ids,
                    "difficulty": seed.difficulty.value,
                    "split": seed.split.value,
                })

        logger.info(f"Query generation complete: {len(generated)} ok, {len(failed)} failed")
        return generated, failed

    def _generate_single(self, seed: QuestionSeed) -> str:
        """Generate a query for a single seed with retry logic.

        Args:
            seed: The seed to generate a query for.

        Returns:
            Generated query string, or empty string on failure.
        """
        prompt = self._build_prompt(seed)

        config = prompt_loader.get_config_by_type("query_generator")
        messages = [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._llm.generate(messages, **config)
                query = response.strip()
                if query and len(query.split()) >= 4:
                    return query
                logger.warning(f"{seed.seed_id}: attempt {attempt} returned short/empty output")
            except Exception as exc:
                logger.warning(f"{seed.seed_id}: attempt {attempt} error — {exc}")

        return ""

    def _build_prompt(self, seed: QuestionSeed) -> str:
        """Fill the prompt template with seed data.

        Args:
            seed: Seed containing tool, params, difficulty, and context facts.

        Returns:
            Formatted prompt string for the LLM.
        """
        context_block = "\n".join(f"- {fact}" for fact in seed.context_facts) if seed.context_facts else "N/A"
        params_block = ", ".join(f"{k}={v!r}" for k, v in seed.tool_params.items()) if seed.tool_params else "none"

        return self._template.format(
            tool_name=seed.tool_name,
            tool_call=seed.tool_call,
            difficulty=seed.difficulty.value,
            phrasing_index=seed.phrasing_index,
            params=params_block,
            context_facts=context_block,
        )
