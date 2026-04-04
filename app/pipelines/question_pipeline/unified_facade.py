import re
from pathlib import Path

from loguru import logger

from app.llms.llm import BaseLLM
from app.pipelines.question_pipeline.steps.step_01_generate_rag_seeds import GenerateRagSeedsStep
from app.pipelines.question_pipeline.steps.step_01_generate_seeds import GenerateSeedsStep
from app.pipelines.question_pipeline.steps.step_03_generate_queries import GenerateQueriesStep
from app.pipelines.question_pipeline.steps.step_03_generate_rag_queries import GenerateRagQueriesStep
from app.pipelines.question_pipeline.steps.step_05_save_unified import SaveUnifiedStep
from app.skeleton.schemas.skeleton_schema import SkeletonSchema


CANONICAL_PATTERN = re.compile(
    r"^(get_telemetry|get_crew_status|get_module_status|send_alert|send_message"
    r"|schedule_maintenance|activate_protocol|control_system|calculate_trajectory"
    r"|request_supply|no_action)"
    r"(\(.*\))?$"
)

DIRECT_TOOLS = [
    "get_telemetry",
    "get_crew_status",
    "get_module_status",
    "send_message",
    "schedule_maintenance",
    "control_system",
    "calculate_trajectory",
    "request_supply",
    "no_action",
]


class UnifiedQuestionPipelineSettings:
    """Configuration for the unified question generation pipeline.

    Attributes:
        rag_target: Target number of RAG-dependent questions.
        direct_target_per_tool: Target questions per direct tool.
        rag_readings_per_combo: Sensor reading variants per (protocol, module) pair.
        output_dir: Output directory for train.csv and gold_standard.json.
        random_seed: Seed for reproducible generation.
        max_retries: Max LLM retries per seed.
    """

    def __init__(
        self,
        rag_target: int = 2000,
        direct_target_per_tool: int = 120,
        rag_readings_per_combo: int = 15,
        output_dir: str = "proyecto_artemis/datos_entrenamiento",
        random_seed: int = 42,
        max_retries: int = 3,
    ) -> None:
        self.rag_target = rag_target
        self.direct_target_per_tool = direct_target_per_tool
        self.rag_readings_per_combo = rag_readings_per_combo
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.max_retries = max_retries


class UnifiedQuestionPipeline:
    """Orchestrates generation of both RAG-dependent and direct questions.

    Produces a single train.csv with id, query, tool_call for all questions.
    RAG questions use the rag_query_generator prompt (sensor readings, severity
    from docs). Direct questions use the query_generator prompt (all params
    inferable from query text).

    Args:
        llm: LLM provider for query generation.
        skeleton: Validated SkeletonSchema instance.
        settings: Pipeline configuration.
    """

    def __init__(
        self,
        llm: BaseLLM,
        skeleton: SkeletonSchema,
        settings: UnifiedQuestionPipelineSettings | None = None,
    ) -> None:
        self._llm = llm
        self._skeleton = skeleton
        self._settings = settings or UnifiedQuestionPipelineSettings()

    def run(self) -> dict:
        """Execute the full unified question generation pipeline.

        Returns:
            Summary dict with total counts, distribution, and file paths.
        """
        logger.info("=== UNIFIED QUESTION PIPELINE START ===")

        logger.info("Phase A: Generating RAG-dependent questions")
        rag_questions = self._generate_rag_questions()

        logger.info("Phase B: Generating direct questions")
        direct_questions = self._generate_direct_questions(id_offset=len(rag_questions))

        all_questions = rag_questions + direct_questions
        logger.info(f"Total generated: {len(all_questions)} ({len(rag_questions)} RAG + {len(direct_questions)} direct)")

        logger.info("Phase C: Validation")
        valid_questions = self._validate(all_questions)

        logger.info("Phase D: Saving outputs")
        save_step = SaveUnifiedStep(output_dir=self._settings.output_dir)
        result = save_step.execute(valid_questions)

        logger.info(f"=== PIPELINE COMPLETE: {result['total']} questions saved ===")
        return result

    def _generate_rag_questions(self) -> list[dict]:
        """Generate RAG-dependent questions from protocol thresholds.

        Returns:
            List of question dicts.
        """
        seed_step = GenerateRagSeedsStep(
            target_total=self._settings.rag_target,
            random_seed=self._settings.random_seed,
            readings_per_combo=self._settings.rag_readings_per_combo,
        )
        seeds = seed_step.execute()

        checkpoint_path = Path(self._settings.output_dir) / ".checkpoint_rag.jsonl"
        query_step = GenerateRagQueriesStep(
            llm=self._llm,
            max_retries=self._settings.max_retries,
            checkpoint_path=checkpoint_path,
        )
        generated, failed = query_step.execute(seeds, id_prefix="Q", id_offset=0)

        logger.info(f"RAG generation: {len(generated)} ok, {len(failed)} failed")

        return [
            {
                "question_id": q.question_id,
                "query": q.query,
                "tool_call": q.tool_call,
                "tool_name": q.tool_name,
                "seed_type": "rag",
            }
            for q in generated
        ]

    def _generate_direct_questions(self, id_offset: int = 0) -> list[dict]:
        """Generate direct questions where all params are inferable from query.

        Uses the existing seed generator and query generator, filtering to
        tools that don't require RAG (everything except send_alert severity
        and activate_protocol protocol_id).

        Args:
            id_offset: Starting offset for question IDs.

        Returns:
            List of question dicts.
        """
        seed_step = GenerateSeedsStep(
            skeleton=self._skeleton,
            target_per_tool=self._settings.direct_target_per_tool,
            random_seed=self._settings.random_seed + 1000,
        )
        all_seeds = seed_step.execute()

        direct_seeds = [s for s in all_seeds if s.tool_name in DIRECT_TOOLS]
        logger.info(f"Direct seeds: {len(direct_seeds)} (filtered from {len(all_seeds)} total)")

        checkpoint_path = Path(self._settings.output_dir) / ".checkpoint_direct.jsonl"
        query_step = GenerateQueriesStep(
            llm=self._llm,
            max_retries=self._settings.max_retries,
            checkpoint_path=checkpoint_path,
        )
        questions, failed = query_step.execute(direct_seeds, question_id_offset=id_offset)

        logger.info(f"Direct generation: {len(questions)} ok, {len(failed)} failed")

        return [
            {
                "question_id": q.question_id,
                "query": q.query,
                "tool_call": q.tool_call,
                "tool_name": q.tool_name,
                "seed_type": "direct",
            }
            for q in questions
        ]

    def _validate(self, questions: list[dict]) -> list[dict]:
        """Validate generated questions for format correctness.

        Checks:
        - Tool call matches canonical format
        - No spaces after commas in tool call
        - Query minimum word count
        - No tool call leakage in query text

        Args:
            questions: All generated questions.

        Returns:
            Filtered list of valid questions.
        """
        valid = []
        rejected = 0

        for q in questions:
            reason = self._check_question(q)
            if reason:
                logger.debug(f"{q['question_id']}: rejected — {reason}")
                rejected += 1
            else:
                valid.append(q)

        logger.info(f"Validation: {len(valid)} valid, {rejected} rejected")
        return valid

    def _check_question(self, q: dict) -> str:
        """Validate a single question.

        Args:
            q: Question dict.

        Returns:
            Reason string if invalid, empty string if valid.
        """
        tc = q["tool_call"]
        if not CANONICAL_PATTERN.match(tc):
            return f"tool_call format invalid: {tc!r}"

        if ", " in tc:
            return f"tool_call contains space after comma: {tc!r}"

        word_count = len(q["query"].split())
        if word_count < 5:
            return f"query too short ({word_count} words)"
        if word_count > 60:
            return f"query too long ({word_count} words)"

        leakage_tokens = ["get_telemetry(", "send_alert(", "activate_protocol(", "no_action"]
        if any(token in q["query"] for token in leakage_tokens):
            return f"query contains tool call leakage"

        return ""
