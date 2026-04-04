import json
from pathlib import Path

from loguru import logger

from app.llms.llm import BaseLLM
from app.pipelines.question_pipeline.schemas.question_schema import GenerationBatchResult
from app.pipelines.question_pipeline.settings import QuestionPipelineSettings
from app.pipelines.question_pipeline.steps.step_01_generate_seeds import GenerateSeedsStep
from app.pipelines.question_pipeline.steps.step_02_assign_docs import AssignDocsStep
from app.pipelines.question_pipeline.steps.step_03_generate_queries import GenerateQueriesStep
from app.pipelines.question_pipeline.steps.step_04_validate_question import ValidateQuestionStep
from app.pipelines.question_pipeline.steps.step_05_save_outputs import SaveOutputsStep
from app.skeleton.schemas.skeleton_schema import SkeletonSchema


class QuestionPipelineFacade:
    """Orchestrates the full question generation pipeline.

    Single entry point for generating 2000 MASA operator questions with
    gold-standard tool calls. Coordinates five sequential steps:
    1. Deterministic seed generation from skeleton parameter combinations.
    2. Doc ID assignment validation against the generated corpus.
    3. LLM-based natural language query generation per seed.
    4. Format and quality validation of generated questions.
    5. Stratified output to train/test/hidden splits.

    The canonical tool call for each question is built deterministically in
    Step 1 and never modified by the LLM.

    Args:
        llm: LLM provider for query text generation.
        skeleton: Validated SkeletonSchema instance.
        pipeline_settings: Pipeline configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        llm: BaseLLM,
        skeleton: SkeletonSchema,
        pipeline_settings: QuestionPipelineSettings | None = None,
    ) -> None:
        self._settings = pipeline_settings or QuestionPipelineSettings()
        self._skeleton = skeleton

        valid_doc_ids = self._load_valid_doc_ids()

        self._step_seeds = GenerateSeedsStep(
            skeleton=skeleton,
            target_per_tool=self._settings.target_per_tool,
            random_seed=self._settings.random_seed,
        )
        self._step_docs = AssignDocsStep(valid_doc_ids=valid_doc_ids)
        self._step_queries = GenerateQueriesStep(
            llm=llm,
            max_retries=self._settings.max_retries,
        )
        self._step_validate = ValidateQuestionStep()
        self._step_save = SaveOutputsStep(output_base_dir=self._settings.output_base_dir)

    def run(self, progress_callback=None) -> GenerationBatchResult:
        """Execute the full question generation pipeline.

        Args:
            progress_callback: Optional callable(tool_name, generated, total) invoked
                after each tool's batch is processed.

        Returns:
            GenerationBatchResult with counts, distribution, and all questions.
        """
        logger.info("Question pipeline starting — generating seeds")
        seeds = self._step_seeds.execute()

        if self._settings.tool_filter:
            seeds = [s for s in seeds if s.tool_name in self._settings.tool_filter]
            logger.info(f"Filtered to {len(seeds)} seeds for tools: {self._settings.tool_filter}")

        logger.info("Validating doc assignments")
        seeds = self._step_docs.execute(seeds)

        logger.info(f"Generating queries for {len(seeds)} seeds")
        questions, failed_seeds = self._step_queries.execute(seeds)

        logger.info("Validating generated questions")
        questions, rejected = self._step_validate.execute(questions)

        logger.info("Saving outputs")
        result = self._step_save.execute(questions)
        result = result.model_copy(update={"total_failed": len(failed_seeds) + len(rejected)})

        self._log_summary(result)
        return result

    def _load_valid_doc_ids(self) -> set[str]:
        """Load the set of valid doc_ids from the document registry.

        Returns:
            Set of doc_id strings from documentos_masa.json.

        Raises:
            FileNotFoundError: If the registry file does not exist.
        """
        registry_path = Path(self._settings.valid_doc_ids_path)
        if not registry_path.exists():
            raise FileNotFoundError(f"Document registry not found: {registry_path}")
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        return set(data.keys())

    def _log_summary(self, result: GenerationBatchResult) -> None:
        """Log a summary of the pipeline run.

        Args:
            result: Final batch result.
        """
        logger.info(
            f"Pipeline complete: {result.total_generated} questions generated, "
            f"{result.total_failed} failed."
        )
        logger.info(f"Split: {result.split_counts}")
        logger.info(f"Distribution: {result.distribution}")
