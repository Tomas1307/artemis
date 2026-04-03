import json
from pathlib import Path

from loguru import logger

from app.chain_methods.llm_document_reviewer import LLMDocumentReviewer
from app.chain_methods.llm_document_writer import LLMDocumentWriter
from app.llms.llm import BaseLLM
from app.pipelines.document_pipeline.schemas.generation_result import GenerationResult
from app.pipelines.document_pipeline.settings import DocumentPipelineSettings
from app.pipelines.document_pipeline.steps.step_01_load_registry import LoadRegistryStep
from app.pipelines.document_pipeline.steps.step_02_extract_context import ExtractContextStep
from app.pipelines.document_pipeline.steps.step_03_generate import GenerateDocumentStep
from app.pipelines.document_pipeline.steps.step_04_validate import ValidateDocumentStep
from app.pipelines.document_pipeline.steps.step_05_save import SaveDocumentStep
from app.skeleton.schemas.skeleton_schema import SkeletonSchema
from app.utils.skeleton_extractor import _build_reference_index


PROGRESS_FILENAME = "progress.json"
DOC_FILENAME = "doc.md"


class DocumentPipelineFacade:
    """Orchestrates the full document generation pipeline with resume support.

    This is the single entry point for generating MASA technical documents.
    It coordinates loading the registry, extracting skeleton context,
    generating documents via LLM, reviewing with parallel LangChain chains,
    correcting issues, and saving the output files.

    Supports resuming interrupted runs by tracking per-document progress
    in a JSON file. On restart, already-completed documents are skipped.

    Args:
        generator_llm: LLM provider for document generation (Devstral-2-123B).
        skeleton: Validated SkeletonSchema instance.
        pipeline_settings: Pipeline configuration. Uses defaults if not provided.
        reviewer_model: Model to use for review chains. Defaults to generator model.
        reviewer_api_key: API key for reviewer. Defaults to generator key.
    """

    def __init__(
        self,
        generator_llm: BaseLLM,
        skeleton: SkeletonSchema,
        pipeline_settings: DocumentPipelineSettings | None = None,
        reviewer_model: str | None = None,
        reviewer_api_key: str | None = None,
    ) -> None:
        self._settings = pipeline_settings or DocumentPipelineSettings()
        self._skeleton = skeleton
        self._reference_index = _build_reference_index(skeleton)
        self._output_dir = Path(self._settings.output_dir)

        self._step_load = LoadRegistryStep(self._settings.registry_path)
        self._step_extract = ExtractContextStep(skeleton)
        self._step_generate = GenerateDocumentStep(
            LLMDocumentWriter(generator_llm),
            max_retries=self._settings.max_retries,
        )
        self._step_validate = ValidateDocumentStep(
            LLMDocumentReviewer(
                model=reviewer_model,
                api_key=reviewer_api_key,
                max_correction_cycles=self._settings.max_correction_cycles,
                max_api_retries=self._settings.max_api_retries,
            )
        )
        self._step_save = SaveDocumentStep(self._settings.output_dir)

    def run(self, progress_callback=None) -> list[GenerationResult]:
        """Execute the full document generation pipeline with resume support.

        On each run, loads existing progress from disk. Documents already
        marked as 'success' or 'needs_review' are skipped. Failed documents
        are retried. Progress is saved after EACH document.

        Args:
            progress_callback: Optional callable(doc_id, index, total) invoked
                after each document is processed.

        Returns:
            List of GenerationResult instances for ALL documents (including
            previously completed ones loaded from progress).
        """
        specs = self._step_load.execute()

        if self._settings.doc_filter:
            specs = [s for s in specs if s.doc_id in self._settings.doc_filter]
            logger.info(f"Filtered to {len(specs)} documents: {self._settings.doc_filter}")

        results = []
        total = len(specs)
        skipped = 0

        for idx, spec in enumerate(specs, 1):
            existing = self._load_doc_progress(spec.doc_id)

            if existing and existing.get("status") in ("success", "needs_review"):
                doc_path = self._output_dir / spec.doc_id / DOC_FILENAME
                if doc_path.exists():
                    logger.info(
                        f"[{idx}/{total}] SKIP {spec.doc_id}: already {existing['status']} "
                        f"({existing.get('word_count', '?')}w)"
                    )
                    results.append(GenerationResult(
                        doc_id=existing["doc_id"],
                        title=existing["title"],
                        type=existing["type"],
                        status=existing["status"],
                        word_count=existing.get("word_count", 0),
                        file_path=str(doc_path),
                        attempts=existing.get("attempts", 1),
                    ))
                    skipped += 1
                    continue

            logger.info(f"[{idx}/{total}] Processing {spec.doc_id}: {spec.title}")

            result = self._process_single(spec)
            results.append(result)

            self._save_manifest(results)

            if progress_callback:
                progress_callback(spec.doc_id, idx, total)

        if skipped > 0:
            logger.info(f"Skipped {skipped} already-completed documents")

        self._log_summary(results)

        return results

    def _process_single(self, spec) -> GenerationResult:
        """Process a single document through all pipeline steps.

        Args:
            spec: DocumentSpec for the document to generate.

        Returns:
            GenerationResult tracking the outcome.
        """
        try:
            skeleton_context, required_facts = self._step_extract.execute(spec)

            document_text, attempts = self._step_generate.execute(spec, skeleton_context)
            word_count = len(document_text.split())

            validation = None
            final_text = document_text

            if not self._settings.skip_validation:
                final_text, validation = self._step_validate.execute(
                    doc_id=spec.doc_id,
                    document_text=document_text,
                    skeleton_context=skeleton_context,
                    reference_index=self._reference_index,
                    is_noise=(spec.type == "noise"),
                )
                word_count = len(final_text.split())

            if validation and not validation.passed:
                status = "needs_review"
            else:
                status = "success"

            result = GenerationResult(
                doc_id=spec.doc_id,
                title=spec.title,
                type=spec.type,
                status=status,
                word_count=word_count,
                file_path="",
                attempts=attempts,
                validation=validation,
            )

            file_path = self._step_save.execute(spec.doc_id, final_text, result)
            result.file_path = file_path

            return result

        except Exception as exc:
            logger.error(f"{spec.doc_id}: generation failed — {exc}")
            return GenerationResult(
                doc_id=spec.doc_id,
                title=spec.title,
                type=spec.type,
                status="failed",
                error=str(exc),
            )

    def _load_doc_progress(self, doc_id: str) -> dict | None:
        """Load progress for a single document from its per-doc folder.

        Args:
            doc_id: Document identifier to look up.

        Returns:
            Progress dictionary if found, None otherwise.
        """
        progress_path = self._output_dir / doc_id / PROGRESS_FILENAME
        if not progress_path.exists():
            return None

        try:
            return json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not load progress for {doc_id}: {exc}")
            return None

    def _save_manifest(self, results: list[GenerationResult]) -> None:
        """Save a manifest JSON mapping doc_ids to file paths and metadata.

        Args:
            results: List of all generation results (including resumed ones).
        """
        manifest = {}
        for r in results:
            if r.file_path:
                manifest[r.doc_id] = {
                    "title": r.title,
                    "type": r.type,
                    "file_path": r.file_path,
                    "word_count": r.word_count,
                    "status": r.status,
                    "validation_passed": r.validation.passed if r.validation else None,
                }

        manifest_path = self._output_dir / "documentos_masa.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _log_summary(self, results: list[GenerationResult]) -> None:
        """Log a summary of the pipeline run.

        Args:
            results: List of all generation results.
        """
        success = sum(1 for r in results if r.status == "success")
        needs_review = sum(1 for r in results if r.status == "needs_review")
        failed = sum(1 for r in results if r.status == "failed")
        total_words = sum(r.word_count for r in results)

        logger.info(
            f"Pipeline complete: {success} success, {needs_review} needs_review, "
            f"{failed} failed. Total corpus: {total_words} words."
        )
