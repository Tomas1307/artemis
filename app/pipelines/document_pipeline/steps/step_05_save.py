import json
from pathlib import Path

from loguru import logger

from app.pipelines.document_pipeline.schemas.generation_result import GenerationResult


class SaveDocumentStep:
    """Saves a generated document and its metadata into a per-doc folder.

    Creates a folder per doc_id inside the output directory containing:
    - doc.md: the generated document
    - progress.json: status, word count, attempts, validation info
    - needs_revision.md: only created if the document has unresolved issues

    Args:
        output_dir: Root directory for the knowledge base.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, doc_id: str, document_text: str, result: GenerationResult) -> str:
        """Save a document, its progress, and revision notes to a per-doc folder.

        Args:
            doc_id: Document identifier used as the folder name.
            document_text: Full markdown content to write.
            result: GenerationResult with status and validation details.

        Returns:
            Absolute path to the saved doc.md file as a string.
        """
        doc_dir = self._output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        doc_path = doc_dir / "doc.md"
        doc_path.write_text(document_text, encoding="utf-8")

        self._save_progress(doc_dir, result)
        self._handle_revision(doc_dir, result)

        logger.info(f"{doc_id}: saved to {doc_dir}/")
        return str(doc_path)

    def _save_progress(self, doc_dir: Path, result: GenerationResult) -> None:
        """Write per-doc progress.json with status and metadata.

        Args:
            doc_dir: The document's folder path.
            result: GenerationResult to serialize.
        """
        progress = {
            "doc_id": result.doc_id,
            "title": result.title,
            "type": result.type,
            "status": result.status,
            "word_count": result.word_count,
            "attempts": result.attempts,
        }

        if result.validation:
            progress["validation"] = {
                "passed": result.validation.passed,
                "facts_checked": result.validation.facts_checked,
                "facts_present": result.validation.facts_present,
                "facts_missing": result.validation.facts_missing,
            }

        if result.error:
            progress["error"] = result.error

        progress_path = doc_dir / "progress.json"
        progress_path.write_text(
            json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _handle_revision(self, doc_dir: Path, result: GenerationResult) -> None:
        """Create or remove needs_revision.md based on document status.

        If the document needs review, creates needs_revision.md listing the
        remaining issues. If the document is clean, removes any existing
        needs_revision.md from a previous run.

        Args:
            doc_dir: The document's folder path.
            result: GenerationResult with validation details.
        """
        revision_path = doc_dir / "needs_revision.md"

        if result.status == "needs_review" and result.validation:
            lines = [
                f"# {result.doc_id}: Needs Revision",
                "",
                f"**Title:** {result.title}",
                f"**Word count:** {result.word_count}",
                f"**Attempts:** {result.attempts}",
                "",
                "## Remaining Issues",
                "",
            ]
            for issue in result.validation.facts_missing:
                lines.append(f"- {issue}")

            revision_path.write_text("\n".join(lines), encoding="utf-8")
            logger.warning(f"{result.doc_id}: needs_revision.md created")
        else:
            if revision_path.exists():
                revision_path.unlink()
