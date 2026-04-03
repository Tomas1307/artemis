import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.config import settings
from app.llms.llm_nvidia import NvidiaLLM
from app.pipelines.document_pipeline.pipeline_facade import DocumentPipelineFacade
from app.pipelines.document_pipeline.settings import DocumentPipelineSettings
from app.skeleton.skeleton_loader import skeleton_loader


def setup_logging():
    """Configure loguru with both console and file sinks.

    Creates a timestamped log file in proyecto_artemis/logs/ for each run.
    Console shows INFO+, file captures DEBUG+ for full traceability.
    """
    log_dir = Path(__file__).parent.parent.parent / "proyecto_artemis" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_run_{timestamp}.log"

    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_file}")


USAGE = """
Usage: python run_document_pipeline.py [options] [doc_id ...]

Options:
  --skip-validation   Skip the review-correct validation step
  --force             Force regeneration of specified docs (ignores progress)
  --status            Show current progress without generating anything

Examples:
  python run_document_pipeline.py                          # Generate all, resume if interrupted
  python run_document_pipeline.py MASA-DOC-003             # Generate only MASA-DOC-003
  python run_document_pipeline.py --force MASA-DOC-003     # Force regenerate MASA-DOC-003
  python run_document_pipeline.py --skip-validation        # Generate all without review
  python run_document_pipeline.py --status                 # Show progress report
"""


def show_status():
    """Display current generation progress by scanning per-doc folders."""
    import json
    base_dir = (
        Path(__file__).parent.parent.parent / "proyecto_artemis" / "base_conocimiento"
    )
    if not base_dir.exists():
        print("No output directory found. Pipeline has not been run yet.")
        return

    success = []
    needs_review = []
    failed = []
    total_words = 0

    for doc_dir in sorted(base_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        progress_file = doc_dir / "progress.json"
        if not progress_file.exists():
            continue

        progress = json.loads(progress_file.read_text(encoding="utf-8"))
        doc_id = progress.get("doc_id", doc_dir.name)
        status = progress.get("status", "unknown")
        word_count = progress.get("word_count", 0)
        total_words += word_count

        if status == "success":
            success.append((doc_id, progress))
        elif status == "needs_review":
            needs_review.append((doc_id, progress))
        elif status == "failed":
            failed.append((doc_id, progress))

    total = len(success) + len(needs_review) + len(failed)
    if total == 0:
        print("No documents generated yet.")
        return

    print(f"\n=== GENERATION PROGRESS ({total}/67 docs) ===")
    print(f"  Success:      {len(success)}")
    print(f"  Needs review: {len(needs_review)}")
    print(f"  Failed:       {len(failed)}")
    print(f"  Total words:  {total_words}")

    if needs_review:
        print(f"\n  Needs review:")
        for doc_id, p in needs_review:
            print(f"    - {doc_id}: {p.get('title', '?')}")

    if failed:
        print(f"\n  Failed:")
        for doc_id, p in failed:
            err = p.get("error", "unknown")
            print(f"    - {doc_id}: {err[:80]}")

    print()


def clear_progress_for_docs(doc_ids: list[str]):
    """Remove specific doc folders to force regeneration."""
    import shutil
    base_dir = (
        Path(__file__).parent.parent.parent / "proyecto_artemis" / "base_conocimiento"
    )
    for doc_id in doc_ids:
        doc_dir = base_dir / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
            print(f"Cleared folder for {doc_id}")
        else:
            print(f"No folder found for {doc_id} (nothing to clear)")


def main(doc_filter: list[str] | None = None, skip_validation: bool = False):
    """Run the document generation pipeline.

    Args:
        doc_filter: Optional list of doc_ids to generate. None generates all.
        skip_validation: If True, skip the review-correct validation step.
    """
    skeleton = skeleton_loader.load()

    generator_llm = NvidiaLLM(
        model=settings.NVIDIA_MODEL,
        api_key=settings.NVIDIA_API_KEY,
        temperature=0.2,
        max_tokens=16384,
        top_p=0.95,
    )

    pipeline_settings = DocumentPipelineSettings(
        doc_filter=doc_filter,
        skip_validation=skip_validation,
    )

    facade = DocumentPipelineFacade(
        generator_llm=generator_llm,
        skeleton=skeleton,
        pipeline_settings=pipeline_settings,
        reviewer_model=settings.NVIDIA_MODEL,
        reviewer_api_key=settings.NVIDIA_API_KEY,
    )

    results = facade.run()

    print("\n=== PIPELINE RESULTS ===")
    for r in results:
        icon = {"success": "OK", "needs_review": "!!", "failed": "XX"}.get(r.status, "??")
        print(f"[{icon}] {r.doc_id}: {r.title} ({r.word_count}w, {r.status})")

    needs_review = [r for r in results if r.status == "needs_review"]
    if needs_review:
        print(f"\n!! {len(needs_review)} documents need manual review. See NEEDS_REVIEW.md")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--status" in args:
        show_status()
        sys.exit(0)

    skip_val = "--skip-validation" in args
    force = "--force" in args
    doc_ids = [a for a in args if not a.startswith("--")]

    setup_logging()

    if force and doc_ids:
        clear_progress_for_docs(doc_ids)

    main(doc_filter=doc_ids or None, skip_validation=skip_val)
