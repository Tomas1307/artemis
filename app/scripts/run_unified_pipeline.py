"""Entry point for the unified ARTEMIS question generation pipeline.

Generates both RAG-dependent and direct questions, outputting a single
train.csv with id, query, tool_call.

Usage:
    python app/scripts/run_unified_pipeline.py
    python app/scripts/run_unified_pipeline.py --rag-target 100 --direct-per-tool 20
    python app/scripts/run_unified_pipeline.py --rag-only
    python app/scripts/run_unified_pipeline.py --direct-only
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.llms.llm_nvidia import NvidiaLLM
from app.pipelines.question_pipeline.unified_facade import (
    UnifiedQuestionPipeline,
    UnifiedQuestionPipelineSettings,
)
from app.skeleton.skeleton_loader import skeleton_loader


def setup_logging() -> None:
    """Configure loguru with console and timestamped file sinks."""
    log_dir = Path(__file__).parent.parent.parent / "proyecto_artemis" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"unified_pipeline_{timestamp}.log"

    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_file}")


def main() -> None:
    """Parse args and run the unified question generation pipeline."""
    args = sys.argv[1:]

    rag_target = 2000
    direct_per_tool = 120
    rag_readings = 15

    if "--rag-target" in args:
        idx = args.index("--rag-target")
        rag_target = int(args[idx + 1])

    if "--direct-per-tool" in args:
        idx = args.index("--direct-per-tool")
        direct_per_tool = int(args[idx + 1])

    if "--rag-only" in args:
        direct_per_tool = 0

    if "--direct-only" in args:
        rag_target = 0

    if "--readings" in args:
        idx = args.index("--readings")
        rag_readings = int(args[idx + 1])

    setup_logging()

    skeleton = skeleton_loader.load()
    llm = NvidiaLLM()

    settings = UnifiedQuestionPipelineSettings(
        rag_target=rag_target,
        direct_target_per_tool=direct_per_tool,
        rag_readings_per_combo=rag_readings,
    )

    pipeline = UnifiedQuestionPipeline(
        llm=llm,
        skeleton=skeleton,
        settings=settings,
    )

    logger.info(
        f"Starting unified pipeline: rag_target={rag_target}, "
        f"direct_per_tool={direct_per_tool}, readings={rag_readings}"
    )

    result = pipeline.run()

    print("\n=== UNIFIED PIPELINE RESULTS ===")
    print(f"  Total questions: {result['total']}")
    print(f"  CSV: {result['csv_path']}")
    print(f"\n  By seed type:")
    for st, count in sorted(result["seed_type_counts"].items()):
        print(f"    {st:10s}: {count}")
    print(f"\n  By tool:")
    for tool, count in sorted(result["distribution"].items()):
        print(f"    {tool:30s}: {count}")


if __name__ == "__main__":
    main()
