"""Entry point for the ARTEMIS question generation pipeline (Phase 3).

Generates 2200 MASA operator queries with gold-standard tool calls.
Output: train/test/hidden splits under proyecto_artemis/datos_entrenamiento/
and proyecto_artemis/evaluacion/.

Usage:
    python app/scripts/run_question_pipeline.py
    python app/scripts/run_question_pipeline.py --tools get_telemetry send_alert
    python app/scripts/run_question_pipeline.py --target 50
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.llms.llm_nvidia import NvidiaLLM
from app.pipelines.question_pipeline.pipeline_facade import QuestionPipelineFacade
from app.pipelines.question_pipeline.settings import QuestionPipelineSettings
from app.skeleton.skeleton_loader import skeleton_loader


def setup_logging() -> None:
    """Configure loguru with both console and timestamped file sinks."""
    log_dir = Path(__file__).parent.parent.parent / "proyecto_artemis" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"question_pipeline_{timestamp}.log"

    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_file}")


def main(tool_filter: list[str] | None = None, target_per_tool: int = 200) -> None:
    """Run the full question generation pipeline.

    Args:
        tool_filter: Optional list of tool names to generate. None generates all.
        target_per_tool: Number of questions per tool. Default 200.
    """
    skeleton = skeleton_loader.load()

    llm = NvidiaLLM()

    pipeline_settings = QuestionPipelineSettings(
        target_per_tool=target_per_tool,
        tool_filter=tool_filter,
    )

    facade = QuestionPipelineFacade(
        llm=llm,
        skeleton=skeleton,
        pipeline_settings=pipeline_settings,
    )

    logger.info(
        f"Starting question pipeline: target_per_tool={target_per_tool}, "
        f"tool_filter={tool_filter or 'all'}"
    )

    result = facade.run()

    print("\n=== QUESTION PIPELINE RESULTS ===")
    print(f"  Total generated: {result.total_generated}")
    print(f"  Total failed:    {result.total_failed}")
    print(f"\n  Split counts:")
    for split, count in sorted(result.split_counts.items()):
        print(f"    {split:8s}: {count}")
    print(f"\n  Distribution by tool:")
    for tool, count in sorted(result.distribution.items()):
        print(f"    {tool:30s}: {count}")


if __name__ == "__main__":
    args = sys.argv[1:]

    tool_filter_arg: list[str] | None = None
    target_arg = 200

    if "--tools" in args:
        idx = args.index("--tools")
        remaining = args[idx + 1:]
        tool_filter_arg = [t for t in remaining if not t.startswith("--")]

    if "--target" in args:
        idx = args.index("--target")
        if idx + 1 < len(args):
            target_arg = int(args[idx + 1])

    setup_logging()
    main(tool_filter=tool_filter_arg, target_per_tool=target_arg)
