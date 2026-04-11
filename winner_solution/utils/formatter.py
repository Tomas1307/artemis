"""Output formatting and context building for winner RAG solution."""

import re


TOOL_NAMES = [
    "get_telemetry", "get_crew_status", "get_module_status",
    "send_alert", "send_message", "schedule_maintenance",
    "activate_protocol", "control_system", "calculate_trajectory",
    "request_supply", "no_action",
]

MODULES = ["condor", "quetzal", "jaguar", "colibri", "vicuna", "tucan"]

ACCENT_MAP = {
    "cóndor": "condor", "córdor": "condor",
    "colibrí": "colibri", "colibří": "colibri",
    "vicuña": "vicuna",
    "tucán": "tucan",
}


def build_rich_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as structured context for the decoder.

    Each chunk includes its document path (topic > subtopic) and content,
    giving the decoder hierarchical context for better reasoning.

    Args:
        chunks: List of chunk dicts with topic, subtopic, keypoint, content.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{chunk['doc_id']}] {chunk.get('topic', '')} > {chunk.get('subtopic', '')}"
        if chunk.get("keypoint"):
            header += f" > {chunk['keypoint']}"
        content = chunk.get("content", "")[:500]
        parts.append(f"[Chunk {i}] {header}\n{content}")
    return "\n\n".join(parts)


def extract_tool_call(text: str) -> str:
    """Extract and normalize a tool call from raw LLM output.

    Args:
        text: Raw LLM generation output.

    Returns:
        Normalized tool call in canonical format, or 'no_action'.
    """
    for tool_name in TOOL_NAMES:
        pattern = rf'{tool_name}\([^)]*\)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_tool_call(match.group(0))

    if "no_action" in text.lower():
        return "no_action"

    return "no_action"


def normalize_tool_call(tool_call: str) -> str:
    """Normalize a tool call to strict canonical format.

    Applies: no spaces after commas, single quotes, lowercase enums,
    numeric values unquoted, accent normalization for module names.

    Args:
        tool_call: Raw tool call string.

    Returns:
        Canonical tool call string.
    """
    tool_call = tool_call.replace(", ", ",")
    tool_call = tool_call.replace('"', "'")
    tool_call = tool_call.lower()

    for accented, canonical in ACCENT_MAP.items():
        tool_call = tool_call.replace(accented, canonical)

    tool_call = re.sub(r"timeframe_hours='(\d+)'", r"timeframe_hours=\1", tool_call)

    return tool_call
