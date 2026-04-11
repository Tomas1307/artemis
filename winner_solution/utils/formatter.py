"""Output formatting and context building for winner RAG solution."""

import re

from loguru import logger


TOOL_NAMES = [
    "get_telemetry", "get_crew_status", "get_module_status",
    "send_alert", "send_message", "schedule_maintenance",
    "activate_protocol", "control_system", "calculate_trajectory",
    "request_supply", "no_action",
]

TOOL_PARAM_ORDER: dict[str, list[str]] = {
    "get_telemetry": ["module", "metric", "timeframe_hours"],
    "get_crew_status": ["module", "info"],
    "get_module_status": ["module", "system"],
    "send_alert": ["module", "severity", "reason"],
    "send_message": ["recipient", "priority"],
    "schedule_maintenance": ["module", "task", "priority"],
    "activate_protocol": ["protocol_id", "scope"],
    "control_system": ["module", "system", "action"],
    "calculate_trajectory": ["maneuver", "urgency"],
    "request_supply": ["category", "urgency"],
    "no_action": [],
}

NUMERIC_PARAMS: set[str] = {"timeframe_hours"}

ACCENT_MAP: dict[str, str] = {
    "cóndor": "condor",
    "córdor": "condor",
    "colibrí": "colibri",
    "colibří": "colibri",
    "vicuña": "vicuna",
    "tucán": "tucan",
}


def build_rich_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as structured context for the decoder.

    Each chunk includes its document path (topic > subtopic > keypoint)
    and full content with no truncation.

    Args:
        chunks: List of chunk dicts with doc_id, topic, subtopic, keypoint, content.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{chunk['doc_id']}] {chunk.get('topic', '')} > {chunk.get('subtopic', '')}"
        if chunk.get("keypoint"):
            header += f" > {chunk['keypoint']}"
        parts.append(f"[Chunk {i}] {header}\n{chunk.get('content', '')}")
    return "\n\n".join(parts)


def extract_tool_call(text: str) -> str:
    """Extract a tool call from raw LLM output using positional first match.

    Finds all recognized tool names in the text ordered by their position,
    returns the first one. Logs a warning if multiple tool calls are detected,
    as that indicates the prompt needs improvement.

    Args:
        text: Raw LLM generation output.

    Returns:
        Canonical tool call string, or 'no_action' if none found.
    """
    matches = []
    for tool_name in TOOL_NAMES:
        if tool_name == "no_action":
            if re.search(r'\bno_action\b', text, re.IGNORECASE):
                pos = text.lower().find("no_action")
                matches.append((pos, "no_action", None))
            continue
        pattern = rf'{tool_name}\([^)]*\)'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((m.start(), tool_name, m.group(0)))

    if not matches:
        return "no_action"

    matches.sort(key=lambda x: x[0])

    if len(matches) > 1:
        logger.warning(f"Multiple tool calls detected — prompt needs improvement: {[m[2] or m[1] for m in matches]}")

    _, tool_name, raw = matches[0]
    if raw is None:
        return "no_action"
    return canonicalize_tool_call(raw)


def canonicalize_tool_call(raw: str) -> str:
    """Parse a tool call and rebuild it in strict canonical format.

    Extracts all parameter key=value pairs regardless of order, then
    rebuilds using the defined canonical parameter order. Applies accent
    normalization for module name values. Ensures single quotes for string
    values, unquoted numeric values, no spaces after commas, all lowercase.

    Args:
        raw: Raw tool call string from LLM output.

    Returns:
        Canonical tool call string.
    """
    raw = raw.strip().lower()

    for accented, canonical in ACCENT_MAP.items():
        raw = raw.replace(accented, canonical)

    tool_match = re.match(r'(\w+)\s*\(', raw)
    if not tool_match:
        return raw

    tool_name = tool_match.group(1)
    param_order = TOOL_PARAM_ORDER.get(tool_name)
    if param_order is None:
        return raw

    inner = raw[tool_match.end():]
    inner = inner.rstrip(")")

    param_pattern = re.compile(r'(\w+)\s*=\s*(?:\'([^\']*)\'|"([^"]*)"|([^\s,)]+))')
    parsed: dict[str, str] = {}
    for m in param_pattern.finditer(inner):
        key = m.group(1)
        value = m.group(2) or m.group(3) or m.group(4) or ""
        parsed[key] = value

    parts = []
    for param in param_order:
        if param not in parsed:
            continue
        value = parsed[param]
        if param in NUMERIC_PARAMS:
            parts.append(f"{param}={value}")
        else:
            parts.append(f"{param}='{value}'")

    if not parts:
        return tool_name

    return f"{tool_name}({','.join(parts)})"
