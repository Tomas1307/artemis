"""Output formatting utilities for baseline RAG solution."""

import re


TOOL_NAMES = [
    "get_telemetry", "get_crew_status", "get_module_status",
    "send_alert", "send_message", "schedule_maintenance",
    "activate_protocol", "control_system", "calculate_trajectory",
    "request_supply", "no_action",
]


def extract_tool_call(text: str) -> str:
    """Extract a tool call from raw LLM output.

    Searches for any recognized tool name followed by parenthesized arguments.
    Falls back to 'no_action' if no tool pattern is found.

    Args:
        text: Raw LLM generation output.

    Returns:
        Extracted tool call string in canonical format, or 'no_action'.
    """
    for tool_name in TOOL_NAMES:
        pattern = rf'{tool_name}\([^)]*\)'
        match = re.search(pattern, text)
        if match:
            return normalize_tool_call(match.group(0))

    if "no_action" in text.lower():
        return "no_action"

    return "no_action"


def normalize_tool_call(tool_call: str) -> str:
    """Normalize a tool call to canonical format.

    Applies rules: no spaces after commas, single quotes for strings,
    all lowercase enum values, numeric values unquoted.

    Args:
        tool_call: Raw tool call string.

    Returns:
        Normalized tool call string.
    """
    tool_call = tool_call.replace(", ", ",")
    tool_call = tool_call.replace('"', "'")
    tool_call = tool_call.lower()
    tool_call = re.sub(r"timeframe_hours='(\d+)'", r"timeframe_hours=\1", tool_call)
    return tool_call
