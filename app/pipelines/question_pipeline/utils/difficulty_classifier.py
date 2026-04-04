import itertools

from app.pipelines.question_pipeline.schemas.question_schema import DifficultyLevel


_DIFFICULTY_CYCLE = [
    DifficultyLevel.easy,
    DifficultyLevel.medium,
    DifficultyLevel.hard,
    DifficultyLevel.trap,
]

_HIGH_STAKES_REASONS = {
    "oxygen_leak",
    "pressure_drop",
    "radiation_spike",
    "structural_damage",
    "system_failure",
}

_CRITICAL_PROTOCOLS = {
    "MASA-SEC-001",
    "MASA-SEC-002",
    "MASA-SEC-003",
    "MASA-SEC-004",
    "MASA-SEC-020",
}


def assign_difficulty(
    tool_name: str,
    tool_params: dict[str, str | int],
    phrasing_index: int,
) -> DifficultyLevel:
    """Assign a difficulty level to a question seed.

    Rules applied in priority order:
    1. no_action seeds with phrasing_index % 4 == 3 are always trap.
    2. send_alert seeds with critical severity or high-stakes reason get hard/trap.
    3. activate_protocol seeds for critical protocols get hard.
    4. All other seeds cycle through easy → medium → hard → trap based on phrasing_index.

    This ensures each tool produces a balanced 25% spread across difficulties
    across its full seed population.

    Args:
        tool_name: Tool being called.
        tool_params: Parameters for the tool call.
        phrasing_index: Zero-based index distinguishing multiple phrasings per combo.

    Returns:
        Assigned DifficultyLevel.
    """
    base_difficulty = _DIFFICULTY_CYCLE[phrasing_index % 4]

    if tool_name == "no_action":
        if phrasing_index % 4 == 3:
            return DifficultyLevel.trap
        return base_difficulty

    if tool_name == "send_alert":
        severity = tool_params.get("severity", "")
        reason = tool_params.get("reason", "")
        if severity == "critical" and reason in _HIGH_STAKES_REASONS:
            return DifficultyLevel.hard if phrasing_index % 2 == 0 else DifficultyLevel.trap
        if severity == "critical":
            return DifficultyLevel.hard

    if tool_name == "activate_protocol":
        protocol_id = tool_params.get("protocol_id", "")
        if protocol_id in _CRITICAL_PROTOCOLS:
            return DifficultyLevel.hard if phrasing_index % 2 == 0 else DifficultyLevel.trap

    return base_difficulty
