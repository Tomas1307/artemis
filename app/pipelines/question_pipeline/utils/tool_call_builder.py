def build_tool_call(tool_name: str, tool_params: dict[str, str | int]) -> str:
    """Build a canonical tool call string from a tool name and its parameters.

    Applies strict canonical format rules:
    - No spaces after commas
    - Parameters in defined order (as passed via tool_params, which must be ordered)
    - String values wrapped in single quotes
    - Numeric values unquoted
    - All lowercase (tool name and string values)

    Args:
        tool_name: Name of the tool (e.g., 'get_telemetry').
        tool_params: Ordered mapping of parameter names to values. Order must match
            the tool's defined parameter_order from tools_definition.json.

    Returns:
        Canonical tool call string (e.g., "get_telemetry(module='jaguar',metric='temperature',timeframe_hours=6)").

    Examples:
        >>> build_tool_call("get_telemetry", {"module": "jaguar", "metric": "temperature", "timeframe_hours": 6})
        "get_telemetry(module='jaguar',metric='temperature',timeframe_hours=6)"
        >>> build_tool_call("no_action", {})
        "no_action"
    """
    if tool_name == "no_action":
        return "no_action"

    parts = []
    for key, value in tool_params.items():
        if isinstance(value, int):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}='{value}'")

    return f"{tool_name}({','.join(parts)})"
