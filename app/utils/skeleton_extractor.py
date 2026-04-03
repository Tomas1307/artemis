from typing import Any

from app.skeleton.schemas.skeleton_schema import SkeletonSchema


def resolve_skeleton_ref(skeleton: SkeletonSchema, ref: str) -> Any:
    """Resolve a dotted reference path against the skeleton.

    Navigates the skeleton data structure following a dotted path
    like 'modules.jaguar.telemetry.temperature' and returns the
    value at that path.

    Args:
        skeleton: Validated SkeletonSchema instance.
        ref: Dotted path string (e.g., 'modules.condor', 'crew.commander',
            'security_protocols.MASA-SEC-001').

    Returns:
        The value at the resolved path. Can be a Pydantic model, dict, or
        primitive value depending on the depth of the path.

    Raises:
        AttributeError: If a path segment does not exist on the object.
        KeyError: If a dict key does not exist.
    """
    parts = ref.split(".")
    current = skeleton

    for part in parts:
        if isinstance(current, dict):
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            current = current.model_dump() if hasattr(current, "model_dump") else current
            current = current[part]

    return current


def format_skeleton_context(skeleton: SkeletonSchema, refs: list[str]) -> str:
    """Extract and format skeleton data for a set of references into readable text.

    Resolves each reference against the skeleton and formats the result
    as a structured text block suitable for injection into an LLM prompt.
    Appends a reference index of all protocol IDs/names and crew members
    to prevent hallucination of wrong names.

    Args:
        skeleton: Validated SkeletonSchema instance.
        refs: List of dotted path strings into the skeleton.

    Returns:
        Formatted string containing all resolved skeleton data plus
        protocol and crew reference indexes, ready for prompt injection.
    """
    if not refs:
        return "No skeleton data required for this document."

    sections = []
    for ref in refs:
        data = resolve_skeleton_ref(skeleton, ref)
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        sections.append(f"[{ref}]\n{_format_value(data, indent=0)}")

    sections.append(_build_reference_index(skeleton))

    return "\n\n".join(sections)


def _build_reference_index(skeleton: SkeletonSchema) -> str:
    """Build a reference index of protocol names and crew members.

    This index is appended to every document's skeleton context to ensure
    the LLM uses correct protocol names and only real crew members.

    Args:
        skeleton: Validated SkeletonSchema instance.

    Returns:
        Formatted string with protocol ID-to-name mapping and crew roster.
    """
    lines = ["[REFERENCE INDEX — use ONLY these names, do NOT invent others]"]

    lines.append("\nPROTOCOL ID-TO-NAME MAPPING (use EXACTLY these names):")
    for proto_id, proto in skeleton.security_protocols.items():
        lines.append(f"  {proto_id}: {proto.name} (severity={proto.severity.value}, scope={proto.scope.value})")

    lines.append("\nCREW ROSTER (these are the ONLY crew members who exist):")
    for role, member in skeleton.crew.items():
        lines.append(f"  {role}: {member.name}, {member.nationality}, assigned to {member.assigned_module}")

    lines.append("\nOPERATIONAL PROCEDURES:")
    for proc_id, proc in skeleton.operational_procedures.items():
        lines.append(f"  {proc_id}: {proc.name}")

    return "\n".join(lines)


def extract_required_facts(skeleton: SkeletonSchema, refs: list[str]) -> list[str]:
    """Extract a flat list of verifiable facts from skeleton references.

    Walks the resolved skeleton data and produces human-readable fact
    statements suitable for the LLM judge validation prompt.

    Args:
        skeleton: Validated SkeletonSchema instance.
        refs: List of dotted path strings into the skeleton.

    Returns:
        List of plain-language fact strings (e.g., 'Jaguar critical
        temperature threshold is above 50.0 celsius').
    """
    facts = []
    for ref in refs:
        data = resolve_skeleton_ref(skeleton, ref)
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        _collect_facts(data, ref, facts)
    return facts


def _format_value(value: Any, indent: int) -> str:
    """Recursively format a value into readable indented text.

    Args:
        value: Value to format (dict, list, or primitive).
        indent: Current indentation level.

    Returns:
        Formatted string representation.
    """
    prefix = "  " * indent
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(_format_value(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {v}")
        return "\n".join(lines)
    elif isinstance(value, list):
        return "\n".join(f"{prefix}- {item}" for item in value)
    else:
        return f"{prefix}{value}"


def _collect_facts(data: Any, path: str, facts: list[str]) -> None:
    """Recursively collect verifiable facts from skeleton data.

    Args:
        data: Current data node (dict, list, or primitive).
        path: Current dotted path for context.
        facts: Accumulator list for discovered facts.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}"
            if isinstance(value, (int, float)) and value is not None:
                facts.append(f"{child_path} = {value}")
            elif isinstance(value, str) and len(value) < 100:
                facts.append(f"{child_path} = '{value}'")
            elif isinstance(value, dict):
                _collect_facts(value, child_path, facts)
            elif isinstance(value, list) and all(isinstance(i, str) for i in value):
                facts.append(f"{child_path} = {value}")
    elif isinstance(data, (int, float)):
        facts.append(f"{path} = {data}")
    elif isinstance(data, str):
        facts.append(f"{path} = '{data}'")
