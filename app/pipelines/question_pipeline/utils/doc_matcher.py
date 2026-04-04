MODULE_TO_DOC: dict[str, list[str]] = {
    "condor": ["MASA-DOC-001"],
    "quetzal": ["MASA-DOC-002"],
    "jaguar": ["MASA-DOC-003"],
    "colibri": ["MASA-DOC-004"],
    "vicuna": ["MASA-DOC-005"],
    "tucan": ["MASA-DOC-006"],
}

METRIC_TO_DOC: dict[str, list[str]] = {
    "temperature": ["MASA-DOC-013"],
    "pressure": ["MASA-DOC-015"],
    "oxygen": ["MASA-DOC-015"],
    "radiation": ["MASA-DOC-009"],
    "humidity": ["MASA-DOC-017"],
    "power": ["MASA-DOC-014"],
}

SYSTEM_TO_DOC: dict[str, list[str]] = {
    "life_support": ["MASA-DOC-015"],
    "power": ["MASA-DOC-014"],
    "thermal": ["MASA-DOC-013"],
    "structural": ["MASA-DOC-010"],
    "communications": ["MASA-DOC-016"],
}

REASON_TO_DOC: dict[str, list[str]] = {
    "abnormal_temperature": ["MASA-DOC-008", "MASA-DOC-013"],
    "pressure_drop": ["MASA-DOC-007", "MASA-DOC-015"],
    "oxygen_leak": ["MASA-DOC-007", "MASA-DOC-015"],
    "radiation_spike": ["MASA-DOC-009"],
    "system_failure": ["MASA-DOC-011"],
    "power_fluctuation": ["MASA-DOC-014"],
    "communication_loss": ["MASA-DOC-016"],
    "structural_damage": ["MASA-DOC-010"],
}

PROTOCOL_TO_DOC: dict[str, list[str]] = {
    "MASA-SEC-001": ["MASA-DOC-007"],
    "MASA-SEC-002": ["MASA-DOC-007"],
    "MASA-SEC-003": ["MASA-DOC-008"],
    "MASA-SEC-004": ["MASA-DOC-009"],
    "MASA-SEC-005": ["MASA-DOC-006"],
    "MASA-SEC-006": ["MASA-DOC-014"],
    "MASA-SEC-007": ["MASA-DOC-016"],
    "MASA-SEC-008": ["MASA-DOC-010"],
    "MASA-SEC-009": ["MASA-DOC-015"],
    "MASA-SEC-010": ["MASA-DOC-007"],
    "MASA-SEC-011": ["MASA-DOC-015"],
    "MASA-SEC-012": ["MASA-DOC-009"],
    "MASA-SEC-013": ["MASA-DOC-013"],
    "MASA-SEC-014": ["MASA-DOC-015"],
    "MASA-SEC-015": ["MASA-DOC-015"],
    "MASA-SEC-016": ["MASA-DOC-040"],
    "MASA-SEC-017": ["MASA-DOC-020"],
    "MASA-SEC-018": ["MASA-DOC-014"],
    "MASA-SEC-019": ["MASA-DOC-002"],
    "MASA-SEC-020": ["MASA-DOC-038"],
}

TASK_TO_DOC: dict[str, list[str]] = {
    "sensor_repair": ["MASA-DOC-019"],
    "filter_replacement": ["MASA-DOC-017", "MASA-DOC-019"],
    "system_calibration": ["MASA-DOC-019"],
    "hull_inspection": ["MASA-DOC-010", "MASA-DOC-019"],
    "power_cell_swap": ["MASA-DOC-014", "MASA-DOC-019"],
    "software_update": ["MASA-DOC-019"],
}

MANEUVER_TO_DOC: dict[str, list[str]] = {
    "orbit_adjustment": ["MASA-DOC-040"],
    "docking": ["MASA-DOC-021", "MASA-DOC-040"],
    "debris_avoidance": ["MASA-DOC-040"],
    "reentry": ["MASA-DOC-040"],
    "station_keeping": ["MASA-DOC-040"],
}

SUPPLY_CATEGORY_TO_DOC: dict[str, list[str]] = {
    "medical": ["MASA-DOC-006", "MASA-DOC-039"],
    "food": ["MASA-DOC-039"],
    "equipment": ["MASA-DOC-039"],
    "fuel": ["MASA-DOC-039"],
    "spare_parts": ["MASA-DOC-039"],
    "scientific": ["MASA-DOC-039"],
}

CONTROL_SYSTEM_TO_DOC: dict[str, list[str]] = {
    "ventilation": ["MASA-DOC-017"],
    "heating": ["MASA-DOC-013"],
    "lighting": ["MASA-DOC-018"],
    "cooling": ["MASA-DOC-013"],
    "filtration": ["MASA-DOC-017"],
}

CREW_ROLE_TO_DOC: dict[str, list[str]] = {
    "commander": ["MASA-DOC-030"],
    "pilot": ["MASA-DOC-031"],
    "specialist_1": ["MASA-DOC-032"],
    "specialist_2": ["MASA-DOC-033"],
    "engineer": ["MASA-DOC-034"],
    "medical_officer": ["MASA-DOC-035"],
}

_ALWAYS_INCLUDE_FOR_ALERTS = ["MASA-DOC-038"]
_ALWAYS_INCLUDE_FOR_PROTOCOLS = ["MASA-DOC-038"]


def match_docs(tool_name: str, tool_params: dict[str, str | int]) -> list[str]:
    """Determine which document IDs are relevant for a given tool call.

    Uses static lookup tables to map tool parameters to the document IDs that
    contain the facts a student's RAG system must retrieve to answer correctly.
    All returned lists are deduplicated and sorted.

    Args:
        tool_name: Name of the tool being called.
        tool_params: Parameter dictionary for the tool call.

    Returns:
        Sorted, deduplicated list of relevant document IDs.
    """
    docs: set[str] = set()

    if tool_name == "no_action":
        return []

    module = tool_params.get("module", "")
    if module and module in MODULE_TO_DOC:
        docs.update(MODULE_TO_DOC[module])

    if tool_name == "get_telemetry":
        metric = tool_params.get("metric", "")
        if metric in METRIC_TO_DOC:
            docs.update(METRIC_TO_DOC[metric])

    elif tool_name == "get_module_status":
        system = tool_params.get("system", "")
        if system in SYSTEM_TO_DOC:
            docs.update(SYSTEM_TO_DOC[system])

    elif tool_name == "send_alert":
        reason = tool_params.get("reason", "")
        if reason in REASON_TO_DOC:
            docs.update(REASON_TO_DOC[reason])
        docs.update(_ALWAYS_INCLUDE_FOR_ALERTS)

    elif tool_name == "activate_protocol":
        protocol_id = tool_params.get("protocol_id", "")
        if protocol_id in PROTOCOL_TO_DOC:
            docs.update(PROTOCOL_TO_DOC[protocol_id])
        docs.update(_ALWAYS_INCLUDE_FOR_PROTOCOLS)

    elif tool_name == "schedule_maintenance":
        task = tool_params.get("task", "")
        if task in TASK_TO_DOC:
            docs.update(TASK_TO_DOC[task])

    elif tool_name == "control_system":
        system = tool_params.get("system", "")
        if system in CONTROL_SYSTEM_TO_DOC:
            docs.update(CONTROL_SYSTEM_TO_DOC[system])

    elif tool_name == "calculate_trajectory":
        maneuver = tool_params.get("maneuver", "")
        if maneuver in MANEUVER_TO_DOC:
            docs.update(MANEUVER_TO_DOC[maneuver])

    elif tool_name == "request_supply":
        category = tool_params.get("category", "")
        if category in SUPPLY_CATEGORY_TO_DOC:
            docs.update(SUPPLY_CATEGORY_TO_DOC[category])

    elif tool_name == "send_message":
        recipient = tool_params.get("recipient", "")
        if recipient in CREW_ROLE_TO_DOC:
            docs.update(CREW_ROLE_TO_DOC[recipient])

    return sorted(docs)
