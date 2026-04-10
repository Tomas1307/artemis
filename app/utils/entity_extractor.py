"""Pure functions for extracting MASA universe entities from text.

Extracts module names, protocol IDs, crew members, and numeric thresholds
from chunk content. These entities serve as structured metadata for
retrieval filtering and validation.
"""

import re


MASA_MODULES = {
    "condor": ["cóndor", "condor", "córdor"],
    "quetzal": ["quetzal"],
    "jaguar": ["jaguar"],
    "colibri": ["colibrí", "colibri", "colibří"],
    "vicuna": ["vicuña", "vicuna", "vicuña"],
    "tucan": ["tucán", "tucan"],
}

MASA_CREW = {
    "Santiago Reyes": ["santiago reyes", "commander reyes", "commander santiago"],
    "Ana Valdivia": ["ana valdivia", "pilot valdivia", "pilot ana"],
    "Kai Nakamura": ["kai nakamura", "specialist nakamura", "specialist kai"],
    "Fátima Al-Hassan": [
        "fátima al-hassan", "fatima al-hassan", "al-hassan",
        "specialist al-hassan", "specialist fátima",
    ],
    "Pavel Kozlov": ["pavel kozlov", "engineer kozlov", "engineer pavel"],
    "Lucía Mendoza": [
        "lucía mendoza", "lucia mendoza", "mendoza",
        "medical officer mendoza", "medical officer lucía",
    ],
}

PROTOCOL_PATTERN = re.compile(r'MASA-(?:SEC|OPS)-\d{3}')

THRESHOLD_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*'
    r'(mSv/(?:hr|hour)|kPa|°C|%|mSv)'
)

TOOL_KEYWORDS = {
    "get_telemetry": ["telemetry", "reading", "sensor", "measurement", "monitor"],
    "get_crew_status": ["crew status", "crew member", "vital", "health check"],
    "get_module_status": ["module status", "system status", "operational status"],
    "send_alert": ["alert", "severity", "warning", "critical alert"],
    "send_message": ["message", "notify", "communication", "transmit"],
    "schedule_maintenance": ["maintenance", "repair", "calibration", "inspection"],
    "activate_protocol": ["activate protocol", "protocol activation", "MASA-SEC", "MASA-OPS"],
    "control_system": ["power down", "shut down", "restart", "activate system", "enable", "disable"],
    "calculate_trajectory": ["trajectory", "orbit", "maneuver", "reentry", "deorbit"],
    "request_supply": ["supply", "resupply", "cargo", "inventory", "provision"],
}


def extract_modules(text: str) -> list[str]:
    """Extract MASA module canonical names mentioned in text.

    Args:
        text: Input text to scan.

    Returns:
        Sorted list of canonical module names (e.g., ['condor', 'jaguar']).
    """
    text_lower = text.lower()
    found = set()
    for canonical, variants in MASA_MODULES.items():
        for variant in variants:
            if variant in text_lower:
                found.add(canonical)
                break
    return sorted(found)


def extract_protocols(text: str) -> list[str]:
    """Extract MASA protocol IDs mentioned in text.

    Args:
        text: Input text to scan.

    Returns:
        Sorted list of unique protocol IDs (e.g., ['MASA-OPS-003', 'MASA-SEC-001']).
    """
    return sorted(set(PROTOCOL_PATTERN.findall(text)))


def extract_crew(text: str) -> list[str]:
    """Extract MASA crew member names mentioned in text.

    Args:
        text: Input text to scan.

    Returns:
        Sorted list of crew member full names (e.g., ['Ana Valdivia', 'Pavel Kozlov']).
    """
    text_lower = text.lower()
    found = set()
    for full_name, variants in MASA_CREW.items():
        for variant in variants:
            if variant in text_lower:
                found.add(full_name)
                break
    return sorted(found)


def extract_thresholds(text: str) -> list[dict]:
    """Extract numeric threshold values with units from text.

    Captures values like '85.0 kPa', '5.0 mSv/hr', '14.0%', '55°C'.

    Args:
        text: Input text to scan.

    Returns:
        List of dicts with 'value' (float) and 'unit' (str) keys,
        deduplicated by (value, unit) pairs.
    """
    matches = THRESHOLD_PATTERN.findall(text)
    seen = set()
    results = []
    for value_str, unit in matches:
        key = (float(value_str), unit)
        if key not in seen:
            seen.add(key)
            results.append({"value": float(value_str), "unit": unit})
    return results


def extract_relevant_tools(text: str) -> list[str]:
    """Infer which MASA tools are likely relevant to the chunk content.

    Uses keyword matching against known tool-topic associations.

    Args:
        text: Input text to scan.

    Returns:
        Sorted list of tool names that have keyword matches in the text.
    """
    text_lower = text.lower()
    found = set()
    for tool, keywords in TOOL_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found.add(tool)
                break
    return sorted(found)
