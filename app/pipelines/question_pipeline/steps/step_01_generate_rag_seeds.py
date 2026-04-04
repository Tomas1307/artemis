import random

from loguru import logger

from app.pipelines.question_pipeline.schemas.rag_seed_schema import RagQuestionSeed


MODULES = ["condor", "quetzal", "jaguar", "colibri", "vicuna", "tucan"]

THRESHOLD_DEFINITIONS = [
    {
        "protocol_id": "MASA-SEC-001",
        "metric": "pressure",
        "doc_id": "MASA-DOC-007",
        "condition": "pressure < 85.0 kPa",
        "tool_type": "both",
        "send_alert_severity": "critical",
        "send_alert_reason": "pressure_drop",
        "activate_scope": "station_wide",
        "reading_range": (70.0, 84.9),
        "reading_unit": "kPa",
        "incident_template": "Rapid pressure drop detected in {module}, possible hull breach",
        "rag_template": "Severity/protocol depends on threshold: pressure < 85.0 kPa = critical (MASA-SEC-001), station_wide. Reading {reading} is below 85.0.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-011",
        "metric": "pressure",
        "doc_id": "MASA-DOC-007",
        "condition": "pressure 85.0-89.9 kPa",
        "tool_type": "both",
        "send_alert_severity": "high",
        "send_alert_reason": "pressure_drop",
        "activate_scope": "module_only",
        "reading_range": (85.0, 89.9),
        "reading_unit": "kPa",
        "incident_template": "Pressure anomaly detected in {module}, readings declining",
        "rag_template": "Severity/protocol depends on threshold: pressure 85.0-89.9 kPa = high (MASA-SEC-011), module_only. Reading {reading} is in range.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-002",
        "metric": "oxygen",
        "doc_id": "MASA-DOC-007",
        "condition": "oxygen < 14.0%",
        "tool_type": "both",
        "send_alert_severity": "critical",
        "send_alert_reason": "oxygen_leak",
        "activate_scope": "station_wide",
        "reading_range": (10.0, 13.9),
        "reading_unit": "%",
        "incident_template": "Oxygen concentration critically low in {module}, crew may be affected",
        "rag_template": "Severity/protocol depends on threshold: O2 < 14.0% = critical (MASA-SEC-002), station_wide. Reading {reading} is below 14.0.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-010",
        "metric": "oxygen",
        "doc_id": "MASA-DOC-007",
        "condition": "oxygen 14.0-15.9%",
        "tool_type": "both",
        "send_alert_severity": "high",
        "send_alert_reason": "oxygen_leak",
        "activate_scope": "module_only",
        "reading_range": (14.0, 15.9),
        "reading_unit": "%",
        "incident_template": "Oxygen declining in {module}, entering warning band",
        "rag_template": "Severity/protocol depends on threshold: O2 14.0-15.9% = high (MASA-SEC-010), module_only. Reading {reading} is in range.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-004",
        "metric": "radiation",
        "doc_id": "MASA-DOC-009",
        "condition": "radiation > 5.0 mSv/hr",
        "tool_type": "both",
        "send_alert_severity": "critical",
        "send_alert_reason": "radiation_spike",
        "activate_scope": "station_wide",
        "reading_range": (5.1, 12.0),
        "reading_unit": "mSv/hr",
        "incident_template": "Radiation spike detected in {module}, sensors well above normal",
        "rag_template": "Severity/protocol depends on threshold: radiation > 5.0 mSv/hr = critical (MASA-SEC-004), station_wide. Reading {reading} exceeds 5.0.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-012",
        "metric": "radiation",
        "doc_id": "MASA-DOC-009",
        "condition": "radiation 1.1-5.0 mSv/hr",
        "tool_type": "both",
        "send_alert_severity": "medium",
        "send_alert_reason": "radiation_spike",
        "activate_scope": "station_wide",
        "reading_range": (1.1, 4.9),
        "reading_unit": "mSv/hr",
        "incident_template": "Elevated radiation readings in {module}, solar activity may be increasing",
        "rag_template": "Severity/protocol depends on threshold: radiation 1.1-5.0 mSv/hr = medium (MASA-SEC-012), station_wide. Reading {reading} is in range.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-003",
        "metric": "temperature_rate",
        "doc_id": "MASA-DOC-008",
        "condition": "temperature rise > 2.0 C/min",
        "tool_type": "both",
        "send_alert_severity": "critical",
        "send_alert_reason": "abnormal_temperature",
        "activate_scope": "module_only",
        "reading_range": (2.1, 5.0),
        "reading_unit": "degrees per minute",
        "incident_template": "Rapid temperature rise in {module}, possible fire or thermal runaway",
        "rag_template": "Severity/protocol depends on threshold: temp rise > 2.0 C/min = critical (MASA-SEC-003), module_only. Reading {reading} exceeds 2.0.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-006",
        "metric": "power",
        "doc_id": "MASA-DOC-011",
        "condition": "power > 97% rated capacity",
        "tool_type": "both",
        "send_alert_severity": "critical",
        "send_alert_reason": "power_fluctuation",
        "activate_scope": "station_wide",
        "reading_range": (97.1, 99.5),
        "reading_unit": "%",
        "incident_template": "Power consumption at near-maximum capacity, grid stress in {module}",
        "rag_template": "Severity/protocol depends on threshold: power > 97% = critical (MASA-SEC-006), station_wide. Reading {reading} exceeds 97%.",
        "applicable_modules": ["condor", "jaguar"],
    },
    {
        "protocol_id": "MASA-SEC-007",
        "metric": "communication",
        "doc_id": "MASA-DOC-011",
        "condition": "ground control contact lost > 30 minutes",
        "tool_type": "both",
        "send_alert_severity": "high",
        "send_alert_reason": "communication_loss",
        "activate_scope": "station_wide",
        "reading_range": (31, 90),
        "reading_unit": "minutes",
        "incident_template": "Ground control contact lost, blackout extending from {module}",
        "rag_template": "Severity/protocol depends on threshold: no contact > 30 min = high (MASA-SEC-007), station_wide. Reading {reading} exceeds 30 min.",
        "applicable_modules": ["colibri"],
    },
    {
        "protocol_id": "MASA-SEC-008",
        "metric": "hull_stress",
        "doc_id": "MASA-DOC-010",
        "condition": "hull stress > 85% rated capacity",
        "tool_type": "both",
        "send_alert_severity": "high",
        "send_alert_reason": "structural_damage",
        "activate_scope": "module_only",
        "reading_range": (85.1, 95.0),
        "reading_unit": "%",
        "incident_template": "Hull stress elevated in {module}, possible micrometeorite impact",
        "rag_template": "Severity/protocol depends on threshold: hull stress > 85% = high (MASA-SEC-008), module_only. Reading {reading} exceeds 85%.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-009",
        "metric": "co2",
        "doc_id": "MASA-DOC-007",
        "condition": "CO2 > 1.5%",
        "tool_type": "both",
        "send_alert_severity": "high",
        "send_alert_reason": "oxygen_leak",
        "activate_scope": "module_only",
        "reading_range": (1.6, 3.0),
        "reading_unit": "%",
        "incident_template": "CO2 concentration rising in {module}, crew reporting headaches",
        "rag_template": "Severity/protocol depends on threshold: CO2 > 1.5% = high (MASA-SEC-009), module_only. Reading {reading} exceeds 1.5%.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-013",
        "metric": "docking_pressure",
        "doc_id": "MASA-DOC-010",
        "condition": "docking pressure differential > 5.0 kPa",
        "tool_type": "activate_only",
        "send_alert_severity": "high",
        "send_alert_reason": "pressure_drop",
        "activate_scope": "module_only",
        "reading_range": (5.1, 8.0),
        "reading_unit": "kPa",
        "incident_template": "Pressure differential anomaly at {module} docking port during active docking",
        "rag_template": "Protocol depends on threshold: docking differential > 5.0 kPa = MASA-SEC-013, module_only. Reading {reading} exceeds 5.0 kPa.",
        "applicable_modules": ["vicuna"],
    },
    {
        "protocol_id": "MASA-SEC-014",
        "metric": "scrubber_efficiency",
        "doc_id": "MASA-DOC-011",
        "condition": "CO2 scrubber efficiency < 60%",
        "tool_type": "activate_only",
        "send_alert_severity": "high",
        "send_alert_reason": "system_failure",
        "activate_scope": "station_wide",
        "reading_range": (30.0, 59.0),
        "reading_unit": "%",
        "incident_template": "CO2 scrubber efficiency dropping in {module}, air quality degrading",
        "rag_template": "Protocol depends on threshold: scrubber < 60% = MASA-SEC-014, station_wide. Reading {reading} is below 60%.",
        "applicable_modules": ["jaguar"],
    },
    {
        "protocol_id": "MASA-SEC-017",
        "metric": "airlock_rate",
        "doc_id": "MASA-DOC-010",
        "condition": "airlock equalization rate > 0.5 kPa/min",
        "tool_type": "activate_only",
        "send_alert_severity": "critical",
        "send_alert_reason": "pressure_drop",
        "activate_scope": "module_only",
        "reading_range": (0.6, 1.5),
        "reading_unit": "kPa per minute",
        "incident_template": "Airlock equalization rate exceeding safe limits in {module}",
        "rag_template": "Protocol depends on threshold: equalization > 0.5 kPa/min = MASA-SEC-017 (critical), module_only. Reading {reading} exceeds 0.5.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-018",
        "metric": "voltage",
        "doc_id": "MASA-DOC-011",
        "condition": "voltage outside 106.2V-129.8V",
        "tool_type": "both",
        "send_alert_severity": "medium",
        "send_alert_reason": "power_fluctuation",
        "activate_scope": "module_only",
        "reading_range_low": (95.0, 106.1),
        "reading_range_high": (129.9, 140.0),
        "reading_unit": "V",
        "incident_template": "Voltage fluctuation detected on {module} power bus",
        "rag_template": "Severity/protocol depends on threshold: voltage outside 106.2-129.8V = medium (MASA-SEC-018), module_only. Reading {reading} is outside safe range.",
        "applicable_modules": MODULES,
    },
    {
        "protocol_id": "MASA-SEC-015",
        "metric": "water_recycling",
        "doc_id": "MASA-DOC-011",
        "condition": "water recycling output < 40% for 2+ hours",
        "tool_type": "activate_only",
        "send_alert_severity": "medium",
        "send_alert_reason": "system_failure",
        "activate_scope": "station_wide",
        "reading_range": (20.0, 39.0),
        "reading_unit": "%",
        "incident_template": "Water recycling output dropping in {module}, sustained degradation for hours",
        "rag_template": "Protocol depends on threshold: water recycling < 40% for 2h = MASA-SEC-015 (medium), station_wide. Reading {reading} is below 40%.",
        "applicable_modules": ["jaguar", "vicuna"],
    },
    {
        "protocol_id": "MASA-SEC-005",
        "metric": "crew_vitals",
        "doc_id": "MASA-DOC-012",
        "condition": "crew member critical vital signs or medical emergency",
        "tool_type": "activate_only",
        "send_alert_severity": "high",
        "send_alert_reason": "system_failure",
        "activate_scope": "module_only",
        "reading_range": None,
        "reading_unit": None,
        "incident_template": "Crew member exhibiting distress in {module}, immediate medical attention needed",
        "rag_template": "Protocol depends on situation: medical emergency triggers MASA-SEC-005, module_only. Student must match symptoms to protocol.",
        "applicable_modules": MODULES,
        "qualitative": True,
    },
    {
        "protocol_id": "MASA-SEC-019",
        "metric": "bio_contamination",
        "doc_id": "MASA-DOC-012",
        "condition": "biological contamination sensor positive in Quetzal",
        "tool_type": "activate_only",
        "send_alert_severity": "high",
        "send_alert_reason": "system_failure",
        "activate_scope": "module_only",
        "reading_range": None,
        "reading_unit": None,
        "incident_template": "Biological contamination alarm triggered in {module} laboratory",
        "rag_template": "Protocol depends on situation: bio contamination triggers MASA-SEC-019, module_only. Student must match to correct protocol.",
        "applicable_modules": ["quetzal"],
        "qualitative": True,
    },
]


class GenerateRagSeedsStep:
    """Step 1-RAG — Generate seeds for RAG-dependent questions from protocol thresholds.

    Systematically enumerates (protocol x module x reading_value) combinations
    to produce seeds where at least one tool parameter requires consulting
    MASA-SEC protocol documents.

    Each threshold definition produces both send_alert and activate_protocol
    seeds (where applicable), with multiple sensor readings per combination
    for variety.

    Args:
        target_total: Target number of RAG seeds to generate. Default 2000.
        random_seed: Seed for reproducible random sampling. Default 42.
        readings_per_combo: Number of distinct readings per (protocol, module) pair.
    """

    def __init__(
        self,
        target_total: int = 2000,
        random_seed: int = 42,
        readings_per_combo: int = 3,
    ) -> None:
        self._target = target_total
        self._rng = random.Random(random_seed)
        self._readings_per_combo = readings_per_combo

    def execute(self) -> list[RagQuestionSeed]:
        """Generate all RAG-dependent seeds.

        Returns:
            List of RagQuestionSeed instances ready for LLM query generation.
        """
        all_seeds: list[RagQuestionSeed] = []
        seed_counter = 0

        for defn in THRESHOLD_DEFINITIONS:
            is_qualitative = defn.get("qualitative", False)

            for module in defn["applicable_modules"]:
                tool_types = self._get_tool_types(defn)

                if is_qualitative:
                    readings = [("reported", 1)]
                else:
                    readings = self._generate_readings(defn)

                for reading_str, phrasing_idx in readings:
                    for tool_type in tool_types:
                        seed_counter += 1
                        tool_call = self._build_tool_call(defn, module, tool_type)
                        seed = RagQuestionSeed(
                            seed_id=f"rag-seed-{seed_counter:05d}",
                            tool_name=tool_type,
                            tool_call=tool_call,
                            module=module,
                            metric=defn["metric"],
                            sensor_reading=reading_str,
                            incident_description=defn["incident_template"].format(module=module),
                            rag_requirement=defn["rag_template"].format(reading=reading_str),
                            doc_id=defn["doc_id"],
                            protocol_id=defn["protocol_id"],
                            phrasing_index=phrasing_idx,
                        )
                        all_seeds.append(seed)

        self._rng.shuffle(all_seeds)

        if len(all_seeds) > self._target:
            all_seeds = all_seeds[: self._target]

        logger.info(
            f"Generated {len(all_seeds)} RAG seeds from "
            f"{len(THRESHOLD_DEFINITIONS)} threshold definitions"
        )
        return all_seeds

    def _get_tool_types(self, defn: dict) -> list[str]:
        """Determine which tool types to generate for a threshold definition.

        Args:
            defn: Threshold definition dictionary.

        Returns:
            List of tool name strings ('send_alert', 'activate_protocol', or both).
        """
        tool_type = defn["tool_type"]
        if tool_type == "both":
            return ["send_alert", "activate_protocol"]
        elif tool_type == "activate_only":
            return ["activate_protocol"]
        return ["send_alert"]

    def _generate_readings(self, defn: dict) -> list[tuple[str, int]]:
        """Generate diverse sensor reading values within the threshold range.

        Args:
            defn: Threshold definition with reading_range or reading_range_low/high.

        Returns:
            List of (reading_string, phrasing_index) tuples.
        """
        results = []
        unit = defn["reading_unit"]

        if "reading_range" in defn and defn["reading_range"] is not None:
            low, high = defn["reading_range"]
            for i in range(self._readings_per_combo):
                value = round(self._rng.uniform(low, high), 1)
                reading_str = self._format_reading(value, unit)
                results.append((reading_str, i + 1))
        elif "reading_range_low" in defn:
            low_range = defn["reading_range_low"]
            high_range = defn["reading_range_high"]
            half = max(1, self._readings_per_combo // 2)
            for i in range(half):
                value = round(self._rng.uniform(*low_range), 1)
                results.append((self._format_reading(value, unit), i + 1))
            for i in range(self._readings_per_combo - half):
                value = round(self._rng.uniform(*high_range), 1)
                results.append((self._format_reading(value, unit), half + i + 1))

        return results

    def _format_reading(self, value: float, unit: str) -> str:
        """Format a sensor reading as a natural string.

        Args:
            value: Numeric reading value.
            unit: Unit string (kPa, %, mSv/hr, etc.).

        Returns:
            Formatted reading string (e.g., '83.7 kPa', '13.2%').
        """
        if unit == "%":
            return f"{value}%"
        if unit == "minutes":
            return f"{int(value)} minutes"
        return f"{value} {unit}"

    def _build_tool_call(self, defn: dict, module: str, tool_type: str) -> str:
        """Build the canonical tool call string for a seed.

        Args:
            defn: Threshold definition.
            module: Module name.
            tool_type: 'send_alert' or 'activate_protocol'.

        Returns:
            Canonical tool call string.
        """
        if tool_type == "send_alert":
            severity = defn["send_alert_severity"]
            reason = defn["send_alert_reason"]
            return f"send_alert(module='{module}',severity='{severity}',reason='{reason}')"
        else:
            protocol_id = defn["protocol_id"]
            scope = defn["activate_scope"]
            return f"activate_protocol(protocol_id='{protocol_id}',scope='{scope}')"
