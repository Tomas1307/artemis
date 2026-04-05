import itertools
import random
from collections import defaultdict

from loguru import logger

from app.pipelines.question_pipeline.schemas.question_schema import DifficultyLevel, QuestionSplit
from app.pipelines.question_pipeline.schemas.seed_schema import QuestionSeed
from app.pipelines.question_pipeline.utils.difficulty_classifier import assign_difficulty
from app.pipelines.question_pipeline.utils.doc_matcher import match_docs
from app.pipelines.question_pipeline.utils.tool_call_builder import build_tool_call
from app.skeleton.schemas.skeleton_schema import SkeletonSchema


_MODULES = ["condor", "quetzal", "jaguar", "colibri", "vicuna", "tucan"]
_METRICS = ["temperature", "pressure", "oxygen", "radiation", "humidity", "power"]
_TIMEFRAMES = [1, 6, 12, 24]
_CREW_INFO = ["health", "location", "current_activity", "schedule"]
_MODULE_SYSTEMS = ["life_support", "power", "thermal", "structural", "communications"]
_SEVERITIES = ["low", "medium", "high", "critical"]
_REASONS = [
    "abnormal_temperature", "pressure_drop", "oxygen_leak", "radiation_spike",
    "system_failure", "power_fluctuation", "communication_loss", "structural_damage",
]
_RECIPIENTS = ["commander", "pilot", "specialist_1", "specialist_2", "engineer", "medical_officer", "all_crew"]
_PRIORITIES_MSG = ["low", "medium", "high", "urgent"]
_TASKS = ["sensor_repair", "filter_replacement", "system_calibration", "hull_inspection", "power_cell_swap", "software_update"]
_PRIORITIES_MAINT = ["routine", "urgent"]
_PROTOCOLS = [f"MASA-SEC-{i:03d}" for i in range(1, 21)]
_SCOPES = ["module_only", "station_wide"]
_CTRL_SYSTEMS = ["ventilation", "heating", "lighting", "cooling", "filtration"]
_ACTIONS = ["activate", "deactivate", "increase", "decrease"]
_MANEUVERS = ["orbit_adjustment", "docking", "debris_avoidance", "reentry", "station_keeping"]
_URGENCIES_TRAJ = ["planned", "immediate"]
_SUPPLY_CATEGORIES = ["medical", "food", "equipment", "fuel", "spare_parts", "scientific"]
_URGENCIES_SUPPLY = ["routine", "expedited", "emergency"]

_SPLIT_RATIOS = {
    QuestionSplit.train: 0.40,
    QuestionSplit.test: 0.50,
    QuestionSplit.hidden: 0.10,
}


def _assign_split(index: int, total: int) -> QuestionSplit:
    """Assign a dataset split based on position within the seed population for a tool.

    Uses deterministic thresholds so split distribution is exact across each
    tool's 200 seeds: 80 train / 100 test / 20 hidden.

    Args:
        index: Zero-based position of this seed within its tool's seed list.
        total: Total seeds for this tool (always TARGET_PER_TOOL).

    Returns:
        Assigned QuestionSplit.
    """
    train_cutoff = int(total * _SPLIT_RATIOS[QuestionSplit.train])
    test_cutoff = train_cutoff + int(total * _SPLIT_RATIOS[QuestionSplit.test])
    if index < train_cutoff:
        return QuestionSplit.train
    if index < test_cutoff:
        return QuestionSplit.test
    return QuestionSplit.hidden


def _build_module_context(module: str, skeleton: SkeletonSchema) -> list[str]:
    """Extract telemetry threshold facts for a module from the skeleton.

    Args:
        module: Module identifier (e.g., 'jaguar').
        skeleton: Validated skeleton instance.

    Returns:
        List of formatted fact strings.
    """
    facts = []
    mod = skeleton.modules.get(module)
    if not mod:
        return facts
    facts.append(f"Module: {mod.name} — {mod.function}")
    tel = mod.telemetry
    if tel:
        t = tel.temperature
        facts.append(f"Temperature normal: {t.normal_min}–{t.normal_max}°C, critical above {t.critical_above}°C")
        p = tel.pressure
        facts.append(f"Pressure normal: {p.normal_min}–{p.normal_max} kPa, critical below {p.critical_below} kPa")
        o = tel.oxygen
        facts.append(f"Oxygen normal: {o.normal_min}–{o.normal_max}%, critical below {o.critical_below}%")
        r = tel.radiation
        facts.append(f"Radiation normal: below {r.normal_max} mSv/h, critical above {r.critical_above} mSv/h")
    crew = [name for name, cfg in (skeleton.crew or {}).items() if getattr(cfg, "assigned_module", None) == module]
    if crew:
        facts.append(f"Crew in {module}: {', '.join(crew)}")
    return facts


def _build_protocol_context(protocol_id: str, skeleton: SkeletonSchema) -> list[str]:
    """Extract protocol trigger and severity facts from the skeleton.

    Args:
        protocol_id: Protocol identifier (e.g., 'MASA-SEC-001').
        skeleton: Validated skeleton instance.

    Returns:
        List of formatted fact strings.
    """
    facts = []
    proto = (skeleton.security_protocols or {}).get(protocol_id)
    if not proto:
        return facts
    facts.append(f"Protocol {protocol_id}: {proto.name}")
    facts.append(f"Trigger: {proto.trigger_condition}")
    facts.append(f"Severity: {proto.severity.value}, Scope: {proto.scope.value}")
    return facts


class GenerateSeedsStep:
    """Step 1 — Generate all question seeds deterministically from the skeleton.

    Enumerates valid parameter combinations for each tool, samples to exactly
    TARGET_PER_TOOL seeds, assigns difficulty via difficulty_classifier, assigns
    splits via stratified indexing, and attaches skeleton-derived context_facts
    and doc_ids.

    No LLM is called in this step. Output is a complete list of QuestionSeed
    instances ready for query generation.

    Args:
        skeleton: Validated SkeletonSchema instance.
        target_per_tool: Number of seeds to produce per tool. Default 200.
        random_seed: Seed for reproducible sampling. Default 42.
    """

    def __init__(
        self,
        skeleton: SkeletonSchema,
        target_per_tool: int = 200,
        random_seed: int = 42,
    ) -> None:
        self._skeleton = skeleton
        self._target = target_per_tool
        self._rng = random.Random(random_seed)

    def execute(self) -> list[QuestionSeed]:
        """Generate all seeds for all 10 tools.

        Returns:
            Flat list of QuestionSeed instances, total = 10 * target_per_tool.
        """
        all_seeds: list[QuestionSeed] = []
        seed_counter = 0

        tool_generators = [
            ("get_telemetry", self._combos_get_telemetry),
            ("get_crew_status", self._combos_get_crew_status),
            ("get_module_status", self._combos_get_module_status),
            ("send_alert", self._combos_send_alert),
            ("send_message", self._combos_send_message),
            ("schedule_maintenance", self._combos_schedule_maintenance),
            ("activate_protocol", self._combos_activate_protocol),
            ("control_system", self._combos_control_system),
            ("calculate_trajectory", self._combos_calculate_trajectory),
            ("request_supply", self._combos_request_supply),
            ("no_action", self._combos_no_action),
        ]

        for tool_name, combo_fn in tool_generators:
            combos = combo_fn()
            seeds = self._sample_to_target(tool_name, combos)
            for idx, (params, phrasing_index) in enumerate(seeds):
                tool_call = build_tool_call(tool_name, params)
                difficulty = assign_difficulty(tool_name, params, phrasing_index)
                split = _assign_split(idx, self._target)
                context_facts = self._build_context_facts(tool_name, params)
                doc_ids = match_docs(tool_name, params)

                seed = QuestionSeed(
                    seed_id=f"seed_{seed_counter:05d}",
                    tool_name=tool_name,
                    tool_params=params,
                    tool_call=tool_call,
                    difficulty=difficulty,
                    context_facts=context_facts,
                    doc_ids=doc_ids,
                    split=split,
                    phrasing_index=phrasing_index,
                )
                all_seeds.append(seed)
                seed_counter += 1

            logger.info(f"Generated {len(seeds)} seeds for {tool_name}")

        logger.info(f"Total seeds generated: {len(all_seeds)}")
        return all_seeds

    def _sample_to_target(
        self,
        tool_name: str,
        combos: list[dict],
    ) -> list[tuple[dict, int]]:
        """Sample or expand combos to exactly target_per_tool (combo, phrasing_index) pairs.

        If len(combos) >= target: sample without replacement (no duplicates).
        If len(combos) < target: cycle through combos assigning increasing phrasing_index
        until target is reached.

        Args:
            tool_name: Tool name for logging.
            combos: All unique parameter dicts for this tool.

        Returns:
            List of (params_dict, phrasing_index) tuples, length == target_per_tool.
        """
        if not combos:
            logger.warning(f"No combos for {tool_name}, returning empty list")
            return []

        result: list[tuple[dict, int]] = []

        if len(combos) >= self._target:
            sampled = self._rng.sample(combos, self._target)
            result = [(c, 0) for c in sampled]
        else:
            full_cycles = self._target // len(combos)
            remainder = self._target % len(combos)
            for cycle in range(full_cycles):
                for combo in combos:
                    result.append((combo, cycle))
            extra = self._rng.sample(combos, remainder)
            for combo in extra:
                result.append((combo, full_cycles))

        self._rng.shuffle(result)
        return result

    def _build_context_facts(self, tool_name: str, params: dict) -> list[str]:
        """Build context_facts list for a seed from skeleton data.

        Args:
            tool_name: Tool being seeded.
            params: Parameter dict for this seed.

        Returns:
            List of relevant fact strings extracted from the skeleton.
        """
        facts: list[str] = []
        module = params.get("module", "")

        if module:
            facts.extend(_build_module_context(module, self._skeleton))

        if tool_name == "activate_protocol":
            protocol_id = params.get("protocol_id", "")
            facts.extend(_build_protocol_context(protocol_id, self._skeleton))

        elif tool_name == "send_alert":
            reason = params.get("reason", "")
            severity = params.get("severity", "")
            facts.append(f"Alert reason: {reason}, severity: {severity}")

        elif tool_name == "calculate_trajectory":
            maneuver = params.get("maneuver", "")
            urgency = params.get("urgency", "")
            facts.append(f"Maneuver: {maneuver}, urgency: {urgency}")
            facts.append("Station altitude: 408 km LEO")

        elif tool_name == "no_action":
            facts.extend([
                "no_action is the correct response when the query is informational, historical, or requires no system interaction.",
                "MASA station: Kuntur, 6 modules, current mission Cóndor-7",
            ])

        return facts

    def _combos_get_telemetry(self) -> list[dict]:
        return [
            {"module": m, "metric": me, "timeframe_hours": t}
            for m, me, t in itertools.product(_MODULES, _METRICS, _TIMEFRAMES)
        ]

    def _combos_get_crew_status(self) -> list[dict]:
        return [
            {"module": m, "info": i}
            for m, i in itertools.product(_MODULES, _CREW_INFO)
        ]

    def _combos_get_module_status(self) -> list[dict]:
        return [
            {"module": m, "system": s}
            for m, s in itertools.product(_MODULES, _MODULE_SYSTEMS)
        ]

    def _combos_send_alert(self) -> list[dict]:
        return [
            {"module": m, "severity": sev, "reason": r}
            for m, sev, r in itertools.product(_MODULES, _SEVERITIES, _REASONS)
        ]

    def _combos_send_message(self) -> list[dict]:
        return [
            {"recipient": rec, "priority": p}
            for rec, p in itertools.product(_RECIPIENTS, _PRIORITIES_MSG)
        ]

    def _combos_schedule_maintenance(self) -> list[dict]:
        return [
            {"module": m, "task": t, "priority": p}
            for m, t, p in itertools.product(_MODULES, _TASKS, _PRIORITIES_MAINT)
        ]

    def _combos_activate_protocol(self) -> list[dict]:
        return [
            {"protocol_id": pid, "scope": scope}
            for pid, scope in itertools.product(_PROTOCOLS, _SCOPES)
        ]

    def _combos_control_system(self) -> list[dict]:
        return [
            {"module": m, "system": s, "action": a}
            for m, s, a in itertools.product(_MODULES, _CTRL_SYSTEMS, _ACTIONS)
        ]

    def _combos_calculate_trajectory(self) -> list[dict]:
        return [
            {"maneuver": man, "urgency": urg}
            for man, urg in itertools.product(_MANEUVERS, _URGENCIES_TRAJ)
        ]

    def _combos_request_supply(self) -> list[dict]:
        return [
            {"category": cat, "urgency": urg}
            for cat, urg in itertools.product(_SUPPLY_CATEGORIES, _URGENCIES_SUPPLY)
        ]

    def _combos_no_action(self) -> list[dict]:
        no_action_types = [
            {"topic": "module_function"},
            {"topic": "crew_profile"},
            {"topic": "mission_history"},
            {"topic": "agency_info"},
            {"topic": "general_procedure"},
            {"topic": "station_overview"},
            {"topic": "scientific_program"},
            {"topic": "training_info"},
        ]
        return no_action_types
