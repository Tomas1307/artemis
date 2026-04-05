"""Generate RAG-dependent questions for non-protocol documents.

Expands RAG coverage from 6 protocol docs to ~18 docs by creating questions
where tool parameters are only derivable from document content.

Usage:
    python app/scripts/generate_expanded_rag.py
    python app/scripts/generate_expanded_rag.py --dry-run
"""

import csv
import io
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from app.llms.llm_nvidia import NvidiaLLM

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_CSV = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "data.csv"
GOLD_JSON = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "gold_standard.json"
TEST_CSV = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_queries.csv"
TEST_GOLD = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "test_gold_standard.json"
SAMPLE_SUB = PROJECT_ROOT / "proyecto_artemis" / "evaluacion" / "sample_submission.csv"

SYSTEM_PROMPT = """You are a MASA control center operator simulator. Generate realistic queries
that describe an operational situation requiring action. The query must:
- Describe sensor readings, crew reports, or operational conditions naturally
- Include the specific numeric value given in the scenario
- NOT reveal what tool/action should be taken (the operator is ASKING what to do)
- Use accented module names: Cóndor, Quetzal, Jaguar, Colibrí, Vicuña, Tucán
- Use crew names naturally: Commander Reyes, Pilot Valdivia, Specialist Nakamura,
  Specialist Al-Hassan, Engineer Kozlov, Medical Officer Mendoza
- Be 15-40 words, natural conversational English
- Each query on its own line, no numbering, no bullets, no quotes
- Generate exactly {count} diverse phrasings for the SAME scenario"""

SEEDS = [
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='condor',system='cooling',action='increase')", "scenario": "Cóndor module temperature has risen to 42°C. Per DOC-013, medium threshold is 40°C — cooling must be increased.", "count": 4},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='quetzal',system='cooling',action='increase')", "scenario": "Quetzal module temperature reading 48°C. Per DOC-013, medium threshold is 45°C — cooling must be increased.", "count": 4},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='jaguar',system='cooling',action='increase')", "scenario": "Jaguar module temperature at 41°C. Per DOC-013, medium threshold is 38°C — cooling must be increased.", "count": 4},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='colibri',system='cooling',action='increase')", "scenario": "Colibrí module temperature reached 57°C. Per DOC-013, medium threshold is 55°C — cooling must be increased.", "count": 4},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='vicuna',system='cooling',action='increase')", "scenario": "Vicuña storage area temperature at 58°C. Per DOC-013, medium threshold is 55°C — cooling must be increased.", "count": 4},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='tucan',system='cooling',action='increase')", "scenario": "Tucán quarters temperature reading 43°C. Per DOC-013, medium threshold is 40°C — cooling must be increased.", "count": 4},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='condor',system='cooling',action='activate')", "scenario": "Cóndor temperature rising to 32°C, above normal 20-24°C range. Per DOC-013, activate cooling.", "count": 3},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='jaguar',system='cooling',action='activate')", "scenario": "Jaguar temperature climbing to 30°C, above normal 19-23°C range. Per DOC-013, activate cooling.", "count": 3},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='tucan',system='cooling',action='activate')", "scenario": "Tucán temperature at 28°C, above normal 20-25°C range. Per DOC-013, activate cooling.", "count": 3},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='tucan',system='heating',action='activate')", "scenario": "Tucán temperature dropped to 17°C, below normal 20-25°C range. Per DOC-013, activate heating.", "count": 3},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='jaguar',system='heating',action='activate')", "scenario": "Jaguar temperature dropped to 16°C, below normal 19-23°C range. Per DOC-013, activate heating.", "count": 3},
    {"doc_id": "MASA-DOC-013", "tool_call": "control_system(module='condor',system='heating',action='activate')", "scenario": "Cóndor temperature reading 17°C, below normal 20-24°C range. Per DOC-013, activate heating.", "count": 3},

    {"doc_id": "MASA-DOC-014", "tool_call": "control_system(module='condor',system='lighting',action='deactivate')", "scenario": "Cóndor power load at 94% of capacity. Per DOC-014, at 93% shed Priority 3 loads — deactivate non-critical lighting.", "count": 4},
    {"doc_id": "MASA-DOC-014", "tool_call": "control_system(module='quetzal',system='lighting',action='deactivate')", "scenario": "Quetzal power consumption hitting 95%. Per DOC-014, above 93% medium threshold — deactivate non-essential lighting.", "count": 4},
    {"doc_id": "MASA-DOC-014", "tool_call": "control_system(module='jaguar',system='lighting',action='deactivate')", "scenario": "Jaguar power load approaching 94%. Per DOC-014, above 93% — deactivate non-critical lighting to reduce load.", "count": 3},
    {"doc_id": "MASA-DOC-014", "tool_call": "control_system(module='tucan',system='lighting',action='deactivate')", "scenario": "Tucán power consumption at 96%. Per DOC-014, well above 93% medium threshold — shed Priority 3 (lighting).", "count": 3},
    {"doc_id": "MASA-DOC-014", "tool_call": "control_system(module='colibri',system='lighting',action='deactivate')", "scenario": "Colibrí power load reading 95%. Per DOC-014, exceeds 93% — deactivate non-essential lighting.", "count": 3},
    {"doc_id": "MASA-DOC-014", "tool_call": "control_system(module='vicuna',system='lighting',action='deactivate')", "scenario": "Vicuña power consumption at 94%. Per DOC-014, exceeds 93% medium threshold — shed Priority 3.", "count": 3},

    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='condor',system='ventilation',action='increase')", "scenario": "Cóndor humidity reading 68%. Per DOC-017, above 60% normal range — increase ventilation.", "count": 4},
    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='quetzal',system='ventilation',action='increase')", "scenario": "Quetzal humidity at 73%. Per DOC-017, above 60% — increase ventilation.", "count": 4},
    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='jaguar',system='ventilation',action='increase')", "scenario": "Jaguar humidity dropped to 34%. Per DOC-017, below 40% normal — increase ventilation to normalize.", "count": 4},
    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='tucan',system='ventilation',action='increase')", "scenario": "Tucán humidity climbing to 66%. Per DOC-017, above 60% normal range — increase ventilation.", "count": 3},
    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='vicuna',system='ventilation',action='increase')", "scenario": "Vicuña humidity reading 28%. Per DOC-017, below 30% medium-severity — increase ventilation urgently.", "count": 3},
    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='colibri',system='ventilation',action='increase')", "scenario": "Colibrí humidity at 72%. Per DOC-017, 71-80% is medium severity — increase ventilation.", "count": 3},
    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='condor',system='filtration',action='activate')", "scenario": "VOC sensors in Cóndor detecting elevated chemical readings. Per DOC-017, activate filtration.", "count": 3},
    {"doc_id": "MASA-DOC-017", "tool_call": "control_system(module='quetzal',system='filtration',action='activate')", "scenario": "Chemical odor reported in Quetzal by Nakamura. Per DOC-017, VOC detection triggers filtration activation.", "count": 3},

    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='jaguar',task='filter_replacement',priority='urgent')", "scenario": "Jaguar HEPA filter differential pressure spiked — Kozlov says it's past service life. Per DOC-019, immediate filter replacement needed.", "count": 5},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='condor',task='system_calibration',priority='routine')", "scenario": "Monthly calibration window approaching for Cóndor systems. Per DOC-019 MASA-OPS-014, routine calibration due.", "count": 4},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='colibri',task='software_update',priority='routine')", "scenario": "New communication protocol patch available for Colibrí. Per DOC-019 MASA-OPS-014, monthly software update window.", "count": 4},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='jaguar',task='sensor_repair',priority='urgent')", "scenario": "Jaguar oxygen sensor showing drift beyond tolerance. Per DOC-019, sensors must maintain accuracy — urgent repair.", "count": 4},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='condor',task='power_cell_swap',priority='urgent')", "scenario": "Cóndor power cell capacity dropped below minimum during weekly check. Per DOC-019, urgent swap needed.", "count": 4},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='quetzal',task='system_calibration',priority='routine')", "scenario": "Quetzal workstation sensors due for bi-weekly calibration. Per DOC-019, routine maintenance scheduled.", "count": 3},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='tucan',task='sensor_repair',priority='urgent')", "scenario": "Tucán medical bay temperature sensor giving erratic readings. Per DOC-019, urgent sensor repair.", "count": 3},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='vicuna',task='hull_inspection',priority='urgent')", "scenario": "Micro-debris impact detected on Vicuña hull. Per DOC-019, urgent hull inspection required.", "count": 3},
    {"doc_id": "MASA-DOC-019", "tool_call": "schedule_maintenance(module='colibri',task='system_calibration',priority='urgent')", "scenario": "Colibrí navigation accuracy degraded since last check. Per DOC-019, urgent system calibration needed.", "count": 3},

    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='medical',urgency='expedited')", "scenario": "Medical supply reserves in Vicuña at 29% capacity. Per DOC-039, below 30% threshold triggers expedited request.", "count": 5},
    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='food',urgency='expedited')", "scenario": "Food reserves at 27% of capacity. Per DOC-039, below 30% triggers expedited resupply.", "count": 5},
    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='spare_parts',urgency='expedited')", "scenario": "Spare parts inventory at 25% of projected annual consumption. Per DOC-039, below 30% triggers expedited request.", "count": 4},
    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='medical',urgency='emergency')", "scenario": "Medical supplies critically low at 12% capacity after emergency use. Per DOC-039, below 15% triggers emergency request.", "count": 4},
    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='food',urgency='emergency')", "scenario": "Food reserves dropped to 13% — crew rationing in effect. Per DOC-039, below 15% triggers emergency.", "count": 4},
    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='spare_parts',urgency='emergency')", "scenario": "Spare parts at 10% after multiple system repairs. Per DOC-039, below 15% is emergency level.", "count": 3},
    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='fuel',urgency='expedited')", "scenario": "Fuel reserves for backup systems at 28%. Per DOC-039, below 30% threshold — expedited resupply.", "count": 3},
    {"doc_id": "MASA-DOC-039", "tool_call": "request_supply(category='scientific',urgency='routine')", "scenario": "Scientific equipment consumables projected to last 75 days. Per DOC-039, above 60-day depletion — routine request.", "count": 3},
    {"doc_id": "MASA-DOC-021", "tool_call": "request_supply(category='equipment',urgency='expedited')", "scenario": "Critical docking equipment reserves at 26%. Per DOC-021, below 30% triggers priority resupply.", "count": 3},
    {"doc_id": "MASA-DOC-021", "tool_call": "request_supply(category='fuel',urgency='emergency')", "scenario": "Fuel reserves at 11% after unplanned orbital maneuver. Per DOC-021, below 15% — emergency request.", "count": 3},

    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='orbit_adjustment',urgency='immediate')", "scenario": "Station altitude drifted to 414 km — 6 km above nominal 408 km. Per DOC-040, >5 km deviation requires immediate adjustment.", "count": 5},
    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='orbit_adjustment',urgency='planned')", "scenario": "Station altitude at 410.5 km — 2.5 km above nominal. Per DOC-040, 1-3 km deviation is medium, planned adjustment.", "count": 5},
    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='debris_avoidance',urgency='immediate')", "scenario": "Tracking shows debris with collision probability of 2.3 × 10⁻⁴. Per DOC-040, exceeds 10⁻⁴ threshold — immediate avoidance.", "count": 5},
    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='orbit_adjustment',urgency='immediate')", "scenario": "Velocity delta measured at 12.3 m/s from nominal. Per DOC-040, >10 m/s requires immediate orbit correction.", "count": 4},
    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='orbit_adjustment',urgency='planned')", "scenario": "Velocity delta showing 3.8 m/s deviation. Per DOC-040, 2-5 m/s is medium severity — planned correction.", "count": 4},
    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='station_keeping',urgency='planned')", "scenario": "Routine altitude check shows station at 409.2 km — minor drift from 408 km. Per DOC-040, standard station-keeping.", "count": 4},
    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='debris_avoidance',urgency='immediate')", "scenario": "New debris object detected with collision probability 5.7 × 10⁻⁴. Per DOC-040, far exceeds 10⁻⁴ limit.", "count": 3},
    {"doc_id": "MASA-DOC-040", "tool_call": "calculate_trajectory(maneuver='docking',urgency='planned')", "scenario": "Supply vessel approaching for scheduled docking at Vicuña port. Per DOC-040, standard docking trajectory calculation.", "count": 3},

    {"doc_id": "MASA-DOC-001", "tool_call": "get_telemetry(module='condor',metric='temperature',timeframe_hours=1)", "scenario": "Cóndor thermal alert triggered — need immediate temperature data. Per DOC-001, command module has 20-24°C range.", "count": 3},
    {"doc_id": "MASA-DOC-002", "tool_call": "get_telemetry(module='quetzal',metric='radiation',timeframe_hours=6)", "scenario": "Nakamura reports experiment sensitivity to radiation in Quetzal. Need last 6 hours of radiation data for the lab.", "count": 3},
    {"doc_id": "MASA-DOC-003", "tool_call": "get_telemetry(module='jaguar',metric='oxygen',timeframe_hours=12)", "scenario": "Kozlov wants to verify Jaguar O2 generation efficiency over the last half-day. Per DOC-003, life support module.", "count": 3},
    {"doc_id": "MASA-DOC-004", "tool_call": "get_telemetry(module='colibri',metric='power',timeframe_hours=24)", "scenario": "Valdivia needs Colibrí power consumption trend over the last day — antenna array drawing heavy load.", "count": 3},
    {"doc_id": "MASA-DOC-005", "tool_call": "get_telemetry(module='vicuna',metric='humidity',timeframe_hours=12)", "scenario": "Cargo moisture readings seem off in Vicuña. Need 12-hour humidity trend for the storage module.", "count": 3},
    {"doc_id": "MASA-DOC-006", "tool_call": "get_telemetry(module='tucan',metric='temperature',timeframe_hours=6)", "scenario": "Mendoza reports the medical bay feels warmer than usual. Pull Tucán temperature data for the last 6 hours.", "count": 3},
    {"doc_id": "MASA-DOC-003", "tool_call": "get_module_status(module='jaguar',system='life_support')", "scenario": "Kozlov wants a full status check on Jaguar life support before the bi-weekly maintenance cycle.", "count": 3},
    {"doc_id": "MASA-DOC-004", "tool_call": "get_module_status(module='colibri',system='communications')", "scenario": "Valdivia needs Colibrí comms system status before the next ground contact window.", "count": 3},
    {"doc_id": "MASA-DOC-001", "tool_call": "get_module_status(module='condor',system='power')", "scenario": "Reyes requesting Cóndor power system status — load balancing check before software update window.", "count": 3},
    {"doc_id": "MASA-DOC-014", "tool_call": "get_module_status(module='quetzal',system='power')", "scenario": "Quetzal experiments drawing heavy power — need power system status per DOC-014 load thresholds.", "count": 3},
    {"doc_id": "MASA-DOC-005", "tool_call": "get_module_status(module='vicuna',system='structural')", "scenario": "Post-docking structural integrity check needed for Vicuña. Standard procedure per DOC-005.", "count": 3},
]


def generate_queries(llm: NvidiaLLM, seed: dict, existing: set) -> list[str]:
    """Generate diverse query phrasings for a single seed scenario.

    Args:
        llm: LLM provider instance.
        seed: Seed dict with scenario, count, doc_id, tool_call.
        existing: Set of existing queries to avoid duplicates.

    Returns:
        List of unique query strings.
    """
    scenario_text = seed["scenario"].split(". Per ")[0] + "."
    prompt = f"Scenario: {scenario_text}\n\nGenerate exactly {seed['count']} diverse operator queries describing this situation."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(count=seed["count"])},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        try:
            response = llm.generate(messages, temperature=0.9, max_tokens=2000)
            lines = [line.strip().strip("-").strip("0123456789.").strip() for line in response.strip().split("\n")]
            queries = [line for line in lines if len(line.split()) >= 8 and line not in existing]
            logger.info(f"  [{seed['doc_id']}] {seed['tool_call'][:40]}... attempt {attempt+1}: {len(queries)} queries")
            return queries[:seed["count"]]
        except Exception as exc:
            logger.warning(f"  [{seed['doc_id']}] attempt {attempt+1} failed: {exc}")
            time.sleep(3)

    return []


def main() -> None:
    """Generate expanded RAG questions and add to all data files."""
    dry_run = "--dry-run" in sys.argv
    random.seed(42)

    total_expected = sum(s["count"] for s in SEEDS)
    logger.info(f"Seeds: {len(SEEDS)}, expected queries: {total_expected}")

    with open(DATA_CSV, encoding="utf-8") as f:
        data_rows = list(csv.DictReader(f))
    with open(GOLD_JSON, encoding="utf-8") as f:
        gold = json.load(f)
    with open(TEST_CSV, encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    with open(TEST_GOLD, encoding="utf-8") as f:
        test_gold = json.load(f)

    existing_queries = set(r["query"] for r in data_rows)
    existing_queries.update(r["query"] for r in test_rows)

    llm = NvidiaLLM()
    all_generated: list[dict] = []

    for i, seed in enumerate(SEEDS, 1):
        logger.info(f"[{i}/{len(SEEDS)}] {seed['doc_id']} -> {seed['tool_call'][:50]}")
        queries = generate_queries(llm, seed, existing_queries)
        for q in queries:
            existing_queries.add(q)
            all_generated.append({
                "query": q,
                "tool_call": seed["tool_call"],
                "tool_name": seed["tool_call"].split("(")[0],
                "doc_id": seed["doc_id"],
                "seed_type": "rag",
            })
        time.sleep(0.5)

    logger.info(f"Total generated: {len(all_generated)}")

    if dry_run:
        for item in all_generated[:15]:
            logger.info(f"  {item['doc_id']} | {item['tool_call'][:50]} | {item['query'][:70]}")
        return

    random.shuffle(all_generated)

    test_count = max(1, len(all_generated) // 10)
    test_items = all_generated[:test_count]
    data_items = all_generated[test_count:]

    max_q = max(int(r["id"].split("-")[1]) for r in data_rows)
    max_t = max(int(r["id"].split("-")[1]) for r in test_rows)

    logger.info(f"Splitting: {len(data_items)} data + {len(test_items)} test")

    new_data = []
    new_gold = []
    for i, item in enumerate(data_items):
        qid = f"Q-{max_q + 1 + i:05d}"
        new_data.append({"id": qid, "query": item["query"], "tool_call": item["tool_call"]})
        new_gold.append({
            "question_id": qid,
            "query": item["query"],
            "tool_call": item["tool_call"],
            "tool_name": item["tool_name"],
            "seed_type": "rag",
            "doc_id": item["doc_id"],
            "protocol_id": None,
        })

    new_test = []
    new_test_gold = []
    new_sub = []
    for i, item in enumerate(test_items):
        tid = f"T-{max_t + 1 + i:05d}"
        new_test.append({"id": tid, "query": item["query"]})
        new_test_gold.append({
            "question_id": tid,
            "query": item["query"],
            "tool_call": item["tool_call"],
            "tool_name": item["tool_name"],
            "seed_type": "rag",
            "doc_id": item["doc_id"],
            "protocol_id": None,
        })
        new_sub.append({"id": tid, "tool_call": "no_action"})

    all_data = data_rows + new_data
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["id", "query", "tool_call"])
    writer.writeheader()
    writer.writerows(all_data)
    DATA_CSV.write_text(out.getvalue(), encoding="utf-8", newline="")
    logger.info(f"data.csv: {len(data_rows)} -> {len(all_data)}")

    all_gold = gold + new_gold
    GOLD_JSON.write_text(json.dumps(all_gold, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"gold_standard.json: {len(gold)} -> {len(all_gold)}")

    all_test = test_rows + new_test
    out2 = io.StringIO()
    writer2 = csv.DictWriter(out2, fieldnames=["id", "query"])
    writer2.writeheader()
    writer2.writerows(all_test)
    TEST_CSV.write_text(out2.getvalue(), encoding="utf-8", newline="")
    logger.info(f"test_queries.csv: {len(test_rows)} -> {len(all_test)}")

    all_tg = test_gold + new_test_gold
    TEST_GOLD.write_text(json.dumps(all_tg, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"test_gold_standard.json: {len(test_gold)} -> {len(all_tg)}")

    with open(SAMPLE_SUB, encoding="utf-8") as f:
        existing_sub = list(csv.DictReader(f))
    all_sub = existing_sub + new_sub
    out3 = io.StringIO()
    writer3 = csv.DictWriter(out3, fieldnames=["id", "tool_call"])
    writer3.writeheader()
    writer3.writerows(all_sub)
    SAMPLE_SUB.write_text(out3.getvalue(), encoding="utf-8", newline="")
    logger.info(f"sample_submission.csv: {len(existing_sub)} -> {len(all_sub)}")

    from collections import Counter
    doc_counts = Counter(item["doc_id"] for item in all_generated)
    tool_counts = Counter(item["tool_name"] for item in all_generated)
    print(f"\n=== RESULTS ===")
    print(f"Generated: {len(all_generated)} ({len(data_items)} data + {len(test_items)} test)")
    print(f"Docs covered: {sorted(doc_counts.keys())}")
    print(f"Per doc: {dict(sorted(doc_counts.items()))}")
    print(f"Per tool: {dict(sorted(tool_counts.items()))}")


if __name__ == "__main__":
    main()
