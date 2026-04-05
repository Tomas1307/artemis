"""RAG-dependent prompt iteration test script.

Generates sample seeds from MASA-SEC protocol thresholds and tests
the rag_question_generation prompt against the LLM.

Usage:
    python scripts/test_rag_prompt_iteration.py --iteration 1
    python scripts/test_rag_prompt_iteration.py --iteration 1 --full
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llms.llm_nvidia import NvidiaLLM
from app.prompts.prompt_loader import prompt_loader


RAG_SEEDS = [
    {
        "id": "RAG-001",
        "tool_name": "send_alert",
        "module": "jaguar",
        "metric": "pressure",
        "sensor_reading": "83.7 kPa",
        "incident_description": "Pressure dropping in Jaguar life support module",
        "rag_requirement": "Severity depends on threshold: pressure < 85.0 kPa = critical (MASA-SEC-001), 85.0-89.9 kPa = high (MASA-SEC-011). Reading 83.7 is below 85.0, so severity = critical.",
        "expected_tool_call": "send_alert(module='jaguar',severity='critical',reason='pressure_drop')",
    },
    {
        "id": "RAG-002",
        "tool_name": "send_alert",
        "module": "quetzal",
        "metric": "pressure",
        "sensor_reading": "87.2 kPa",
        "incident_description": "Pressure anomaly detected in Quetzal science lab",
        "rag_requirement": "Severity depends on threshold: 85.0-89.9 kPa = high (MASA-SEC-011), < 85.0 = critical (MASA-SEC-001). Reading 87.2 is in 85.0-89.9 range, so severity = high.",
        "expected_tool_call": "send_alert(module='quetzal',severity='high',reason='pressure_drop')",
    },
    {
        "id": "RAG-003",
        "tool_name": "send_alert",
        "module": "colibri",
        "metric": "radiation",
        "sensor_reading": "6.3 mSv/hr",
        "incident_description": "Radiation spike detected in Colibrí communications module",
        "rag_requirement": "Severity depends on threshold: radiation > 5.0 mSv/hr = critical (MASA-SEC-004). Reading 6.3 exceeds 5.0, so severity = critical.",
        "expected_tool_call": "send_alert(module='colibri',severity='critical',reason='radiation_spike')",
    },
    {
        "id": "RAG-004",
        "tool_name": "send_alert",
        "module": "tucan",
        "metric": "radiation",
        "sensor_reading": "2.8 mSv/hr",
        "incident_description": "Elevated radiation readings in Tucán crew quarters",
        "rag_requirement": "Severity depends on threshold: radiation 1.1-5.0 mSv/hr = medium (MASA-SEC-012), > 5.0 = critical (MASA-SEC-004). Reading 2.8 is in 1.1-5.0 range, so severity = medium.",
        "expected_tool_call": "send_alert(module='tucan',severity='medium',reason='radiation_spike')",
    },
    {
        "id": "RAG-005",
        "tool_name": "send_alert",
        "module": "jaguar",
        "metric": "oxygen",
        "sensor_reading": "13.2%",
        "incident_description": "Oxygen concentration critically low in Jaguar",
        "rag_requirement": "Severity depends on threshold: O2 < 14.0% = critical (MASA-SEC-002), O2 14.0-15.9% = high (MASA-SEC-010). Reading 13.2 is below 14.0, so severity = critical.",
        "expected_tool_call": "send_alert(module='jaguar',severity='critical',reason='oxygen_leak')",
    },
    {
        "id": "RAG-006",
        "tool_name": "send_alert",
        "module": "condor",
        "metric": "oxygen",
        "sensor_reading": "15.4%",
        "incident_description": "Oxygen levels declining in Cóndor command module",
        "rag_requirement": "Severity depends on threshold: O2 14.0-15.9% = high (MASA-SEC-010), < 14.0% = critical (MASA-SEC-002). Reading 15.4 is in 14.0-15.9 range, so severity = high.",
        "expected_tool_call": "send_alert(module='condor',severity='high',reason='oxygen_leak')",
    },
    {
        "id": "RAG-007",
        "tool_name": "send_alert",
        "module": "quetzal",
        "metric": "temperature",
        "sensor_reading": "3.1 degrees per minute",
        "incident_description": "Rapid temperature rise detected in Quetzal laboratory",
        "rag_requirement": "Severity depends on threshold: temperature rise > 2.0 C/min = critical (MASA-SEC-003). Reading 3.1 exceeds 2.0, so severity = critical.",
        "expected_tool_call": "send_alert(module='quetzal',severity='critical',reason='abnormal_temperature')",
    },
    {
        "id": "RAG-008",
        "tool_name": "send_alert",
        "module": "vicuna",
        "metric": "power",
        "sensor_reading": "104.8V",
        "incident_description": "Voltage fluctuation detected on Vicuña power bus, below safe range",
        "rag_requirement": "Severity depends on threshold: voltage outside 106.2V-129.8V range = medium (MASA-SEC-018). Reading 104.8V is below 106.2V, so severity = medium.",
        "expected_tool_call": "send_alert(module='vicuna',severity='medium',reason='power_fluctuation')",
    },
    {
        "id": "RAG-009",
        "tool_name": "activate_protocol",
        "module": "colibri",
        "metric": "radiation",
        "sensor_reading": "6.8 mSv/hr",
        "incident_description": "Extreme radiation event across all modules, highest in Colibrí",
        "rag_requirement": "Protocol depends on reading: radiation > 5.0 mSv/hr triggers MASA-SEC-004 (Radiation Lockdown), scope = station_wide. Student must find this in MASA-DOC-009.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-004',scope='station_wide')",
    },
    {
        "id": "RAG-010",
        "tool_name": "activate_protocol",
        "module": "jaguar",
        "metric": "pressure",
        "sensor_reading": "78.3 kPa",
        "incident_description": "Severe pressure loss in Jaguar, potential hull breach",
        "rag_requirement": "Protocol depends on reading: pressure < 85.0 kPa triggers MASA-SEC-001 (Rapid Decompression Response), scope = station_wide. Student must find this in MASA-DOC-007.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-001',scope='station_wide')",
    },
    {
        "id": "RAG-011",
        "tool_name": "activate_protocol",
        "module": "quetzal",
        "metric": "co2",
        "sensor_reading": "1.8%",
        "incident_description": "CO2 concentration rising in Quetzal, crew reporting headaches",
        "rag_requirement": "Protocol depends on reading: CO2 > 1.5% triggers MASA-SEC-009 (Toxic Atmosphere Detection), scope = module_only. Student must find this in MASA-DOC-007.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-009',scope='module_only')",
    },
    {
        "id": "RAG-012",
        "tool_name": "activate_protocol",
        "module": "condor",
        "metric": "temperature",
        "sensor_reading": "2.4 degrees per minute",
        "incident_description": "Temperature rising rapidly in Cóndor, possible fire or thermal runaway",
        "rag_requirement": "Protocol depends on reading: temp rise > 2.0 C/min triggers MASA-SEC-003 (Fire and Thermal Runaway Suppression), scope = module_only. Student must find this in MASA-DOC-008.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-003',scope='module_only')",
    },
    {
        "id": "RAG-013",
        "tool_name": "send_alert",
        "module": "vicuna",
        "metric": "hull_stress",
        "sensor_reading": "88%",
        "incident_description": "Hull stress elevated in Vicuña docking module",
        "rag_requirement": "Severity depends on threshold: hull stress > 85% rated capacity = high (MASA-SEC-008), scope = module_only. Reading 88% exceeds 85%, so severity = high.",
        "expected_tool_call": "send_alert(module='vicuna',severity='high',reason='structural_damage')",
    },
    {
        "id": "RAG-014",
        "tool_name": "activate_protocol",
        "module": "tucan",
        "metric": "radiation",
        "sensor_reading": "3.2 mSv/hr",
        "incident_description": "Elevated radiation in crew quarters, solar activity index rising",
        "rag_requirement": "Protocol depends on reading: radiation 1.1-5.0 mSv/hr triggers MASA-SEC-012 (Solar Flare Advisory), scope = station_wide. Student must find this in MASA-DOC-009.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-012',scope='station_wide')",
    },
    {
        "id": "RAG-015",
        "tool_name": "send_alert",
        "module": "condor",
        "metric": "power",
        "sensor_reading": "98.2%",
        "incident_description": "Power consumption at near-maximum capacity across the station",
        "rag_requirement": "Severity depends on threshold: power > 97% rated capacity = critical (MASA-SEC-006). Reading 98.2% exceeds 97%, so severity = critical.",
        "expected_tool_call": "send_alert(module='condor',severity='critical',reason='power_fluctuation')",
    },
    {
        "id": "RAG-016",
        "tool_name": "activate_protocol",
        "module": "vicuna",
        "metric": "docking_pressure",
        "sensor_reading": "6.1 kPa",
        "incident_description": "Pressure differential anomaly at Vicuña docking port during active docking",
        "rag_requirement": "Protocol depends on reading: docking pressure differential > 5.0 kPa triggers MASA-SEC-013 (Docking Bay Pressure Anomaly), scope = module_only. Student must find this in MASA-DOC-010.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-013',scope='module_only')",
    },
    {
        "id": "RAG-017",
        "tool_name": "send_alert",
        "module": "jaguar",
        "metric": "oxygen",
        "sensor_reading": "14.8%",
        "incident_description": "Oxygen trending downward in Jaguar, now in warning band",
        "rag_requirement": "Severity depends on threshold: O2 14.0-15.9% = high (MASA-SEC-010). Reading 14.8 is in this range, so severity = high.",
        "expected_tool_call": "send_alert(module='jaguar',severity='high',reason='oxygen_leak')",
    },
    {
        "id": "RAG-018",
        "tool_name": "activate_protocol",
        "module": "jaguar",
        "metric": "oxygen",
        "sensor_reading": "12.8%",
        "incident_description": "Oxygen critically low in Jaguar, crew experiencing dizziness",
        "rag_requirement": "Protocol depends on reading: O2 < 14.0% triggers MASA-SEC-002 (Oxygen Depletion Response), scope = station_wide. Student must find this in MASA-DOC-007.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-002',scope='station_wide')",
    },
    {
        "id": "RAG-019",
        "tool_name": "send_alert",
        "module": "colibri",
        "metric": "communication",
        "sensor_reading": "35 minutes",
        "incident_description": "Ground control contact lost for over half an hour from Colibrí",
        "rag_requirement": "Severity depends on threshold: no contact > 30 minutes = high (MASA-SEC-007). 35 minutes exceeds 30, so severity = high.",
        "expected_tool_call": "send_alert(module='colibri',severity='high',reason='communication_loss')",
    },
    {
        "id": "RAG-020",
        "tool_name": "activate_protocol",
        "module": "quetzal",
        "metric": "airlock",
        "sensor_reading": "0.7 kPa per minute",
        "incident_description": "Airlock equalization rate exceeding safe limits in Quetzal",
        "rag_requirement": "Protocol depends on reading: equalization rate > 0.5 kPa/min triggers MASA-SEC-017 (Airlock Seal Failure), scope = module_only. Student must find this in MASA-DOC-010.",
        "expected_tool_call": "activate_protocol(protocol_id='MASA-SEC-017',scope='module_only')",
    },
]


SEVERITY_KEYWORDS = [
    "critical", "severity", "tier", "priority",
    "extreme", "catastrophic", "minor", "major", "serious",
    "severe", "urgent",
]

SEVERITY_WORD_PATTERNS = [
    "levels",
    " level ",
    " high ",
    " medium ",
    " low ",
]

PROTOCOL_KEYWORDS = [
    "MASA-SEC", "SEC-", "protocol id", "lockdown protocol",
    "decompression response", "radiation lockdown",
]

SCOPE_KEYWORDS = [
    "station-wide", "station wide", "module only", "module_only",
    "all modules", "entire station", "every module", "whole station",
    "across all",
]


def check_violations(query: str, seed: dict) -> list[str]:
    """Check generated query for rule violations."""
    violations = []
    q_lower = query.lower()

    for kw in SEVERITY_KEYWORDS:
        if kw.lower() in q_lower:
            violations.append(f"SEVERITY_LEAK: contains '{kw}'")

    for pattern in SEVERITY_WORD_PATTERNS:
        if pattern.lower() in f" {q_lower} ":
            violations.append(f"SEVERITY_LEAK: contains '{pattern.strip()}'")

    for kw in PROTOCOL_KEYWORDS:
        if kw.lower() in q_lower:
            violations.append(f"PROTOCOL_LEAK: contains '{kw}'")

    for kw in SCOPE_KEYWORDS:
        if kw.lower() in q_lower:
            violations.append(f"SCOPE_LEAK: contains '{kw}'")

    reading = seed["sensor_reading"]
    reading_number = ""
    for part in reading.split():
        try:
            float(part.replace("%", "").replace(",", ""))
            reading_number = part.replace("%", "").replace(",", "")
            break
        except ValueError:
            continue

    if reading_number and reading_number not in query:
        alt_found = False
        for part in reading.split():
            if part in query:
                alt_found = True
                break
        if not alt_found:
            violations.append(f"MISSING_READING: '{seed['sensor_reading']}' not found in query")

    word_count = len(query.split())
    if word_count < 12:
        violations.append(f"TOO_SHORT: {word_count} words (min 12)")
    if word_count > 45:
        violations.append(f"TOO_LONG: {word_count} words (max 45)")

    return violations


def main(iteration: int = 1, full: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"RAG PROMPT ITERATION {iteration} — TEST RUN")
    print(f"{'='*60}\n")

    llm = NvidiaLLM()

    system_message = prompt_loader.get_system_message_by_type("rag_query_generator")
    template = prompt_loader.get_prompt_template_by_type("rag_query_generator")
    config = prompt_loader.get_config_by_type("rag_query_generator")

    rng = random.Random(iteration * 42)

    if full:
        seeds = RAG_SEEDS
    else:
        seeds = rng.sample(RAG_SEEDS, min(10, len(RAG_SEEDS)))

    results = []
    total_violations = 0

    for i, seed in enumerate(seeds, 1):
        phrasing_idx = rng.randint(1, 15)

        prompt = template.format(
            tool_name=seed["tool_name"],
            module=seed["module"],
            metric=seed["metric"],
            sensor_reading=seed["sensor_reading"],
            incident_description=seed["incident_description"],
            rag_requirement=seed["rag_requirement"],
            phrasing_index=phrasing_idx,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        print(f"[{i:02d}/{len(seeds)}] {seed['id']} | {seed['tool_name']} | {seed['module']} | {seed['metric']}")
        print(f"       reading: {seed['sensor_reading']} | phrasing: {phrasing_idx}")
        print(f"       expected: {seed['expected_tool_call']}")

        try:
            response = llm.generate(messages, **config)
            query = response.strip().strip('"').strip("'")
            word_count = len(query.split())
            violations = check_violations(query, seed)
            total_violations += len(violations)

            status = "PASS" if not violations else "FAIL"
            print(f"       [{status}] QUERY ({word_count}w): {query}")
            if violations:
                for v in violations:
                    print(f"       !! {v}")
        except Exception as exc:
            query = f"ERROR: {exc}"
            violations = [f"LLM_ERROR: {exc}"]
            total_violations += 1
            print(f"       ERROR: {exc}")

        print()
        results.append({
            "seed_id": seed["id"],
            "tool_name": seed["tool_name"],
            "module": seed["module"],
            "metric": seed["metric"],
            "sensor_reading": seed["sensor_reading"],
            "expected_tool_call": seed["expected_tool_call"],
            "phrasing_index": phrasing_idx,
            "generated_query": query,
            "word_count": len(query.split()) if not query.startswith("ERROR") else 0,
            "violations": violations,
        })

    passed = sum(1 for r in results if not r["violations"])
    total = len(results)
    print(f"{'='*60}")
    print(f"RESULTS: {passed}/{total} passed ({passed/total*100:.0f}%), {total_violations} total violations")
    print(f"{'='*60}\n")

    output_path = Path(f"scripts/rag_iteration_{iteration:02d}_results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--full", action="store_true", help="Test all 20 seeds instead of 10")
    args = parser.parse_args()
    main(args.iteration, args.full)
