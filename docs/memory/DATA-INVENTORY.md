# Data Inventory — proyecto_artemis

Last updated: 2026-04-05 (Session 9)

## File Counts

| File | Count | Description |
|------|-------|-------------|
| data.csv | 3188 rows | Student training data (id, query, tool_call) |
| gold_standard.json | 3188 entries | Internal answer key with metadata (not for students) |
| test_queries.csv | 362 rows | Kaggle evaluation queries (id, query — no answers) |
| test_gold_standard.json | 362 entries | Test answers for grading (not for students) |
| sample_submission.csv | 362 rows | Example submission format (all no_action placeholder) |
| consultas_centro_control.json | 800 pairs | Encoder training hints (query, doc_id) |
| tools_definition.json | 10 tools | Tool definitions with enum params |
| documentos_masa.json | 54 docs | Knowledge base index |

## Tool Distribution

| Tool | data.csv | % | test | % |
|------|----------|---|------|---|
| activate_protocol | 934 | 29.3% | 107 | 29.6% |
| send_alert | 796 | 25.0% | 92 | 25.4% |
| no_action | 394 | 12.4% | 45 | 12.4% |
| control_system | 190 | 6.0% | 19 | 5.2% |
| request_supply | 140 | 4.4% | 17 | 4.7% |
| schedule_maintenance | 137 | 4.3% | 16 | 4.4% |
| calculate_trajectory | 136 | 4.3% | 17 | 4.7% |
| get_telemetry | 125 | 3.9% | 13 | 3.6% |
| get_module_status | 123 | 3.9% | 12 | 3.3% |
| get_crew_status | 108 | 3.4% | 12 | 3.3% |
| send_message | 105 | 3.3% | 12 | 3.3% |

RAG-dependent tools (activate_protocol + send_alert) still dominate at ~54%. control_system, request_supply, schedule_maintenance, and calculate_trajectory now have expanded RAG variants. no_action at 12.4% (within 10-15% spec).

## RAG Document Coverage

RAG questions now span **19 documents** (up from 6):

| Doc ID | Type | data.csv | test | Content |
|--------|------|----------|------|---------|
| MASA-DOC-007 | protocol_group | 762 | 88 | Atmospheric Emergency Protocols |
| MASA-DOC-009 | protocol_group | 302 | 32 | Radiation Safety Protocols |
| MASA-DOC-011 | protocol_group | 263 | 36 | System Failure Response Protocols |
| MASA-DOC-010 | protocol_group | 250 | 26 | Structural/Mechanical Safety Protocols |
| MASA-DOC-008 | protocol_group | 148 | 16 | Thermal/Fire Safety Protocols |
| MASA-DOC-013 | system_guide | 39 | 3 | Thermal Management System Guide |
| MASA-DOC-019 | operational_procedure | 29 | 4 | Maintenance & Calibration Procedures |
| MASA-DOC-040 | cross_cutting | 28 | 5 | Orbital Navigation & Trajectory Guide |
| MASA-DOC-039 | cross_cutting | 27 | 4 | Inventory Management & Supply Chain |
| MASA-DOC-017 | system_guide | 26 | 1 | Ventilation & Filtration Systems |
| MASA-DOC-014 | system_guide | 20 | 3 | Power Distribution & Management |
| MASA-DOC-001 | module_manual | 6 | 0 | Cóndor Module Manual |
| MASA-DOC-003 | module_manual | 6 | 0 | Jaguar Module Manual |
| MASA-DOC-004 | module_manual | 6 | 0 | Colibrí Module Manual |
| MASA-DOC-005 | module_manual | 6 | 0 | Vicuña Module Manual |
| MASA-DOC-012 | protocol_group | 5 | 1 | Critical Response & Evacuation |
| MASA-DOC-021 | operational_procedure | 5 | 1 | Docking, Cargo & Supply Procedures |
| MASA-DOC-006 | module_manual | 3 | 0 | Tucán Module Manual |
| MASA-DOC-002 | module_manual | 2 | 1 | Quetzal Module Manual |
| **Total RAG** | | **1933** | **221** | |

## Full Question Distribution by Document

| Doc ID | Train | Test | Total | Content |
|--------|-------|------|-------|---------|
| NO_DOC (direct) | 1255 | 141 | 1396 | Direct questions — no doc needed |
| MASA-DOC-007 | 762 | 88 | 850 | Atmospheric Emergency Protocols |
| MASA-DOC-009 | 302 | 32 | 334 | Radiation Safety Protocols |
| MASA-DOC-011 | 263 | 36 | 299 | System Failure Response |
| MASA-DOC-010 | 250 | 26 | 276 | Structural/Mechanical Safety |
| MASA-DOC-008 | 148 | 16 | 164 | Thermal/Fire Safety |
| MASA-DOC-013 | 39 | 3 | 42 | Thermal Management System Guide |
| MASA-DOC-019 | 29 | 4 | 33 | Maintenance & Calibration |
| MASA-DOC-040 | 28 | 5 | 33 | Orbital Navigation & Trajectory |
| MASA-DOC-039 | 27 | 4 | 31 | Inventory & Supply Chain |
| MASA-DOC-017 | 26 | 1 | 27 | Ventilation & Filtration |
| MASA-DOC-014 | 20 | 3 | 23 | Power Distribution |
| MASA-DOC-001 | 6 | 0 | 6 | Cóndor Module Manual |
| MASA-DOC-003 | 6 | 0 | 6 | Jaguar Module Manual |
| MASA-DOC-004 | 6 | 0 | 6 | Colibrí Module Manual |
| MASA-DOC-005 | 6 | 0 | 6 | Vicuña Module Manual |
| MASA-DOC-012 | 5 | 1 | 6 | Critical Response & Evacuation |
| MASA-DOC-021 | 5 | 1 | 6 | Docking, Cargo & Supply |
| MASA-DOC-006 | 3 | 0 | 3 | Tucán Module Manual |
| MASA-DOC-002 | 2 | 1 | 3 | Quetzal Module Manual |
| **TOTAL** | **3188** | **362** | **3550** | |

DOC-007/009/010/011 dominate by design — core protocol docs driving activate_protocol and send_alert. Expanded RAG docs (013–040) form the long tail. 35 docs have zero questions (noise corpus for retrieval).

## Seed Types

| Split | RAG | Direct | Total |
|-------|-----|--------|-------|
| data.csv | 1933 (60.6%) | 1255 (39.4%) | 3188 |
| test | 221 (61.0%) | 141 (39.0%) | 362 |

## Duplicates in data.csv

Intentionally left for students to clean as a data quality exercise.

- Duplicate groups: ~109 (same query text, same tool_call)
- Extra rows: ~170

58 conflicting duplicate groups (same query, different tool_call) were removed earlier in session 8.

## Protocol Usage

| Protocol | data.csv | test | Notes |
|----------|----------|------|-------|
| MASA-SEC-001 | 71 | 16 | |
| MASA-SEC-002 | 78 | 7 | |
| MASA-SEC-003 | 69 | 9 | |
| MASA-SEC-004 | 78 | 6 | |
| MASA-SEC-005 | 4 | 1 | Low usage |
| MASA-SEC-006 | 27 | 0 | Not in test |
| MASA-SEC-007 | 14 | 1 | |
| MASA-SEC-008 | 80 | 9 | |
| MASA-SEC-009 | 78 | 10 | |
| MASA-SEC-010 | 77 | 6 | |
| MASA-SEC-011 | 70 | 15 | |
| MASA-SEC-012 | 80 | 6 | |
| MASA-SEC-013 | 15 | 0 | Not in test |
| MASA-SEC-014 | 11 | 2 | |
| MASA-SEC-015 | 26 | 3 | |
| MASA-SEC-016 | 0 | 0 | **UNUSED** |
| MASA-SEC-017 | 80 | 8 | |
| MASA-SEC-018 | 75 | 8 | |
| MASA-SEC-019 | 1 | 0 | Low usage, not in test |
| MASA-SEC-020 | 0 | 0 | **UNUSED** |

18 of 20 protocols used. MASA-SEC-016 and 020 are dead protocols (realistic noise).

## Module Distribution in Tool Calls (data.csv)

Roughly even across all 6 modules (excludes no_action and tools without module param).

## Consultas Centro Control Coverage

- 800 pairs covering all 54 documents
- 15 pairs per doc (target), except MASA-DOC-012 with only 5
- Protocol docs (007-012): sampled from existing RAG questions
- Other 48 docs: generated via NVIDIA API

## Known Issues

| Issue | Status | Impact |
|-------|--------|--------|
| ~170 pure duplicate queries | Intentional | Students should deduplicate |
| MASA-SEC-016, 020 unused | Intentional | Realistic noise, consistent train/test |
| MASA-SEC-006, 013, 019 not in test | Noted | No impact on evaluation |
| ID gaps in data.csv | Cosmetic | From conflict dedup + multiple generation rounds |
| MASA-DOC-012 only 5 consultas pairs | Minor | Limited RAG questions for that protocol |
| Module manuals (001-006) not in test | Noted | Only in training data, 0 test questions for these docs |

## Intentional Student Cleaning Challenges

| Challenge | Description | What Students Must Do |
|-----------|-------------|----------------------|
| ~170 pure duplicate rows | Same query + same tool_call, intentionally left | Deduplicate before training |
| MASA-SEC-016, 020 unused | Defined in tools_definition.json, zero questions | Handle gracefully — don't crash on unseen protocols |
| MASA-SEC-006, 013, 019 train-only | Appear in data.csv but not in test | Don't overfit to protocol distribution |
| Non-sequential IDs | Gaps from conflict dedup + generation rounds | Don't assume contiguous IDs |
| Accent mismatch query↔enum | Queries use Vicuña/Colibrí/Tucán/Cóndor, enums use vicuna/colibri/tucan/condor | Normalize in tokenizer or post-processing |
| Module manuals (DOC-001–006) in train only | Training has questions for these docs, test has ~0 | Don't over-index on module manual retrieval |

## Accent Convention

- Query text: accented Spanish names (Vicuña, Colibrí, Tucán, Cóndor, Quetzal, Jaguar)
- Tool call enums: lowercase, no accents (vicuna, colibri, tucan, condor, quetzal, jaguar)
- Intentional OOV tokens that test the encoder/decoder
