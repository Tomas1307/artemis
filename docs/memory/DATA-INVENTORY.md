# Data Inventory — proyecto_artemis

Last updated: 2026-04-05 (Session 8)

## File Counts

| File | Count | Description |
|------|-------|-------------|
| data.csv | 2985 rows | Student training data (id, query, tool_call) |
| gold_standard.json | 2985 entries | Internal answer key with metadata (not for students) |
| test_queries.csv | 340 rows | Kaggle evaluation queries (id, query — no answers) |
| test_gold_standard.json | 340 entries | Test answers for grading (not for students) |
| sample_submission.csv | 340 rows | Example submission format (all no_action placeholder) |
| consultas_centro_control.json | 800 pairs | Encoder training hints (query, doc_id) |
| tools_definition.json | 10 tools | Tool definitions with enum params |
| documentos_masa.json | 54 docs | Knowledge base index |

## Duplicates in data.csv

Intentionally left for students to clean as a data quality exercise.

- Unique queries: 2815
- Duplicate groups: 109 (same query text, same tool_call)
- Extra rows: 170
- Breakdown by tool:
  - no_action: 46 extra
  - calculate_trajectory: 42 extra
  - activate_protocol: 27 extra
  - get_crew_status: 18 extra
  - send_alert: 18 extra
  - send_message: 9 extra
  - request_supply: 7 extra
  - get_module_status: 3 extra

58 conflicting duplicate groups (same query, different tool_call) were removed in session 8. These were RAG questions where the LLM generated identical query text for different seeds.

## Seed Types

| Split | RAG | Direct | Total |
|-------|-----|--------|-------|
| data.csv | 1730 (58.0%) | 1255 (42.0%) | 2985 |
| test | 199 (58.5%) | 141 (41.5%) | 340 |

## Tool Distribution

| Tool | data.csv | % | test | % |
|------|----------|---|------|---|
| activate_protocol | 934 | 31.3% | 107 | 31.5% |
| send_alert | 796 | 26.7% | 92 | 27.1% |
| no_action | 394 | 13.2% | 45 | 13.2% |
| calculate_trajectory | 108 | 3.6% | 12 | 3.5% |
| control_system | 108 | 3.6% | 12 | 3.5% |
| get_crew_status | 108 | 3.6% | 12 | 3.5% |
| get_module_status | 108 | 3.6% | 12 | 3.5% |
| get_telemetry | 108 | 3.6% | 12 | 3.5% |
| request_supply | 108 | 3.6% | 12 | 3.5% |
| schedule_maintenance | 108 | 3.6% | 12 | 3.5% |
| send_message | 105 | 3.5% | 12 | 3.5% |

RAG-dependent tools (activate_protocol + send_alert) dominate at ~58% because each requires document retrieval to determine severity/protocol_id. no_action is the third largest at 13.2%, within the 10-15% spec target. The remaining 8 tools are evenly distributed at ~3.5% each.

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

18 of 20 protocols are used. MASA-SEC-016 and MASA-SEC-020 are never triggered — they exist in the knowledge base as realistic noise. MASA-SEC-006, 013, 019 appear in training but not test.

## RAG Questions per Document

Only 6 documents are referenced by RAG questions (the MASA-SEC protocol documents):

| Doc ID | data.csv | test | Content |
|--------|----------|------|---------|
| MASA-DOC-007 | 762 | 88 | Primary protocol group (pressure, O2, airlock, etc.) |
| MASA-DOC-008 | 148 | 16 | Thermal protocols |
| MASA-DOC-009 | 302 | 32 | Radiation protocols |
| MASA-DOC-010 | 250 | 26 | Oxygen/atmosphere protocols |
| MASA-DOC-011 | 263 | 36 | Pressure protocols |
| MASA-DOC-012 | 5 | 1 | Radiation (secondary) — very low coverage |
| **Total** | **1730** | **199** | |

The remaining 48 documents are referenced only in consultas_centro_control.json (encoder training) and as general knowledge base noise.

## Module Distribution in Tool Calls (data.csv)

Roughly even across all 6 modules (excludes no_action and tools without module param):

| Module | Count |
|--------|-------|
| colibri | 234 |
| condor | 227 |
| jaguar | 226 |
| vicuna | 217 |
| quetzal | 217 |
| tucan | 215 |

## Consultas Centro Control Coverage

- 800 pairs covering all 54 documents
- 15 pairs per doc (target), except MASA-DOC-012 with only 5
- Protocol docs (007-012): sampled from existing RAG questions
- Other 48 docs: generated via NVIDIA API

## Known Issues

| Issue | Status | Impact |
|-------|--------|--------|
| 170 pure duplicate queries | Intentional | Students should deduplicate |
| MASA-SEC-016, 020 unused | Intentional | Realistic noise, consistent train/test |
| MASA-SEC-006, 013, 019 not in test | Noted | No impact on evaluation |
| ID gaps in data.csv | Cosmetic | From conflict dedup, no impact |
| MASA-DOC-012 only 5 consultas pairs | Minor | Limited RAG questions for that protocol |

## Accent Convention

- Query text: accented Spanish names (Vicuña, Colibrí, Tucán, Cóndor, Quetzal, Jaguar)
- Tool call enums: lowercase, no accents (vicuna, colibri, tucan, condor, quetzal, jaguar)
- This is intentional — accented names are OOV tokens that test the encoder/decoder
