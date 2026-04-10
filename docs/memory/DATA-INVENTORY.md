# Data Inventory — proyecto_artemis

Last updated: 2026-04-10 (Session 11, post-fix)

## File Counts

| File | Count | Description |
|------|-------|-------------|
| data.csv | 3186 rows | Student training data (id, query, tool_call) |
| gold_standard.json | 3186 entries | Internal answer key with metadata (not for students) |
| test_queries.csv | 319 rows | Kaggle evaluation queries (id, query — no answers) |
| test_gold_standard.json | 319 entries | Test answers for grading (not for students) |
| sample_submission.csv | 319 rows | Example submission format (all no_action placeholder) |
| consultas_centro_control.json | 810 pairs | Encoder training hints (query, doc_id, hard_negative_doc_id) |
| tools_definition.json | 10 tools | Tool definitions with enum params |
| documentos_masa.json | 54 docs | Knowledge base index |

### Changes Since Session 9
- data.csv: 3188 → 3186 (removed 2 duplicates in session 10)
- test_queries.csv: 362 → 319 (removed 35 train-test leaks in session 10)
- test_gold_standard.json: 362 → 319 (same)
- sample_submission.csv: 362 → 319 (same)
- consultas_centro_control.json: 800 → 810 (added 10 for MASA-DOC-012), now includes hard_negative_doc_id field (622 with negatives, 128 removed as ambiguous, 60 without)

## Consultas Centro Control — Detailed

### Structure
```json
{
  "query": "...",
  "doc_id": "MASA-DOC-XXX",
  "hard_negative_doc_id": "MASA-DOC-YYY"  // optional, 622 out of 810
}
```

### Distribution
- 54 documents × 15 queries each = 810 total (uniform)
- MASA-DOC-012 brought from 5 → 15 in session 11
- Hard negatives mined from base bge-small-en-v1.5 retrieval audit (top-1 wrong doc)

### Hard Negative Removal Rules (128 removed)
| Reason | Count |
|--------|-------|
| Overlapping system docs (e.g., Jaguar LS Manual vs LS Guide) | 43 |
| Crew profile as negative when query mentions that person | 33 |
| Quick-Reference Card (DOC-038) summarizes all protocols | 18 |
| Daily Operations (DOC-018) references many procedures | 13 |
| Super document (DOC-036) overlap with everything | 11 |
| Both docs are protocol_group (heavy cross-referencing) | 10 |

### Content Verification Results (Session 11 — 9 subagents read all docs)

| Batch | Docs | CORRECT | WRONG | WEAK |
|-------|------|---------|-------|------|
| 1 | Module manuals (001-006) | 85 | 1 | 4 |
| 2 | Protocols (007-012) | 89 | 1 | 0 |
| 3 | System guides (013-018) | 64 | 2 | 23 |
| 4 | Procedures/missions (019-024) | 81 | 2 | 7 |
| 5 | Missions/crew (025-030) | 85 | 1 | 4 |
| 6 | Crew/overview (031-036) | 88 | 0 | 2 |
| 7 | Cross-cutting (037-042) | 80 | 2 | 8 |
| 8 | Noise docs (050-055) | 77 | 3 | 10 |
| 9 | Noise docs (056-061) | 40 | 20 | 30 |
| **TOTAL** | **810** | **689 (85.1%)** | **32 (3.9%)** | **88 (10.9%)** |

### 32 WRONG Consultas by IDX

**DOC-056 (Recreation) — 9 WRONG**: 109, 260, 338, 340, 376, 441, 442, 648, 781
- Root cause: Doc is vague overview with zero operational specifics. Queries ask for procedures, schedules, backup plans.

**DOC-057 (Inter-Agency) — 6 WRONG**: 57, 135, 170, 203, 562, 767
- Root cause: Doc covers strategic-level frameworks only. Queries ask for crew names, altitudes, emergency coordination.

**Scattered (17 WRONG)**: 255 (DOC-004), 666 (DOC-011), 381, 466 (DOC-013), 710 (DOC-021), 444 (DOC-023), 489 (DOC-028), 532 (DOC-041), 582 (DOC-040), 51, 558 (DOC-052), 673 (DOC-055), 742 (DOC-058), 484, 757 (DOC-059), 112 (DOC-060), 622 (DOC-061)
- Root cause: Queries ask for specific values (counts, durations, frequencies) that LLM-generated docs replaced with vague language.

### 88 WEAK Consultas — Main Clusters
- DOC-014 (Power Guide): 8 WEAK — values redacted ("high efficiency", "a voltage")
- DOC-016 (Comms Guide): 8 WEAK — same redacted values problem
- DOC-057 (Inter-Agency): 9 WEAK — strategic doc, operational queries
- DOC-058 (Environmental Impact): 7 WEAK — general language, queries want specifics
- DOC-059 (Photography): 5 WEAK — broad doc, module-specific queries
- DOC-060 (Patches): 5 WEAK — queries want details doc doesn't have

### Fixes Applied
- DOC-056: all 15 queries rewritten to match actual doc content (recreation areas, digital library, cultural celebrations, team-building). 9 WRONG → 0.
- DOC-057: all 15 queries rewritten to match actual doc content (ADCO/MBS/HECP projects, partner agencies, ground stations, personnel exchange). 6 WRONG + 9 WEAK → 0.
- Remaining 17 scattered WRONG: acceptable noise (one-offs, broadly correct doc assignment)
- 58 remaining WEAK (was 88, minus 30 rewritten): acceptable (partial info, encoder learns correct association)

## Document Corpus — Chunk Analysis (Session 11)

### Chunking Stats
| Metric | Value |
|--------|-------|
| Total documents | 54 |
| Total chunks | 387 |
| Subchunks (from oversized sections) | 172 (44%) |
| Chunks per doc | min=4, max=16, mean=7.2 |
| Tokens per chunk | min=36, max=390, mean=258 |
| LLM summaries | Yes (Devstral-2-123B via NVIDIA API) |

### Chunk Schema
Each chunk has: doc_id, chunk_id, topic (H1), subtopic (H2), keypoint (H3), content, summary (LLM-generated), parent_summary (for subchunks), parent_id (UUID linking subchunks), embedding_text (semantic search format), metadata (modules_mentioned, protocols_mentioned, crew_mentioned, thresholds, tools_relevant, document_type)

### Retrieval Audit (base bge-small-en-v1.5, NO fine-tuning)
| Rank | Count | % | Meaning |
|------|-------|---|---------|
| 1 | 445 | 55% | Perfect match |
| 2-3 | 149 | 18% | Close, fine-tuning should fix |
| 4-5 | 57 | 7% | Needs fine-tuning |
| 6-10 | 66 | 8% | Problematic |
| 11+ | 93 | 11.5% | Likely broken or deeply ambiguous |
| Not found | 0 | 0% | All docs represented in top-50 |

### Cross-Document Overlap
- 1314/1431 doc pairs (92%) above 0.75 cosine similarity
- Top "super document": MASA-DOC-036 (avg_sim=0.87 with all others)
- Top similar pairs: DOC-002↔DOC-041 (0.966), DOC-012↔DOC-038 (0.963), DOC-003↔DOC-015 (0.962)

### Document Quality Issues Found
| Doc | Issue | Impact |
|-----|-------|--------|
| DOC-007 | Line 7: "All atmospheric protocols operate at module-only scope" — WRONG for SEC-001, SEC-002 (station_wide in skeleton) | RAG queries about protocol scope get wrong answer |
| DOC-014 | Values redacted: "high efficiency", "a voltage", "a duration" throughout | 8/15 consultas can't be answered precisely |
| DOC-016 | Same redacted values problem as DOC-014 | 8/15 consultas can't be answered precisely |
| DOC-056 | Vague overview, zero operational specifics | 9/15 consultas WRONG |
| DOC-057 | Strategic-level only, no operational details | 6/15 WRONG, 9/15 WEAK |

### Skeleton Fidelity
- All 16 protocol threshold VALUES: exact match in documents
- All scopes correct EXCEPT DOC-007 line 7 (SEC-001/SEC-002 mismatch)
- SEC-012 scope implicitly station_wide but not explicitly labeled

## Tool Distribution

| Tool | data.csv | % | test | % |
|------|----------|---|------|---|
| activate_protocol | 934 | 29.3% | 107 | 33.5% |
| send_alert | 796 | 25.0% | 92 | 28.8% |
| no_action | 394 | 12.4% | 45 | 14.1% |
| control_system | 190 | 6.0% | 19 | 6.0% |
| request_supply | 140 | 4.4% | 17 | 5.3% |
| schedule_maintenance | 137 | 4.3% | 16 | 5.0% |
| calculate_trajectory | 136 | 4.3% | 17 | 5.3% |
| get_telemetry | 125 | 3.9% | — | — |
| get_module_status | 123 | 3.9% | — | — |
| get_crew_status | 108 | 3.4% | — | — |
| send_message | 105 | 3.3% | — | — |

Note: test counts updated after 35 leak removals in session 10.

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

## Known Issues

| Issue | Status | Impact | Session |
|-------|--------|--------|---------|
| ~170 pure duplicate queries in data.csv | Intentional | Students should deduplicate | S8 |
| MASA-SEC-016, 020 unused | Intentional | Realistic noise | S8 |
| MASA-SEC-006, 013, 019 not in test | Noted | No impact on evaluation | S9 |
| ID gaps in data.csv | Cosmetic | From conflict dedup + generation rounds | S8 |
| DOC-007 scope text wrong (line 7) | **NEEDS FIX** | SEC-001/002 shown as module_only, should be station_wide | S11 |
| DOC-014 redacted values | **NEEDS FIX** | 8/15 consultas WEAK — can't answer precisely | S11 |
| DOC-016 redacted values | **NEEDS FIX** | 8/15 consultas WEAK — same issue | S11 |
| DOC-056 thin content | **FIXED (queries rewritten)** | 15 queries now match doc content | S11 |
| DOC-057 thin content | **FIXED (queries rewritten)** | 15 queries now match doc content | S11 |
| IDX 666 wrong doc assignment | **NEEDS FIX** | SEC-017 assigned to DOC-011, should be DOC-010 | S11 |
| 17 scattered WRONG consultas | Acceptable noise | Queries ask for specifics docs left vague | S11 |
| 88 WEAK consultas | Acceptable | Partial info, encoder still learns broadly correct association | S11 |
| 92% pairwise doc similarity | By design | Base encoder struggles — fine-tuning essential | S11 |

## Intentional Student Cleaning Challenges

| Challenge | Description | What Students Must Do |
|-----------|-------------|----------------------|
| ~170 pure duplicate rows | Same query + same tool_call, intentionally left | Deduplicate before training |
| MASA-SEC-016, 020 unused | Defined in tools_definition.json, zero questions | Handle gracefully |
| MASA-SEC-006, 013, 019 train-only | Appear in data.csv but not in test | Don't overfit to protocol distribution |
| Non-sequential IDs | Gaps from conflict dedup + generation rounds | Don't assume contiguous IDs |
| Accent mismatch query↔enum | Queries use Vicuña/Colibrí/Tucán/Cóndor, enums use vicuna/colibri/tucan/condor | Normalize in tokenizer or post-processing |
| Module manuals (DOC-001–006) in train only | Training has questions for these docs, test has ~0 | Don't over-index on module manual retrieval |
| 92% doc pairwise similarity | Base encoder confuses most doc pairs | Must fine-tune encoder to discriminate |
| Hard negatives in consultas | 622/810 have hard_negative_doc_id field | Use for contrastive training (TripletLoss) |

## Accent Convention

- Query text: accented Spanish names (Vicuña, Colibrí, Tucán, Cóndor, Quetzal, Jaguar)
- Tool call enums: lowercase, no accents (vicuna, colibri, tucan, condor, quetzal, jaguar)
- Intentional OOV tokens that test the encoder/decoder

## Artifacts Generated (Session 11)

| File | Location | Description |
|------|----------|-------------|
| chunks.json | artifacts/data_audit/ | 387 chunks with LLM summaries, hierarchy, entity metadata |
| chunk_embeddings.npy | artifacts/data_audit/ | 387×384 float32 embeddings (base bge-small) |
| faiss_index.bin | artifacts/data_audit/ | FAISS IndexFlatIP over chunk embeddings |
| audit_report.json | artifacts/data_audit/ | Full retrieval audit: rank per consulta, confusion clusters, overlap analysis |
