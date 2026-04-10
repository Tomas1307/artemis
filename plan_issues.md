# ARTEMIS — Plan & Open Issues (Session 10 Handoff)

## What Was Done This Session

### Data Fixes Applied
- **SEC-001/SEC-002 scope**: Skeleton updated from `module_only` → `station_wide`. All 172 gold standard entries were already consistently `station_wide`, skeleton was wrong.
- **SEC-012 severity**: 86 `send_alert` questions had `severity='medium'` (protocol severity) instead of following telemetry band convention. Fixed to use telemetry bands consistently:
  - Radiation 1.1-2.0 mSv/hr → `severity='medium'`
  - Radiation 2.0-5.0 mSv/hr → `severity='high'`
  - Fixed in: gold_standard.json (75), test_gold_standard.json (11), data.csv (75)
- **Train-test leakage**: Found and removed 35 near-duplicate queries between train and test sets. Test set: 354 → 319 rows. Files updated: test_gold_standard.json, test_queries.csv, sample_submission.csv.
- **consultas_centro_control.json**: Rewrote 44 queries for MASA-DOC-007, DOC-011, DOC-036 to be more document-specific (these had generic queries that didn't anchor to their doc's unique content).

### Validators Built (app/validators/)
1. **ThresholdProtocolCorrectnessValidator** — regex-extracts numeric readings from RAG queries, verifies gold protocol_id matches skeleton threshold ranges. Result: 0 errors, 29 warnings (communication uptime queries — no numeric protocol match).
2. **AlertSeverityConsistencyValidator** — checks send_alert severity matches telemetry bands. Result: 0 errors, 4 warnings (boundary values at exactly 14.0%/85.0 kPa).
3. **TestTrainConsistencyValidator** — schema parity, tool coverage, seed_type distribution, near-duplicate leakage detection. Result: 0 errors, 0 warnings.
4. **ConsultasRetrievalValidator** — uses bge-small-en-v1.5 to embed all docs + queries, checks if assigned doc ranks in top-3. Result: 460 rank-1, 169 rank 2-3, 171 rank 4+. See open issue below.

### Files Modified
- `app/skeleton/skeleton.yaml` — SEC-001, SEC-002 scope changed
- `proyecto_artemis/datos_entrenamiento/gold_standard.json` — severity + leakage fixes
- `proyecto_artemis/datos_entrenamiento/data.csv` — severity fixes
- `proyecto_artemis/datos_entrenamiento/consultas_centro_control.json` — 44 queries rewritten
- `proyecto_artemis/evaluacion/test_gold_standard.json` — severity + leakage fixes
- `proyecto_artemis/evaluacion/test_queries.csv` — 35 rows removed (leakage)
- `proyecto_artemis/evaluacion/sample_submission.csv` — 35 rows removed (leakage)
- `anim/enunciado_nuevo.tex` — immersive Artemis II context added

### Current Data Counts
| File | Rows |
|------|------|
| gold_standard.json | 3,186 |
| data.csv | 3,186 |
| test_gold_standard.json | 319 |
| test_queries.csv | 319 |
| sample_submission.csv | 319 |
| consultas_centro_control.json | 800 |
| Documents (base_conocimiento) | 54 |

---

## Open Issues

### 1. consultas_centro_control.json — 171 queries rank below top-3
Using the base encoder (bge-small-en-v1.5 without fine-tuning), 171/800 queries don't retrieve their assigned doc in top-3. Most drops come from noise docs (52), mission records (41), and cross-cutting docs (28). Protocol and system guide docs are clean.

**Decision**: Do NOT judge these with the base encoder. Build the baseline RAG first, fine-tune the encoder, then re-validate. A trained encoder may retrieve these correctly. Only flag pairs that a trained model still fails on.

### 2. Should we add hard negatives to consultas_centro_control.json?
Currently the file has only positive pairs: `(query, doc_id)`. Students construct negatives during training (in-batch negatives with MultipleNegativesRankingLoss). Adding a `hard_negative_doc_id` field would help students, but we should mine these from a TRAINED model, not the base encoder.

**Decision**: Defer until after baseline RAG is built. Use the trained encoder's top-1 wrong retrieval as the hard negative for each query.

### 3. Validators E and F not built yet
- **RAGDocContentGroundingValidator** (LLM-based): For ~200 RAG questions that are NOT activate_protocol or send_alert, verify the cited doc actually justifies the tool_call. Requires NVIDIA API.
- **ConsultasSemanticSpecificityValidator** (LLM-based): For consultas flagged by retrieval validator, use LLM to judge if query is genuinely specific to its assigned doc.

**Decision**: Defer until after baseline. The baseline's retrieval results will tell us which pairs are actually problematic, making LLM validation targeted and cheaper.

### 4. File renames pending
- `data.csv` → `train.csv`
- `test_queries.csv` → `test.csv`
- `enunciado_nuevo.tex` already uses the new names

**Decision**: Do at the very end before delivery to students. Will break validator paths if done now.

### 5. enunciado_nuevo.tex — needs final review
The document is mostly complete. Pending:
- Verify dates (competition open/close) are final
- Add actual Kaggle invitation link (currently TBD)
- Update data counts if they change (currently says 3,186 train, 354 test — test is now 319)

---

## Next Steps: Phase 5 Baseline RAG

### Why build this now
1. Required for Kaggle leaderboard (students must beat it)
2. Validates data quality with a real trained system
3. Generates proper hard negatives for consultas
4. Proves the dataset is solvable

### Pipeline to build

#### Step 1: Encoder Fine-Tuning
- Model: `BAAI/bge-small-en-v1.5`
- Training data: `consultas_centro_control.json` (800 positive pairs)
- Loss: `MultipleNegativesRankingLoss` (in-batch negatives)
- Chunk docs into ~512 token segments
- Train 3-5 epochs, batch_size=16
- Save checkpoint

#### Step 2: Build FAISS Index
- Chunk all 54 docs
- Embed chunks with fine-tuned encoder
- Build FAISS index (IndexFlatIP for cosine similarity)

#### Step 3: Decoder Fine-Tuning
- Model: `meta-llama/Llama-3.2-1B`
- Training data: `data.csv` (3,186 queries with tool_calls)
- For each query: retrieve top-K docs with fine-tuned encoder → format as context
- Train decoder to generate tool_call given (query + context)
- LoRA or full fine-tuning (LoRA preferred for 1B model)
- Save checkpoint

#### Step 4: Inference Pipeline
- For each test query:
  1. Encode query with fine-tuned encoder
  2. Retrieve top-K chunks from FAISS
  3. Format prompt: query + retrieved context
  4. Generate tool_call with fine-tuned decoder
  5. Post-process output (regex normalization to canonical format)
- Save predictions as submission CSV

#### Step 5: Score & Validate
- Run exact match against test_gold_standard.json
- Record baseline score for Kaggle leaderboard
- Analyze errors:
  - Retrieval failures (encoder didn't find right doc)
  - Generation failures (decoder had right context but wrong output)
  - Data issues (gold standard might be wrong)

#### Step 6: Data Validation with Trained RAG
- Re-run ConsultasRetrievalValidator with fine-tuned encoder → should be much better
- Run LLM validators (E, F) on remaining failures
- Mine hard negatives from trained encoder's mistakes
- Optionally add `hard_negative_doc_id` to consultas_centro_control.json
- Fix any data issues found → retrain → final score

### Tech Requirements
- Python 3.11+ with venv (already set up)
- PyTorch with CUDA (RTX 4060 Laptop GPU, torch 2.6.0+cu124 installed)
- sentence-transformers, faiss-cpu, transformers (all installed)
- Llama-3.2-1B weights (download from HuggingFace, need HF token)
- NVIDIA API for LLM validators (configured in .env)

### Estimated Complexity
- Encoder fine-tuning: straightforward, ~30 min GPU time
- FAISS index: trivial, < 5 min
- Decoder fine-tuning: moderate, LoRA on 1B model ~1-2 hours GPU
- Inference: ~10 min for 319 test queries
- Analysis + data fixes: depends on what we find

---

## Summary of Severity Conventions (for reference)

The dataset uses **telemetry alert band severity** for `send_alert`, NOT protocol severity:

| Metric | Low | Medium | High | Critical |
|--------|-----|--------|------|----------|
| Radiation (mSv/hr) | 0.5-1.0 | 1.0-2.0 | 2.0-5.0 | >5.0 |
| Pressure (kPa) | 95-99 | 90-95 | 85-90 | <85 |
| Oxygen (%) | 18-19.5 | 16-18 | 14-16 | <14 |
| Power (% capacity) | 80-88 | 88-93 | 93-97 | >97 |

Boundary values (exactly 14.0%, 85.0 kPa) are treated as the higher band (high, not critical) — 4 such cases exist as acceptable warnings.

## Protocol Scope Reference

| Protocol | Scope | Trigger |
|----------|-------|---------|
| MASA-SEC-001 | station_wide | pressure < 85.0 kPa |
| MASA-SEC-002 | station_wide | oxygen < 14.0% |
| MASA-SEC-003 | module_only | temp rise > 2.0°C/min |
| MASA-SEC-004 | station_wide | radiation > 5.0 mSv/hr |
| MASA-SEC-006 | station_wide | power > 97% capacity |
| MASA-SEC-007 | station_wide | comms lost > 30 min |
| MASA-SEC-008 | module_only | hull stress > 85% |
| MASA-SEC-009 | module_only | CO2 > 1.5% |
| MASA-SEC-010 | module_only | oxygen 14.0-15.9% |
| MASA-SEC-011 | module_only | pressure 85.0-89.9 kPa |
| MASA-SEC-012 | station_wide | radiation 1.1-5.0 mSv/hr |
| MASA-SEC-013 | module_only | docking differential > 5.0 kPa |
| MASA-SEC-014 | station_wide | scrubber < 60% |
| MASA-SEC-015 | station_wide | water recycling < 40% (>2h) |
| MASA-SEC-017 | module_only | airlock rate > 0.5 kPa/min |
| MASA-SEC-018 | module_only | voltage fluctuation > 10% |
