# Cutover / Final Acceptance Report — Financial_RAG

**Date:** 2026-10-15
**Scope:** single-node local/dev deployment only
**Status:** **READY FOR FINAL ACCEPTANCE**
**Extends:** `financial_rag_backups/phase1_20261012_212542/phase2d_integration/CUTOVER_REPORT.md`
(the 2026-10-13 named-volume migration cutover, executed and verified) and
`.../phase2d_integration/DEFECT_TRIAGE_AND_REPAIR_PLAN.md` (D1–D7 authority).

This document is the authoritative, repository-local handover/acceptance record.
The backup directory is a preservation snapshot and is not the place of record.

---

## 1. Purpose

The 2026-10-13 cutover converged the running stack onto the final
`docker-compose.yml` on Docker named volumes and verified data integrity. This
report records the subsequent **D1–D7 defect-remediation pass**, the final test
evidence, and the residual limitations, and sets the acceptance status.

---

## 2. Defect remediation — D1–D7 (resolved)

| ID | Defect | Resolution | Evidence |
|----|--------|-----------|----------|
| **D1** | Guardrail verdict computed then discarded; API/audit always reported `passed` | Verdict propagated end-to-end (`src/pipeline.py`); API builds `GuardrailStatus` from the real verdict (`app/api/main.py`); audit stores real status | `test_pipeline_orchestrator.py` (33), `test_guardrail_verdict_propagation.py`, `test_stream_guardrail_enforcement.py` |
| **D1b** | Semantic cache served entries without enforcing `guardrail_passed` | Cache hit ignored unless `guardrail_passed is True` (`src/pipeline.py`) | `test_2_caching`, `test_caching_integration.py` |
| **D2** | `guardrail_pass_rate` divided by ineligible events | Rate computed over eligible events only; `eligible_queries` exposed (`app/api/analytics.py`) | `test_analytics_scores.py` |
| **D3** | Latency measured with non-monotonic wall clock | `time.perf_counter()` for elapsed time; latency outliers dropped and surfaced (`app/api/analytics.py`, `app/api/main.py`, `src/pipeline.py`) | `test_analytics_scores.py`, `test_api_endpoints.py` |
| **D4** | `context_quality` buckets degenerate | Score normalization on both paths; tertile thresholds `0.66`/`0.33`; normalized descriptions (`app/api/analytics.py`, `app/api/schemas.py`) | `test_analytics_scores.py` |
| **D5** | Refusals validated only via the structured-output failure path; malformed output silently default-filled | First-class refusal handling via `model_validator`, separate `RefusalAnswer`, `classify_generation_output()` (`structured`/`refusal`/`malformed`) (`src/5_generation/schemas.py`) | `test_generation_stage.py` |
| **D6** | MongoDB ↔ Qdrant drift with no visibility or safe reconciliation tooling | Parity health field (`/health.mongo_qdrant_parity`), scoped `safe_replacement.py`, `database_indexer.py` `source_id`, identity reconcilers (`src/1_ingestion/reconcile_identity.py`, `reconcile_readers.py`) — code + tests delivered | `test_api_endpoints.py::TestHealthEndpoint`, `test_ingestion_identity`, `test_safe_replacement`, `test_reconcile_identity`, `test_reconcile_readers` |
| **D7** | Demo script correctness (wrong company label, duplicate numbering) | `docs/DEMO_SCRIPT.md` corrected (Apple→Microsoft; Test 3→4, 4→5, 5→6) | docs review |

**D6 note (data operation):** the *repair tooling and parity observability* are
resolved and tested. The live reconciliation **apply** (re-embed 11 Mongo-only
chunks, delete 6 verified Qdrant orphans) is a data-writing operation that cannot
be auto-derived (all corpus identity rows are legacy/unkeyed) and remains
**approval-gated and unapplied** — tracked under carry-over C3 below. `/health`
currently reports `mongo_qdrant_parity = "drift (mongo=1165, qdrant=1160, delta=5)"`.

---

## 3. Carry-over limitations (C1–C7)

| ID | Severity | Limitation | Status |
|----|----------|-----------|--------|
| **C1** | Medium | MLflow registered-model artifacts v1–v7 absent; `model_loader` falls back to direct pipeline init (functionally equivalent) | Open — schedule out of band |
| **C2** | Medium | `rag_data` corpus is NVDA-only; cross-ticker ingestion coverage not claimed | Open |
| **C3** | Low | 1 accepted Qdrant fixture-vector divergence (AAPL fixture point); plus the D6 reconciliation apply (11 Mongo-only / 6 Qdrant-only, net delta 5) | Open — data write, approval-gated |
| **C4** | Low | Pre-existing test drift recorded in Phase 2D (8 failures / 5 errors) | **Resolved** in this pass (see §4) |
| **C5** | Low | Rate-limit test harness required a hard-coded `REDIS_URL` | **Resolved** — harness now derives host/port from `config.settings.REDIS_URL` and skips cleanly when Redis is unreachable |
| **C6** | Low–Med | Backend `8000` and frontend `3000` published on all host interfaces (`/health`, `/docs`, `/openapi.json`, `/api/v1/chat` reachable) | Open — acceptable for local/dev only |
| **C7** | Low | Backend **image** dependency mismatch: `ragas` imports `langchain_community.chat_models.vertexai`, absent from the image's `langchain-community`, so `test_evaluation_module::test_generate_saves_exactly_25_rows` fails in-container (passes on host) | Open — documented limitation; **do not modify Dockerfile/requirements** |

---

## 4. Test evidence

**Authoritative evidence file:** `test_artifacts/TEST_RUN_REPORT.md`
(raw logs: `test_artifacts/pytest_full.log`, `pytest_live2.log`, `pytest_live3.log`).

| Run | Environment | Result |
|-----|-------------|--------|
| Host baseline (pre-deploy) | `Financial_env` | `15 failed, 612 passed, 1 skipped, 35 errors` |
| Live stack (isolated test DB/collection) | one-off container on `financial-rag_backend-net` | `45 failed, 613 passed, 5 errors` |
| **Final (after test-maintenance fixes)** | one-off container on `financial-rag_backend-net` | **`1 failed, 662 passed, 6 warnings` (9:48)** |

The single remaining failure is **C7 (image-only)**. No failure is attributable to
D1–D7, the `test_api_endpoints.py` RBAC fix, or the D6 ingestion changes.

### Test-only maintenance fixes applied (no production code)
- 4 integration files: generator stubs accept `**kwargs`; `_augment_context` lambda accepts `all_companies=False`.
- `test_api_endpoints.py`, `test_api_warmup.py`, `test_api_advanced.py`: additionally patch `app.api.main.get_or_create_pipeline`; assert `tickers=None`.
- `test_pipeline_orchestrator.py`: fixture docs use distinct sections (section-dedup no longer collapses them).
- `test_4_hybrid_search.py`: schema assertion includes `rescued_table`.
- `test_api_worker_arq.py`: Redis DSN derived from config; `test_api_advanced.py` rate-limit fixture skips when Redis is unreachable.

These changed test files only; no `src/` or `app/` production code was modified for them.

---

## 5. Safety / rollback

- No destructive operations were performed (no `down -v`, `volume prune`, no
  collection/DB/volume deletion).
- Backup `phase1_20261012_212542` preserved (`CHECKSUMS.txt` over the data set
  verified): `qdrant_db/storage.sqlite`, `mlflow/mlflow.db`,
  `mongo_dump/*.bson`, `mlflow/mlruns.tar.gz`.
- Legacy rollback volumes retained and untouched.
- Production corpus untouched by the test runs (isolated `financial_rag_test`
  DB and `financial_vectors_test` collection; integration harnesses use temp
  DBs/paths).

---

## 6. Recommendation

**Accept.** D1–D7 repair work is complete and verified; the backend was rebuilt
and redeployed (healthy) with the parity field live on `/health`; test evidence is
persisted in `test_artifacts/TEST_RUN_REPORT.md`; the only residual test failure is
the documented image-only dependency limitation **C7**. Address C1–C3, C6, and C7
separately; none blocks acceptance.
