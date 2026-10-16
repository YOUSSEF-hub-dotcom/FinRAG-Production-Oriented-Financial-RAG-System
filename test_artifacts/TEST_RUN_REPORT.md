# Full Test Suite Evidence — D2–D7 Repair Pass

**Date:** 2026-10-15
**Interpreter:** `Financial_env/bin/python` (Python 3.12)
**Command:**
```
PYTHONPATH=app ./Financial_env/bin/python -u -m pytest test/ \
    -p no:cacheprovider -q -rA --durations=25
```
**Raw log:** `test_artifacts/pytest_full.log` (full, ~20k lines)
**Collection log:** `test_artifacts/collect.log` (663 tests collected)

> Note: the suite must be run with `app/` on `PYTHONPATH`. `test/test_cors.py`
> (a committed ad-hoc script with no test functions) does `from api.main import
> app`; without `PYTHONPATH=app` collection aborts with `ModuleNotFoundError: No
> module named 'api'`. This is a pre-existing repo quirk, not a code defect.

## Result

```
15 failed, 612 passed, 1 skipped, 35 errors in 1614.47s (0:26:54)
```

## In-scope areas — PASSING

The tests covering the files changed in this pass all pass:

| Area | Defect | Test file(s) | Result |
|------|--------|--------------|--------|
| `app/api/analytics.py` | D2, D3, D4 | `test_analytics_scores.py` | PASS |
| `src/5_generation/schemas.py` | D5 | `test_generation_stage.py` | PASS |
| `app/api/main.py` `/health` | D6 | `test_api_endpoints.py::TestHealthEndpoint` | PASS |
| `test_api_endpoints.py` upload RBAC | D5/tests | `test_api_endpoints.py::TestDocumentUpload` (5) | PASS |
| D6 ingestion identity/reconcilers | D6 | `test_ingestion_identity`, `test_safe_replacement`, `test_reconcile_identity`, `test_reconcile_readers` | PASS |

The full run's failure list contains **no** test that exercises the D2/D3/D4/D5/D6
edits. The 5 previously-unauthorized upload tests now pass with the added
`admin_client` fixture.

## Failures (15) — categorized

| # | Test | Attribution |
|---|------|-------------|
| 1 | `test_4_hybrid_search.py::TestBalancedRRF::test_output_schema` | Pre-existing test drift: chunk now carries an extra `rescued_table` key (table-shield feature); assertion not updated. Not D2–D7. |
| 2-5 | `test_api_advanced.py::TestRateLimiting::*` (4) | Environmental: re-enables the Redis-backed limiter; no Redis on `127.0.0.1:6379`. |
| 6-7 | `test_api_endpoints.py::TestChatEndpoint::{test_query_with_ticker_filter, test_pipeline_called_with_correct_args}` | Pre-existing isolation defect: test patches `app.api.main.FinancialRAGPipeline`, but the lifespan builds the pipeline via `model_loader.get_or_create_pipeline()` (module-global cache, present at HEAD), so the mock never applies and the real pipeline runs. Reproduced in isolation. |
| 8 | `test_api_endpoints.py::TestStreamingEndpoint::test_stream_yields_tokens` | Same root cause as #6–7. |
| 9-12 | `test_api_warmup.py::TestWarmUp::{embedding,qdrant,mongo,redis}` (4) | Same root cause: `fresh_app_instance` patches `FinancialRAGPipeline`, but warm-up runs on the real pipeline from `get_or_create_pipeline()`. |
| 13 | `test_api_worker_arq.py::TestArqWorkerSettings::test_worker_settings_redis_dsn_uses_localhost` | Environmental/config: `WorkerSettings.redis_settings.host == "127.0.0.1"` vs expected `"localhost"`. |
| 14 | `test_pipeline_orchestrator.py::TestCacheMiss::test_generator_called_with_retrieved_docs` | Pre-existing: `_augment_context` reduces 3 retrieved docs to 1 (function unchanged by this pass). |
| 15 | `test_pipeline_orchestrator.py::TestMLflowLogging::test_mlflow_metrics_logged` | Pre-existing (D1 guardrail-restructuring region; not a D2–D7 edit). |

## Errors (35) — environmental

All 35 are **setup errors** in live-integration test modules whose fixtures
construct `MongoDBIndexer` and require a running MongoDB at
`mongodb://127.0.0.1:27017` (host loopback), which is not available in this
environment (services run in Docker with no host port mapping):

- `test_3_pre_retrieval_integration.py` (8)
- `test_4_post_retrieval_integration.py` (12)
- `test_4_retrieval_integration.py` (9)
- `test_caching_integration.py` (6)

Representative cause: `pymongo ... serverSelectionTimeoutMS=5000` →
`127.0.0.1:27017: [Errno 111] Connect call failed`.

## Conclusion

No failure or error is attributable to the D2–D7 edits or the
`test_api_endpoints.py` fix. Confirmed by:

1. Re-running the two suspicious files **in isolation** (`test_api_endpoints.py`,
   `test_pipeline_orchestrator.py`) reproduces the identical failures, proving
   they are order-independent and pre-existing.
2. `_augment_context` does not appear in the working-tree diff; the large
   `pipeline.py` hunk is the **D1** guardrail-verdict work (comments marked
   `D1:`), not the D3 timing edits.
3. The `get_or_create_pipeline` vs `FinancialRAGPipeline` patch mismatch exists
   in committed `HEAD` (`app/api/main.py:87`).
4. All 35 errors are MongoDB-unavailable fixture setup errors.

### Residual (pre-existing, out of scope for D2–D7)

- `_cached_pipeline` module global in `src/model_loader.py` breaks
  `FinancialRAGPipeline`-patching in `test_api_endpoints.py` /
  `test_api_warmup.py` (needs `get_or_create_pipeline`/`invalidate_cache`
  patched, or fixture reset).
- `test_4_hybrid_search` schema assertion missing `rescued_table`.
- `test_api_worker_arq` DSN host expectation.
- Integration tests require host-reachable MongoDB/Qdrant/Redis.

---

# Live-Stack Run (deployed containers, isolated test DB/collection)

**Date:** 2026-10-15
**Host reachability:** Docker Desktop runs in a separate VM; the WSL host cannot
reach container bridge IPs, so the suite was executed inside a one-off container
on `financial-rag_backend-net` (image `financial-rag-backend:latest`, pytest 9.1.1).
**Command (abridged):**
```
docker run --rm --gpus all --network financial-rag_backend-net --user 0:0 \
  --env-file .env \
  -e MONGODB_URI=mongodb://mongo:27017/financial_rag -e MONGODB_DB=financial_rag_test \
  -e REDIS_URL=redis://redis:6379/0 -e QDRANT_HOST=qdrant -e QDRANT_PORT=6333 \
  -e QDRANT_PATH= -e QDRANT_COLLECTION=financial_vectors_test \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 -e DATA_DIR=/tmp/rag_data \
  -v /home/youssef/Financial_RAG:/app:ro -v $PWD/test_artifacts:/evidence \
  -v hf_models:/opt/hf -v nltk_data:/opt/nltk_data --tmpfs /app/logs \
  financial-rag-backend:latest \
  bash -lc 'mkdir -p /tmp/rag_data && (cd /app/data && tar cf - \
      --exclude=qdrant_db --exclude=qdrant_db_test . | (cd /tmp/rag_data && tar xf -)) \
    && cd /app && PYTHONPATH=/app:/app/app python -m pytest test \
      -p no:cacheprovider -q -rN | tee /evidence/pytest_live2.log'
```
**Isolation:** `MONGODB_DB=financial_rag_test`, `QDRANT_COLLECTION=financial_vectors_test`,
and the integration harnesses' own temp DBs/collections → production corpus
(`raw_chunks`, `financial_vectors`) untouched.
**Raw log:** `test_artifacts/pytest_live2.log`

## Result
```
45 failed, 613 passed, 6 warnings, 5 errors in 1165.58s (0:19:25)
```

The earlier in-container attempt (`pytest_live.log`) had 20 extra **setup errors**
caused solely by `DATA_DIR=/tmp/rag_data` pointing at an empty directory; copying
the repo corpus to `/tmp/rag_data` before the run eliminated them (25 → 5 errors).

## Failures (45) + errors (5) — categorized

| Count | Signature | Attribution |
|-------|-----------|-------------|
| 33 | `stub_generate()/lambda() got unexpected kwarg 'multi_ticker'/'all_companies'` | **Pre-existing** integration-test stub drift: `src/pipeline.py` calls `_generator.generate(..., multi_ticker=...)` and `_augment_context(..., all_companies=...)` **at HEAD**; the test stubs (`test_3_/test_4_*_integration.py`, `test_caching_integration.py`) were never updated. Not D2–D7. |
| 7 | api patch isolation (`embed_single`, `query`, `count_points`, `count_documents`, `_get_client` not called / extras) | **Pre-existing** (`get_or_create_pipeline` vs `FinancialRAGPipeline`). |
| 2 | `assert 1 == 3` and `log_metric('documents_retrieved',3)` not found | **Pre-existing** `_augment_context` doc-collapse (D1-era behaviour); orchestrator tests not updated. |
| 1 | `rescued_table` extra key in `test_4_hybrid_search::TestBalancedRRF::test_output_schema` | **Pre-existing** hybrid schema drift. |
| 1 | `assert 'redis' == 'localhost'` (`test_api_worker_arq`) | **Environmental**: `REDIS_URL` host is the service name. |
| 1 | ragas `ModuleNotFoundError: langchain_community.chat_models.vertexai` (`test_evaluation_module`) | **Container-image** dependency mismatch (host env is compatible). |
| 5 | Redis `localhost:6379` Connection refused (setup errors, `TestRateLimiting`) | **Environmental**: `test_api_advanced.py` fixture hardcodes host loopback. |

## Conclusion (live stack)

Every failure/error is pre-existing, test-fixture drift, environmental, or
image-dependency — **none is attributable to the D2–D7 edits or the
`test_api_endpoints.py` fix**. All D2–D7-covered tests pass:
`test_analytics_scores.py`, `test_generation_stage.py`,
`test_api_endpoints.py::TestHealthEndpoint`/`TestDocumentUpload`,
`test_ingestion_identity`, `test_safe_replacement`, `test_reconcile_*`.

### Recommended (separate, out-of-scope) follow-ups
- Update integration-test generator stubs to accept `**kwargs` (or the explicit
  `multi_ticker=`/`all_companies=` params) — 33 failures.
- Fix `get_or_create_pipeline` patching in `test_api_endpoints.py`/`test_api_warmup.py` — 7 failures.
- Update `test_pipeline_orchestrator.py` expectations for the D1 `_augment_context` collapse — 2 failures.
- Add `rescued_table` to `test_4_hybrid_search` schema assertion — 1 failure.
- Point `test_api_worker_arq` / `test_api_advanced` fixtures at a configurable Redis host — 6 failures/errors.
- Pin a compatible `ragas`/`langchain-community` in the backend image (or skip in-image) — 1 failure.

---

# Final Validation Run (after test-maintenance fixes)

**Date:** 2026-10-15
**Command:** identical to the Live-Stack Run above, log → `test_artifacts/pytest_live3.log`.

## Result
```
1 failed, 662 passed, 6 warnings in 588.83s (0:09:48)
```

Relative to the Live-Stack Run (`45 failed, 613 passed, 5 errors`): **+49 passing,
44 failures and all 5 setup errors resolved.** The only remaining failure is the
container-image `ragas`/`langchain-community` mismatch below.

## Test-only maintenance fixes applied (separate from D2–D7 code)

| File(s) | Fix | Cleared |
|---------|-----|---------|
| `test_3_pre_retrieval_integration.py`, `test_4_post_retrieval_integration.py`, `test_4_retrieval_integration.py`, `test_caching_integration.py` | generator stubs now accept `**kwargs`; the `_augment_context` lambda accepts `all_companies=False` | 33 |
| `test_api_endpoints.py`, `test_api_warmup.py`, `test_api_advanced.py` | additionally patch `app.api.main.get_or_create_pipeline` (the call the lifespan actually uses); `assert_called_once_with` now includes `tickers=None` | 8 |
| `test_pipeline_orchestrator.py` | fixture docs use distinct sections so `_augment_context` section-dedup no longer collapses them | 2 |
| `test_4_hybrid_search.py` | schema assertion includes `rescued_table` | 1 |
| `test_api_worker_arq.py`, `test_api_advanced.py` | Redis host/port derived from `config.settings.REDIS_URL`; rate-limit fixture skips (instead of erroring) when Redis is unreachable | 6 |

These are **test-only** changes; no `src/` or `app/` production code was touched
for them. All changes source from pre-existing drift that was present at `HEAD`
and masked because the integration fixtures previously errored out early.

## Remaining failure (1) — container image dependency

- `test/test_evaluation_module.py::TestSyntheticDataGenerator::test_generate_saves_exactly_25_rows`
  → `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`
  raised from `ragas.llms.base` during `evaluation/synthetic_generator.py:generate`.

**Attribution:** the built backend image's `langchain-community` no longer ships
`chat_models.vertexai`, which the installed `ragas` imports at module load. This
**passes on the host** (`Financial_env`), so it is an image/version-pin mismatch,
**not** a D2–D7 code defect. Resolving it requires pinning a compatible
`ragas`/`langchain-community` in the image (Dockerfile/requirements — out of
scope for this pass).

## Final conclusion

With the test-maintenance fixes applied, the suite is **662 passed / 1 failed
(image dependency only)**. No failure is attributable to the D2–D7 edits,
the `test_api_endpoints.py` RBAC fix, or the D6 ingestion changes.
