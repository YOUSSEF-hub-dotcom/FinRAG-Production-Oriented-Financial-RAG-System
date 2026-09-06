# Financial_RAG — Full Forensic End-to-End System Audit
## Single-Ticker + Cross-Ticker / ALL + Memory + Grounding + Full Integration Scan

---

# ROLE

Act as a **Senior RAG QA Engineer, AI Systems Auditor, Retrieval/Grounding Specialist, Backend Integration Engineer, Frontend E2E Tester, and Security-Conscious QA Auditor**.

You are auditing the existing **Financial_RAG enterprise system**.

Your objective is NOT merely to confirm that unit/integration tests pass.

Your objective is to determine whether the **real running system behaves correctly for a normal user**, whether its answers are genuinely grounded in the actual financial source data, and whether every major architectural layer is correctly connected at runtime.

Treat `PROJECT_MAP.md` as an architectural reference and a list of claims to VERIFY — NOT as proof that those claims are currently true.

The audit must be evidence-driven.

---

# 0. IMPORTANT SCOPE DECISION

## CURRENT SYSTEM CAPABILITIES TO AUDIT

The current production architecture is intentionally focused on:

1. **Single-Ticker / Single-Entity RAG**
2. **Multi-Ticker / Cross-Entity RAG**
3. **ALL / Cross-Entity frontend scope**
4. **Conversation memory and coreference across turns**
5. **Grounded financial question answering**
6. **Frontend ↔ API ↔ RAG pipeline integration**

## EXPLICITLY OUT OF SCOPE

DO NOT implement, enable, or redesign the future **multi-hop / agentic / multi-query decomposition architecture** during this audit.

Do NOT turn the current system into an Agent.

Do NOT implement:

- MultiHopPlanner
- LangGraph multi-hop execution
- query decomposition into multiple dependent hops
- agentic tool loops
- autonomous planning
- future `MULTI_HOP_ROADMAP.md` functionality

The PROJECT_MAP explicitly describes the current engine as a **single-pass RAG pipeline**, while the agentic/multi-hop extension is a future roadmap item.

The goal of this audit is to verify that the **CURRENT single-pass system is correct and stable before any future Agentic layer is implemented**.

However, you MAY test complex questions that are already supported by the current single-pass architecture, provided they do not require implementing future multi-hop behavior.

---

# 1. PRIMARY MISSION

Perform a:

> **FULL SYSTEM DISCOVERY + EXHAUSTIVE END-TO-END AUDIT + DATA GROUNDING AUDIT + FAILURE LOCALIZATION**

Do NOT perform only a handful of manual questions.

The audit must cover the complete runtime path:

```text
Repository
    ↓
Project Architecture
    ↓
Frontend
    ↓
Authentication
    ↓
Session Handling
    ↓
Company/Ticker Selection
    ↓
API
    ↓
Semantic Cache
    ↓
Pre-Retrieval
    ↓
Entity Detection
    ↓
Coreference
    ↓
Metadata Filtering
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
Table Augmentation
    ↓
Context Assembly
    ↓
Generation
    ↓
Guardrail
    ↓
Sources/Citations
    ↓
Response
    ↓
Audit Logging
    ↓
MongoDB / Qdrant / Redis
```

For every major failure, determine the **FIRST stage where expected behavior diverges from reality**.

Do not stop at "the answer is wrong."

---

# 2. GOLDEN RULES

## 2.1 Audit first — repair later

During this audit:

```text
OBSERVE
→ REPRODUCE
→ TRACE
→ DIAGNOSE
→ REPORT
```

Do NOT automatically modify production/source code.

Do NOT fix discovered problems while performing the audit.

The purpose of this run is to produce a reliable diagnosis that can later be used for a controlled repair phase.

---

## 2.2 Never trust an answer just because it sounds correct

A plausible financial answer is NOT evidence of correctness.

For factual/numerical answers, verify against:

1. Original SEC/source filing data
2. MongoDB raw chunks + metadata
3. Qdrant retrieval/chunk metadata
4. Runtime retrieved context
5. Generated answer
6. Citations
7. Guardrail result

---

## 2.3 Never use the LLM's answer as ground truth

Ground truth must come from the actual underlying data.

The LLM output is the object being audited.

---

## 2.4 Do not stop after the first bug

If you discover a failure:

1. Record it.
2. Reproduce it.
3. Continue testing.
4. Test related variations.
5. Determine whether it is isolated or systemic.
6. Determine the earliest broken stage.
7. Continue the rest of the audit.

---

# 3. ENVIRONMENT DISCOVERY

Before functional testing:

1. Read the complete `PROJECT_MAP.md`.
2. Inspect the repository tree.
3. Locate the actual frontend.
4. Locate the backend.
5. Locate `demo_script.md`.
6. Locate tests.
7. Locate evaluation artifacts.
8. Locate source financial data.
9. Locate MongoDB configuration.
10. Locate Qdrant configuration.
11. Locate Redis configuration.
12. Locate MLflow configuration.
13. Identify the actual production model loaded by the running server.
14. Identify the live Uvicorn process and port.
15. Identify the frontend URL.
16. Verify that the services are actually healthy.

Do not assume paths or service states from documentation.

Verify them.

---

# 4. LIVE SERVER SAFETY

Keep the live Uvicorn server running throughout the audit.

Do NOT:

- kill Uvicorn
- restart the backend unnecessarily
- clear MongoDB
- clear Qdrant
- delete source data
- re-ingest production data
- modify MLflow Production alias
- modify model registry
- flush Redis globally unless required for a clearly documented controlled cache experiment
- modify frontend source code
- modify backend source code

Especially NEVER call:

```text
DELETE /api/v1/db/clear
```

This endpoint is destructive and must not be used during this audit.

If a restart is absolutely unavoidable, document why, preserve all audit evidence, and verify service health afterward.

---

# 5. AUTHENTICATION

Use the admin credentials supplied separately in the secure execution context.

DO NOT place the actual password inside:

- source files
- this prompt
- generated reports
- logs
- screenshots
- Git
- test artifacts

Authenticate and verify:

```text
POST /api/v1/auth/login
GET  /api/v1/auth/me
```

Verify:

- login success
- JWT acquisition
- authenticated session
- admin role
- token refresh behavior if applicable
- frontend session restoration
- logout behavior if tested
- unauthorized/forbidden behavior where safe to test

Do not expose credentials in the final report.

---

# 6. TEST THE SYSTEM LIKE A REAL USER

The frontend is a first-class part of this audit.

Do not perform only direct API calls.

Use the actual frontend and behave like a normal authenticated user.

You must:

1. Log in.
2. Open the dashboard.
3. Inspect the company/ticker selector.
4. Inspect the recommendation/default questions.
5. Select each supported ticker scope.
6. Ask the recommended questions.
7. Observe the actual UI answer.
8. Inspect displayed sources/citations.
9. Perform follow-up questions where applicable.
10. Test memory in the same session.
11. Test switching ticker scopes.
12. Compare frontend behavior with direct API behavior.

The UI result is part of the system under test.

---

# 7. SUPPORTED TICKER SCOPES

Audit these scopes explicitly:

```text
AAPL
MSFT
NVDA
ALL
```

Where:

```text
AAPL = Apple
MSFT = Microsoft
NVDA = NVIDIA
ALL  = Cross-Entity
```

Do not confuse:

- selecting `ALL`
- explicitly mentioning multiple companies in a query

Both must be tested.

---

# 8. SINGLE-TICKER AUDIT

Perform a dedicated audit for each:

```text
AAPL
MSFT
NVDA
```

For every ticker test:

### A. Direct factual questions

Examples of categories:

- revenue
- net income
- operating margin
- R&D
- cash flow
- segment information
- fiscal year values

### B. Numerical questions

Verify exact:

```text
value
unit
currency
metric
ticker
fiscal year
period
```

### C. Fiscal-year questions

Test supported fiscal years for each company.

Do not assume that all companies have identical fiscal-year coverage.

### D. Table-dependent questions

Test questions whose answers require financial tables.

### E. Narrative/document questions

Test questions whose evidence is primarily in filing text.

### F. Follow-up questions

Test whether the system maintains correct conversational context.

For each request trace:

```text
Question
→ ticker
→ fiscal year
→ intent
→ rewritten/standalone query
→ retrieval
→ reranking
→ augmentation
→ context
→ generation
→ guardrail
→ final answer
→ sources
```

---

# 9. MULTI-TICKER / CROSS-ENTITY AUDIT

This is a PRIMARY capability.

Test:

```text
AAPL + MSFT
AAPL + NVDA
MSFT + NVDA
AAPL + MSFT + NVDA
ALL
```

For every combination test equivalent semantic questions.

Verify:

1. Correct entity detection.
2. Correct ticker propagation.
3. Correct metadata filtering/bypass.
4. Balanced retrieval.
5. No dominant-ticker collapse.
6. Correct per-company evidence.
7. Correct fiscal-year mapping.
8. Correct answer attribution.
9. Correct citations.
10. Correct numerical grounding.

For each multi-ticker query, inspect how many effective chunks/evidence items each company contributed.

If a question requires three companies but the final evidence/context effectively contains only one or two companies, classify the issue as a retrieval/context failure even if the final answer appears plausible.

---

# 10. ALL / CROSS-ENTITY FRONTEND TESTING

In the real frontend:

1. Select `ALL - Cross-Entity`.
2. Verify the request actually sends `ticker="ALL"`.
3. Ask every recommendation question available under the ALL scope.
4. Record every answer.
5. Verify sources for all requested companies.
6. Compare the result against direct API behavior.
7. Repeat selected questions after switching from AAPL/MSFT/NVDA to ALL.
8. Repeat selected questions after switching from ALL back to an individual ticker.

Look specifically for stale ticker state, frontend state leakage, cached responses from another ticker, or incorrect request payloads.

---

# 11. RECOMMENDATION QUESTIONS — COMPLETE MATRIX

The frontend recommendation questions are a required test source.

For each scope:

```text
ALL
AAPL
MSFT
NVDA
```

extract/record every recommendation question shown by the UI.

Do not invent replacements when actual recommendation questions are available.

Execute EVERY available recommendation question.

Build a matrix:

```text
Ticker Scope
Question
Frontend Answer
API Answer
Sources
Detected Tickers
Detected Fiscal Year
Cache Hit/Miss
Model Used
Guardrail Status
Ground Truth
Grounding Verdict
```

No recommendation question should be silently skipped.

---

# 12. DEMO_SCRIPT.MD AUDIT

Locate `demo_script.md` in the project.

Read and parse every question contained in it.

Execute every relevant question against the running system.

For each question compare:

```text
A = Demo Script → System Answer
B = Recommended Question → System Answer
C = Raw Ground Truth
```

Perform:

```text
A ↔ B
A ↔ C
B ↔ C
```

Important:

A and B do NOT need identical wording.

Compare semantic/factual correctness, grounding, entity mapping, fiscal-year mapping, citations, and evidence.

If the same underlying question behaves differently depending on where it came from, investigate why.

---

# 13. GROUNDING AUDIT

For EVERY important factual/numerical answer:

1. Capture the exact answer.
2. Capture all sources/citations.
3. Identify factual claims.
4. Identify numerical claims.
5. Search MongoDB for supporting raw data.
6. Verify corresponding Qdrant chunks/metadata.
7. Verify original filing data when necessary.
8. Determine whether the claim is supported.

Classify each result as:

```text
FULLY GROUNDED
PARTIALLY GROUNDED
UNGROUNDED
INCORRECT
HALLUCINATED
CORRECT REFUSAL
INCORRECT REFUSAL
RETRIEVAL FAILURE
CONTEXT ASSEMBLY FAILURE
GENERATION FAILURE
CITATION FAILURE
GUARDRAIL FAILURE
```

Never mark an answer correct merely because it "looks reasonable."

---

# 14. GROUND-TRUTH TRIANGULATION

Use these evidence layers:

## Layer 1 — Original filing data

Inspect the actual files under:

```text
data/AAPL/
data/MSFT/
data/NVDA/
```

## Layer 2 — MongoDB

Verify:

```text
ticker
fiscal_year
document type
section
chunk_id
raw text
table content
financial values
```

## Layer 3 — Qdrant

Verify:

```text
chunk_id
ticker
fiscal_year
metadata
retrieval eligibility
retrieval score
```

## Layer 4 — Runtime

Capture:

```text
question
ticker
detected entities
fiscal year
rewritten query
retrieved chunks
reranked chunks
augmented context
answer
sources
guardrail
cache
model
latency
request/session ID
```

---

# 15. RETRIEVAL TRACE / FIRST FAILURE LOCALIZATION

For suspicious answers trace the complete chain:

```text
Original Question
        ↓
Intent
        ↓
Detected Tickers
        ↓
Detected Fiscal Year
        ↓
Coreference Resolution
        ↓
Standalone Query
        ↓
Query Expansion (if current system decides it is needed)
        ↓
Metadata Filter
        ↓
Dense Retrieval
        ↓
BM25 Retrieval
        ↓
RRF
        ↓
Reranking
        ↓
Table Shield
        ↓
Cylinder Reordering
        ↓
MongoDB Table Augmentation
        ↓
Final Context
        ↓
Generation
        ↓
Guardrail
        ↓
Final Response
```

Determine the FIRST point where the expected evidence disappears or becomes incorrect.

This is more important than merely labeling the final answer as wrong.

Examples:

- Mongo contains the value, but retrieval never returns it → RETRIEVAL FAILURE
- Retrieval returns it, but augmentation/context removes it → CONTEXT ASSEMBLY FAILURE
- Context contains correct evidence, but LLM answers incorrectly → GENERATION FAILURE
- Answer is correct but source citation is wrong → CITATION FAILURE
- Correct answer is rejected incorrectly → GUARDRAIL FAILURE
- Correct answer is returned from wrong ticker cache → CACHE/SCOPE FAILURE
- Correct behavior at API but wrong behavior in UI → FRONTEND INTEGRATION FAILURE

---

# 16. ENTITY DETECTION

Test explicit entity forms:

```text
Apple
AAPL
Microsoft
MSFT
NVIDIA
NVDA
```

Test combinations.

Test case sensitivity and natural variations where safe.

Do NOT assume the current regex/lexicon implementation is sufficient.

The purpose is to verify the CURRENT supported behavior.

Do not implement semantic ML/LLM entity resolution as part of this audit.

If novel references such as:

```text
the three tech companies
the chipmaker
the first company
the second company
```

are tested, classify them according to the current supported memory/coreference behavior and document limitations rather than implementing new capabilities.

---

# 17. ENTITY ORDER PERMUTATION

For the SAME semantic multi-company question, change entity order.

Example:

```text
Compare Apple, Microsoft, and NVIDIA revenue for FY2025.
```

Then:

```text
Compare NVIDIA, Microsoft, and Apple revenue for FY2025.
```

Then:

```text
Compare Microsoft, Apple, and NVIDIA revenue for FY2025.
```

Then ticker form:

```text
Compare AAPL, MSFT, and NVDA revenue for FY2025.
```

Verify:

- detected entity list
- entity order
- retrieval distribution
- citations
- final answer mapping
- numerical correctness

The textual ordering of the final response may differ.

The factual mapping MUST NOT become incorrect.

---

# 18. MEMORY / MULTI-TURN AUDIT

The current system supports session-based memory.

Test real multi-turn conversations.

## Test A — Basic cross-entity memory

Turn 1:

```text
Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025.
```

Turn 2:

```text
How much did the first company spend on Research and Development in that same fiscal year?
```

Expected conceptual resolution:

```text
first company → Apple/AAPL
same fiscal year → FY2025
```

Then test:

```text
How much did the second company spend on Research and Development?
```

and:

```text
What about the third company?
```

Verify all mappings against ground truth.

---

# 19. MEMORY ENTITY-ORDER TEST

Create separate sessions with different initial company order.

## Session A

```text
Apple, Microsoft, NVIDIA
```

Then:

```text
What did the first company report?
```

Expected mapping:

```text
AAPL
```

## Session B

```text
NVIDIA, Microsoft, Apple
```

Then:

```text
What did the first company report?
```

Expected mapping:

```text
NVDA
```

## Session C

```text
Microsoft, Apple, NVIDIA
```

Then:

```text
What did the first company report?
```

Expected mapping:

```text
MSFT
```

The memory system must preserve the entity order established by the conversation.

---

# 20. MEMORY SESSION ISOLATION

Create at least:

```text
session-A
session-B
```

In session A establish context about Apple.

Then ask an ambiguous follow-up in session B.

Verify session B does NOT inherit session A's context.

Reverse the test.

Look for:

```text
cross-session leakage
stale memory
shared memory contamination
session_id collision
incorrect restoration
incorrect persistence
unexpected reset
```

---

# 21. COREFERENCE TESTING

Test supported follow-up references such as:

```text
the first company
the second company
the third company
that company
the same company
the previous company
the other company
that fiscal year
the same fiscal year
its revenue
their revenue
the company mentioned first
the company mentioned last
```

Where a reference is outside the current documented capability, record it as a limitation rather than implementing new functionality.

For supported memory/coreference behavior, verify the resolved meaning against the previous turns.

---

# 22. TEMPORAL MEMORY

Test:

```text
FY2024
FY2025
FY2026
```

only where those years actually exist for the target company.

Example:

```text
Compare Apple's FY2024 and FY2025 revenue.
```

Follow-up:

```text
How much was R&D in the later year?
```

Verify:

```text
later year → FY2025
```

Also test:

```text
that same fiscal year
the previous fiscal year
the earlier year
the latest year
```

Do not fabricate missing years.

---

# 23. NEGATIVE / REFUSAL TESTING

Test unsupported companies and unsupported data.

Examples:

```text
AMZN
META
TSLA
```

and unsupported fiscal years.

Verify that the system does NOT fabricate values.

A correct refusal is a PASS.

A confident answer based on unsupported model knowledge is a FAILURE.

---

# 24. CACHE AUDIT

The current architecture contains semantic caching.

For selected questions perform:

```text
First execution
Second identical execution
Semantically equivalent execution
```

Record:

```text
cache_hit
model_used
latency
answer
sources
guardrail status
```

Verify cache scope across:

```text
AAPL
MSFT
NVDA
ALL
```

and across fiscal years.

A cached AAPL result must never appear for an MSFT query.

A cached single-ticker result must not contaminate ALL.

If cache behavior is suspicious, inspect cache keys and matching metadata without destructive global flushing.

---

# 25. FRONTEND ↔ API PARITY

For selected questions execute through:

## Path 1

Real frontend UI.

## Path 2

```text
POST /api/v1/chat
```

## Path 3

```text
POST /api/v1/chat/stream
```

Compare:

```text
answer
sources
ticker
session
memory
guardrail
model
cache
latency
```

Identify any divergence between:

```text
frontend
sync API
SSE API
```

---

# 26. STREAMING PARITY

Verify that `/chat/stream` preserves the same core semantics as `/chat`.

Check:

- final answer
- sources
- source text
- ticker
- fiscal year
- guardrail
- model
- session memory
- audit log
- context augmentation

Streaming may differ in transport mechanics.

It must not differ in factual behavior without a documented reason.

---

# 27. AUDIT LOGGING

Verify that real user requests produce appropriate audit records.

Inspect:

```text
rag_audit_logs
```

where safe.

Verify that logs contain useful trace information without exposing secrets.

Check consistency between:

```text
request
session
ticker
retrieved chunks
model
guardrail
cache
latency
answer
```

Do not expose passwords, tokens, or secrets in the audit report.

---

# 28. MONGODB ↔ QDRANT CONSISTENCY

For sampled and problematic chunks verify:

```text
Mongo chunk_id
Qdrant chunk_id
ticker
fiscal_year
document metadata
raw text availability
```

Pay special attention to any documented count discrepancy.

Do NOT automatically classify a count mismatch as a bug.

Determine whether the discrepancy actually affects:

- retrieval
- grounding
- citations
- answers

---

# 29. CODEBASE DEEP SCAN

Perform a full code review of relevant integration points.

At minimum inspect:

```text
app/api/
src/model_loader.py
src/1_ingestion/
src/2_caching/
src/3_pre_retrieval/
src/4_retrieval/
src/5_generation/
src/pipeline.py
frontend/src/
evaluation/
test/
scripts/probes/
```

Focus on integration boundaries.

Look for:

```text
metadata filter loss
ticker propagation errors
fiscal year propagation errors
session_id loss
memory reset
cross-session leakage
query rewrite errors
entity detection failures
retrieval imbalance
Mongo/Qdrant mismatch
context truncation
table augmentation errors
citation mismatch
stream/sync divergence
cache key collisions
guardrail bypass
fallback model inconsistencies
frontend/backend payload mismatch
stale frontend state
```

---

# 30. VERIFY PROJECT_MAP CLAIMS

The project documentation claims:

- modules are complete
- full regression passes
- cross-entity retrieval is complete
- memory/coreference is complete
- Version 7 is Production
- grounding/evaluation metrics pass
- specific retrieval and augmentation behavior exists

Verify these claims against runtime behavior and source code.

If documentation says:

```text
COMPLETE / VERIFIED
```

but runtime evidence shows a failure, report:

```text
DOCUMENTATION CLAIM ≠ RUNTIME BEHAVIOR
```

This is a high-priority finding.

---

# 31. EXISTING TEST SUITE

Inspect the existing tests and, where safe, run the relevant non-destructive regression tests.

Do not assume:

```text
447/447 passing
```

means the live system is correct.

Distinguish:

```text
UNIT PASS
INTEGRATION PASS
LIVE E2E PASS
GROUNDING PASS
```

A unit test can pass while a live frontend/API/data interaction is still broken.

---

# 32. EXISTING EVALUATION ARTIFACTS

Inspect:

```text
artifacts/test_dataset.csv
artifacts/evaluation_results.csv
artifacts/evaluation_scores.json
artifacts/EVALUATION_SUMMARY.md
```

Compare the previous evaluation results with live audit observations.

Determine whether the current behavior is consistent with the reported Production quality.

Do not blindly rerun the complete evaluation if it would interfere with the live system.

---

# 33. FAILURE CLASSIFICATION

Every discovered failure must receive:

```text
ID
Severity
Scope
Ticker(s)
Question
Session
Expected Behavior
Actual Behavior
Ground Truth
First Failure Stage
Root Cause
Evidence
Reproducibility
Recommended Fix
```

Severity:

```text
CRITICAL
HIGH
MEDIUM
LOW
INFO
```

---

# 34. ROOT-CAUSE STANDARD

Never report only:

```text
RAG failed.
```

Instead identify the earliest broken component.

Example:

```text
MongoDB contains the correct value
↓
Qdrant contains the required chunk
↓
retrieval fails to return it
↓
ROOT CAUSE = retrieval/filtering
```

Or:

```text
retrieval contains the correct chunk
↓
augmentation removes/omits it
↓
ROOT CAUSE = context assembly
```

Or:

```text
context contains the correct evidence
↓
LLM generates incorrect value
↓
ROOT CAUSE = generation
```

Or:

```text
answer is correct
↓
citation points to unrelated evidence
↓
ROOT CAUSE = source attribution
```

---

# 35. COVERAGE MATRIX

Produce an explicit coverage matrix.

Example:

| Test Area | AAPL | MSFT | NVDA | AAPL+MSFT | AAPL+NVDA | MSFT+NVDA | AAPL+MSFT+NVDA | ALL |
|---|---|---|---|---|---|---|---|---|
| Direct factual | | | | | | | | |
| Numerical grounding | | | | | | | | |
| Recommendation | | | | | | | | |
| Demo script | | | | | | | |
| Memory | | | | | | | | |
| Coreference | | | | | | | | |
| Entity order | | | | | | | | |
| Temporal memory | | | | | | | | |
| Cache isolation | | | | | | | |
| API parity | | | | | | | |
| Streaming parity | | | | | | | |
| Grounding | | | | | | | |

Do not claim complete audit coverage unless this matrix is populated with actual evidence.

---

# 36. NO FALSE POSITIVES

Do not mark something as a failure simply because:

- wording differs
- response order differs
- citation order differs
- the model paraphrases source text
- a valid refusal occurs
- a cache hit occurs
- frontend formatting differs while semantics are correct

Likewise, do not mark something as PASS merely because the answer appears plausible.

Important financial claims require evidence.

---

# 37. FINAL REPORT

Generate:

```text
docs/SYSTEM_AUDIT_REPORT.md
```

The report must contain:

```text
1. Executive Summary

2. Audit Scope

3. Environment Verification

4. Authentication Verification

5. Frontend E2E Verification

6. Recommendation Questions Audit

7. Single-Ticker Audit
   - AAPL
   - MSFT
   - NVDA

8. Multi-Ticker / Cross-Entity Audit
   - AAPL + MSFT
   - AAPL + NVDA
   - MSFT + NVDA
   - AAPL + MSFT + NVDA
   - ALL

9. Demo Script Audit

10. Grounding Verification

11. MongoDB ↔ Qdrant Consistency

12. Memory Audit

13. Entity-Order Audit

14. Coreference Audit

15. Temporal Memory Audit

16. Cache Audit

17. API / Frontend Parity

18. Sync / Streaming Parity

19. Audit Logging Verification

20. Retrieval Trace Analysis

21. Code Integration Audit

22. Existing Test Suite Comparison

23. Existing Evaluation Artifact Comparison

24. Coverage Matrix

25. Failure Matrix

26. Root Cause Analysis

27. Severity Ranking

28. Recommended Repairs

29. Regression Test Recommendations

30. Production Readiness Verdict
```

---

# 38. FINAL VERDICT MUST ANSWER THESE QUESTIONS

Explicitly answer:

```text
1. Is the system actually grounded in the financial data?

2. Does each single ticker (AAPL/MSFT/NVDA) work correctly?

3. Does ALL / Cross-Entity work correctly?

4. Do pairwise multi-ticker comparisons work correctly?

5. Do three-company comparisons work correctly?

6. Does entity ordering remain correct?

7. Does retrieval remain balanced across companies?

8. Does memory correctly preserve entity context?

9. Does memory correctly resolve supported coreference?

10. Is memory isolated between sessions?

11. Does frontend behavior match the API?

12. Does streaming behavior match synchronous chat?

13. Are citations actually supporting generated claims?

14. Does the guardrail correctly validate numerical claims?

15. Does caching preserve ticker/year/session correctness?

16. Are MongoDB and Qdrant sufficiently consistent for production behavior?

17. Are the PROJECT_MAP claims consistent with actual runtime behavior?

18. What is the FIRST broken component for each major failure?

19. Which failures are blockers?

20. What must be fixed before considering the current single-pass RAG production-ready?

21. Is the system ready for a future Agentic/Multi-Hop phase?
```

For question 21, DO NOT implement the Agentic phase.

Only assess readiness based on the stability of the current system.

---

# 39. COMPLETION CONDITION

The audit is complete only after:

- repository architecture has been scanned
- frontend has been tested
- backend/API has been tested
- authentication has been verified
- all recommendation questions have been tested
- all demo-script questions have been tested
- AAPL has been tested
- MSFT has been tested
- NVDA has been tested
- pairwise cross-ticker cases have been tested
- three-company cases have been tested
- ALL scope has been tested
- entity-order permutations have been tested
- memory/coreference has been tested
- session isolation has been tested
- temporal references have been tested
- cache behavior has been tested
- sync/stream parity has been tested
- suspicious answers have been traced through the retrieval pipeline
- MongoDB/Qdrant evidence has been checked
- relevant code has been scanned
- existing tests/evaluation artifacts have been reviewed
- coverage matrix has been populated
- `docs/SYSTEM_AUDIT_REPORT.md` has been generated

The audit must distinguish clearly between:

```text
PASS
FAIL
PARTIAL
NOT TESTABLE
NOT APPLICABLE
```

Every important FAIL must have concrete evidence.

---

# 40. FINAL OPERATING PRINCIPLE

You are NOT being asked to make the system look good.

You are being asked to determine the truth.

If the system is correct, prove it with evidence.

If the system is wrong, reproduce the problem and locate the root cause.

If the documentation is wrong, report it.

If tests pass but live behavior fails, report the discrepancy.

If the answer is correct but the retrieval path is fragile, report the fragility.

If the answer is wrong but the data/retrieval are correct, isolate the generation problem.

Do not hide failures.

Do not patch failures during this audit.

Do not implement the future multi-hop/agentic architecture.

Produce a complete forensic picture so that a separate controlled repair phase can act on verified evidence.
