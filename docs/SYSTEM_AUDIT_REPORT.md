# Financial_RAG — Full Forensic End-to-End System Audit Report

**Generated:** 2026-09-01T04:53:47.448517+00:00
**Audit Duration:** 1:13:45.737971
**Backend:** http://127.0.0.1:8000
**Total Tests:** 80
- **PASS:** 57
- **FAIL:** 14
- **PARTIAL:** 9
- **NOT_TESTABLE:** 0
- **NOT_TESTED:** 0

---

## 1. Executive Summary

The audit executed 80 test cases against the running Financial_RAG system.
Results: 57 passed, 14 failed, 9 partial, 0 not testable, 0 not tested.

**READINESS: NOT READY** — 23 result(s) require attention before production consideration.

## 2. Environment Verification

- **Backend URL:** http://127.0.0.1:8000
- **Frontend:** Next.js on port 3000 (running)
- **Data Directory:** /home/youssef/Financial_RAG/data
- **MongoDB URI:** mongodb://localhost:27017
- **Qdrant Path:** /home/youssef/Financial_RAG/data/qdrant_db

## 3. Authentication Verification

### Login
- **Status:** PASS
- **Severity:** CRITICAL
- **Actual:** access_token present=True

### Get Current User
- **Status:** PASS
- **Severity:** HIGH
- **Actual:** user_id=1015aebc7831462d8191fe15c4eabaab, role=admin

### Token Refresh
- **Status:** PASS
- **Severity:** MEDIUM
- **Actual:** HTTP 200

### Unauthorized Rejection
- **Status:** PASS
- **Severity:** MEDIUM
- **Actual:** HTTP 401


## 4. Single-Ticker Audit

### AAPL

| Question | Status | Latency (ms) | Model | Cache | Answer Snippet |
|---|---|---|---|---|---|
| What was Apple's total net revenue in FY2025? | FAIL | 2859 | cache | True | $416,161 |
| What was Apple's net income in FY2025? | PASS | 81124 | openai/gpt-oss-120b | False | Apple's net income in FY2025 was $112010 (in millions). |
| What is Apple's operating margin in FY2025? | PASS | 8267 | openai/gpt-oss-120b | False | Apple's operating margin in FY2025 is approximately 31.97%. |
| How much did Apple spend on R&D in FY2025? | PASS | 7346 | openai/gpt-oss-120b | False | Apple spent $34,550 million on research and development in FY2025. |
| What was Apple's cash flow from operations in FY2025? | PASS | 6892 | openai/gpt-oss-120b | False | Apple's cash flow from operations in FY2025 was $111,482. |
| Summarize Apple's Services segment revenue for FY2025 | PASS | 7078 | openai/gpt-oss-120b | False | Apple's Services segment generated $109,158 in fiscal year 2025. |
| What are Apple's top risk factors in the 2025 10-K filing? | PASS | 13052 | openai/gpt-oss-120b | False | {"content":"{\n  \"internal_thought\": \"Identify risk factor statements in the  |
| What are Apple's top risk factors in the 2025 10-K filing? | PASS | 11084 | openai/gpt-oss-120b | False | Apple’s 2025 Form 10‑K highlights several key risk factors:
1. **Supply‑chain ri |
| Analyze Apple's revenue segment breakdown for FY2025 | PASS | 8904 | openai/gpt-oss-120b | False | For fiscal year 2025 Apple reported total net sales of $416,161 million. The rev |
| How does Apple's Services segment margin compare to Products | PASS | 17874 | openai/gpt-oss-120b | False | In fiscal year 2025, Apple’s Services segment generated a gross margin of $82,31 |

### MSFT

| Question | Status | Latency (ms) | Model | Cache | Answer Snippet |
|---|---|---|---|---|---|
| What was Microsoft's total revenue in FY2025? | PASS | 225 | cache | True | Microsoft's total revenue in FY2025 was $281,724 million. |
| What was Microsoft's net income in FY2025? | PASS | 234 | cache | True | Microsoft's net income in FY2025 was $101,832 million. |
| What is Microsoft's operating margin in FY2025? | FAIL | 7174 | openai/gpt-oss-120b | False | Microsoft's operating margin in FY2025 was approximately 45.6%. |
| How much did Microsoft spend on R&D in FY2025? | PASS | 8023 | openai/gpt-oss-120b | False | Microsoft spent $32,488 million on research and development in FY2025. |
| What was Microsoft's Intelligent Cloud segment revenue in FY | PASS | 6925 | openai/gpt-oss-120b | False | Microsoft's Intelligent Cloud segment revenue in FY2025 was $106,265 million. |
| What are Microsoft's key growth drivers for Azure in 2025? | PASS | 7233 | openai/gpt-oss-120b | False | Microsoft cites several key growth drivers for Azure in FY2025: (1) strong deman |
| What regulatory risks does Microsoft disclose in the 10-K? | PASS | 8294 | openai/gpt-oss-120b | False | The requested financial information is not available in the provided reports. |
| What are the key growth drivers for Microsoft Azure in 2025? | PASS | 8330 | openai/gpt-oss-120b | False | In fiscal year 2025, Microsoft Azure’s growth was primarily driven by strong cus |
| Analyze Microsoft's AI capital expenditure plans | FAIL | 9354 | openai/gpt-oss-120b | False | {"content":"{\n  \"internal_thought\": \"Identify statements about AI capital ex |
| Summarize Microsoft's revenue by reporting segment | PASS | 7194 | openai/gpt-oss-120b | False | In FY2025, Microsoft reported segment revenues of $120810 million for Productivi |

### NVDA

| Question | Status | Latency (ms) | Model | Cache | Answer Snippet |
|---|---|---|---|---|---|
| What was NVIDIA's total revenue in FY2025? | PASS | 201 | cache | True | NVIDIA's total revenue in FY2025 was $130497 (in millions of U.S. dollars). |
| What was NVIDIA's net income in FY2025? | PASS | 7512 | openai/gpt-oss-120b | False | NVIDIA's net income in FY2025 was $72880. |
| What is NVIDIA's gross margin in FY2025? | PASS | 7785 | openai/gpt-oss-120b | False | NVIDIA's gross margin in FY2025 was approximately 75.0%. |
| How much did NVIDIA spend on R&D in FY2025? | PASS | 7394 | openai/gpt-oss-120b | False | NVIDIA spent 12,914 (in millions) on research and development in FY2025. |
| What was NVIDIA's Data Center revenue in FY2025? | PASS | 183 | cache | True | NVIDIA's Data Center revenue in FY2025 was $115,186 (in millions of U.S. dollars |
| What export restriction risks does NVIDIA disclose? | PASS | 225 | cache | True | NVIDIA warns that export controls and sanctions pose several risks: reduced dema |
| Summarize NVIDIA's competitive advantages in AI chips | PASS | 225 | cache | True | NVIDIA’s competitive edge in AI chips stems from its full‑stack computing model  |
| Analyze NVIDIA's data center revenue growth trajectory | PASS | 8251 | openai/gpt-oss-120b | False | NVIDIA's Data Center revenue has surged dramatically over the last three fiscal  |
| What are NVIDIA's competitive advantages in AI chips? | PASS | 13241 | openai/gpt-oss-120b | False | {"content":"{\n  \"internal_thought\": \"Identify statements in the provided tex |
| Summarize NVIDIA's gross margin expansion drivers | PASS | 8500 | openai/gpt-oss-120b | False | NVIDIA’s gross margin expanded in FY2025 because revenue more than doubled to $1 |


## 5. Multi-Ticker / Cross-Entity Audit

- **Query: Compare the operating margins and total net revenue between **: PARTIAL (latency=17643ms)
- **Query: Compare the operating margins and total net revenue between **: PARTIAL (latency=16842ms)
- **Query: Compare the operating margins and total net revenue between **: PARTIAL (latency=17078ms)
- **Query: Compare the operating margins and total net revenue between **: PARTIAL (latency=16699ms)
- **Query: Compare the operating margins and total net revenue between **: FAIL (latency=16558ms)
- **Query: Compare the operating margins and total net revenue between **: FAIL (latency=17022ms)
- **Query: Compare the operating margins and total net revenue between **: PARTIAL (latency=17555ms)
- **Query: Compare the operating margins and total net revenue between **: FAIL (latency=17628ms)
- **Query: Compare the operating margins and total net revenue between **: FAIL (latency=17210ms)
- **Query: Compare the financial health of AAPL, MSFT, and NVDA**: FAIL (latency=18350ms)
- **Query: Which company has the best operating margin trend?**: FAIL (latency=17572ms)
- **Query: Summarize key industry risks across all three tech giants**: FAIL (latency=120092ms)
- **Query: Compare the operating margins and total net revenue between **: FAIL (latency=109217ms)
- **Query: Compare the financial health of AAPL, MSFT, and NVDA**: PASS (latency=61584ms)
- **Query: Which company has the best operating margin trend?**: PASS (latency=24320ms)
- **Query: Summarize key industry risks across all three tech giants**: PASS (latency=83208ms)
- **Query: How much did the first company spend on Research and Develop**: FAIL (latency=14214ms)
- **Query: What is Amazon's net income for FY2025, and what is Tesla's **: PASS (latency=20880ms)

### ALL / Cross-Entity

- **Compare the operating margins and total net revenue between **: PARTIAL
- **Compare the operating margins and total net revenue between **: PARTIAL
- **Compare the operating margins and total net revenue between **: PARTIAL
- **Compare the operating margins and total net revenue between **: PARTIAL
- **Compare the operating margins and total net revenue between **: FAIL
- **Compare the operating margins and total net revenue between **: FAIL
- **Compare the operating margins and total net revenue between **: PARTIAL
- **Compare the operating margins and total net revenue between **: FAIL
- **Compare the operating margins and total net revenue between **: FAIL
- **Compare the financial health of AAPL, MSFT, and NVDA**: FAIL
- **Which company has the best operating margin trend?**: FAIL
- **Summarize key industry risks across all three tech giants**: FAIL
- **Compare the operating margins and total net revenue between **: FAIL
- **Compare the financial health of AAPL, MSFT, and NVDA**: PASS
- **Which company has the best operating margin trend?**: PASS
- **Summarize key industry risks across all three tech giants**: PASS
- **How much did the first company spend on Research and Develop**: FAIL
- **What is Amazon's net income for FY2025, and what is Tesla's **: PASS

## 6. Memory Audit

- **Query: How much did the first company spend on Research and Develop**: FAIL
  - Actual: NVIDIA spent $12,914 million on Research and Development in FY2025.
- **Memory: first company -> Apple**: PASS
  - Actual: Apple spent $34,550 million on Research and Development in FY2025.
- **Memory: second company -> Microsoft**: PASS
  - Actual: Microsoft spent $32,488 million on Research and Development in FY2025.
- **Memory: third company -> NVIDIA**: PASS
  - Actual: NVIDIA spent $12,914 million on Research and Development in FY2025.
- **Memory entity order: Session B (NVIDIA first)**: PASS
  - Actual: NVIDIA reported FY2025 revenue of $130,497 M, operating income of $81,453 M, and net income of $72,880 M.
- **Session Isolation (same session coreference)**: PASS
  - Actual: Apple's revenue for FY2025 was $416161.

## 7. Cache Audit

- **Cache Hit (identical query)**: PASS
  - Cache Hit: True
- **Cache Isolation (AAPL vs MSFT)**: PASS
  - Cache Hit: False

## 8. API Parity

- **Sync vs Stream Parity**: PARTIAL

## 9. Grounding Verification

- **Grounding Check (AAPL FY2025 Revenue)**: PASS
  - Grounding verified via live API (Qdrant lock held by server; direct introspection unavailable). MongoDB docs=10.

## 10. Negative / Refusal Testing

- **Negative: What is Amazon's net income for FY2025, and what i**: PASS
  - Actual: The requested financial information is not available in the provided reports.
- **Negative: What is Meta's operating margin for FY2025?**: PASS
  - Actual: The requested financial information is not available in the provided reports.

## 11. Audit Logging Verification

- **Audit Logs Retrieval**: PASS

## 12. Code Integration Scan

- **Code Integration Scan**: PASS
  - Pre-retrieval is OFF by default (opt-in); Balanced per-ticker sub-retrievals implemented; Cross-ticker bypass implemented; Context augmentation implemented

## 13. Failure Matrix

| ID | Test | Ticker | Severity | First Failure Stage | Root Cause |
|---|---|---|---|---|---|
| 1 | Query: What was Apple's total net revenue in FY202 | AAPL | MEDIUM | Generation | Answer does not mention the requested ticker AAPL. |
| 2 | Query: What is Microsoft's operating margin in FY2 | MSFT | MEDIUM | Retrieval | Only 0/1 claims supported by MongoDB evidence. Uns |
| 3 | Query: Compare the operating margins and total net | ALL | MEDIUM | Retrieval | Only 4/13 claims supported by MongoDB evidence. Un |
| 4 | Query: Compare the operating margins and total net | ALL | MEDIUM | Retrieval | Only 5/11 claims supported by MongoDB evidence. Un |
| 5 | Query: Compare the operating margins and total net | ALL | MEDIUM | Retrieval | Only 7/16 claims supported by MongoDB evidence. Un |
| 6 | Query: Compare the operating margins and total net | ALL | MEDIUM | Retrieval | Only 4/12 claims supported by MongoDB evidence. Un |
| 7 | Query: Compare the financial health of AAPL, MSFT, | ALL | MEDIUM | Retrieval | Only 0/18 claims supported by MongoDB evidence. Un |
| 8 | Query: Which company has the best operating margin | ALL | MEDIUM | Retrieval | Only 0/3 claims supported by MongoDB evidence. Uns |
| 9 | Query: Summarize key industry risks across all thr | ALL | MEDIUM | API | HTTP request failed:  |
| 10 | Query: Compare the operating margins and total net | ALL | MEDIUM | Retrieval | Only 0/6 claims supported by MongoDB evidence. Uns |
| 11 | Query: Analyze Microsoft's AI capital expenditure  | MSFT | MEDIUM | Generation | Data exists in MongoDB/Qdrant but system returned  |
| 12 | Query: Compare Microsoft's cloud vs on-premise rev | MSFT | MEDIUM | Generation | Data exists in MongoDB/Qdrant but system returned  |
| 13 | Query: What is Microsoft's cash flow from operatio | MSFT | MEDIUM | Generation | Narrative query refused; MongoDB has data but Qdra |
| 14 | Query: How much did the first company spend on Res | ALL | MEDIUM | Guardrail | System answered out-of-scope question instead of r |

## 14. Coverage Matrix

| Test Area | AAPL | MSFT | NVDA | AAPL+MSFT | AAPL+NVDA | MSFT+NVDA | 3-Company | ALL |
|---|---|---|---|---|---|---|---|---|
| Direct factual | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Numerical grounding | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Recommendation | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Demo script | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Memory | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Coreference | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Entity order | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Temporal memory | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Cache isolation | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| API parity | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Streaming parity | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Grounding | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

## 15. Final Verdict

**Production Readiness:** NOT READY

**Blockers:** 23

**Recommendation:** Fix identified failures, partials, and untested areas before production deployment.