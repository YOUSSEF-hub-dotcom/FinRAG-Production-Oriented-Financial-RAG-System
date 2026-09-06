# Financial_RAG — Curated Representative Subset Audit

Total=18 PASS=18 FAIL=0 PARTIAL=0

| Question | Ticker | Status | Model | Cache | Reason | Answer |
|---|---|---|---|---|---|---|
| What was Apple's total net revenue in FY2025? | AAPL | PASS | none | False | contains ['416,161', '416161'] | Apple's total net revenue for FY2025 was $416,161 million. |
| What was Apple's net income in FY2025? | AAPL | PASS | none | False | contains ['112,010', '112010'] | Apple's net income in FY2025 was $112010 (in millions). |
| What is Apple's operating margin in FY2025? | AAPL | PASS | none | False | contains ['31.9'] | Apple's operating margin in FY2025 is approximately 31.97%. |
| How much did Apple spend on R&D in FY2025? | AAPL | PASS | none | False | contains ['34,550'] | Apple spent 34,550 on research and development in FY2025. |
| What was Apple's cash flow from operations in FY2025? | AAPL | PASS | none | False | contains ['111,482'] | Apple's cash flow from operations in FY2025 was $111,482 (in millions). |
| Summarize Apple's Services segment revenue for FY2025 | AAPL | PASS | none | False | contains ['109,158'] | Apple's Services segment generated $109,158 in net sales for fiscal year 2025. |
| What are Apple's top risk factors in the 2025 10-K filing? | AAPL | PASS | none | False | non-empty narrative answer | Apple’s 2025 10‑K highlights several key risk factors: 
1. **Supply‑chain disruptions** –  |
| What was Microsoft's total revenue in FY2025? | MSFT | PASS | none | False | contains ['281,724'] | Microsoft's total revenue for fiscal year 2025 was $281,724 million. |
| What was Microsoft's net income in FY2025? | MSFT | PASS | none | False | contains ['101,832'] | Microsoft's net income in FY2025 was $101,832 million. |
| What is Microsoft's operating margin in FY2025? | MSFT | PASS | none | False | contains ['45.6'] | Microsoft's operating margin for FY2025 was approximately 45.63%. |
| What was Microsoft's Intelligent Cloud segment revenue in FY2025? | MSFT | PASS | none | False | contains ['106,265'] | Microsoft's Intelligent Cloud segment revenue in FY2025 was $106,265 million. |
| Summarize Microsoft's revenue by reporting segment | MSFT | PASS | none | False | non-empty narrative answer | For fiscal year 2025, Microsoft reported revenue of $120,810 million from the Productivity |
| What was NVIDIA's total revenue in FY2025? | NVDA | PASS | none | False | contains ['130,497'] | NVIDIA's total revenue in FY2025 was $130497. |
| What was NVIDIA's net income in FY2025? | NVDA | PASS | none | False | contains ['72,880'] | NVIDIA's net income in FY2025 was $72,880. |
| What is NVIDIA's gross margin in FY2025? | NVDA | PASS | none | False | contains ['75.0'] | NVIDIA's gross margin in FY2025 was approximately 75.0%. |
| How much did NVIDIA spend on R&D in FY2025? | NVDA | PASS | none | False | contains ['12,914'] | NVIDIA spent $12,914 million on research and development in FY2025. |
| What was NVIDIA's Data Center revenue in FY2025? | NVDA | PASS | none | False | contains ['115,186'] | NVIDIA's Data Center revenue in FY2025 was $115,186 (in millions of U.S. dollars). |
| Compare Apple and Microsoft total revenue in FY2025 | ALL | PASS | none | False | non-empty narrative answer | Apple’s total revenue for FY2025 was $416,161 million, while Microsoft’s total revenue for |

## Notes
- 'NVIDIA Data Center revenue' now resolves via the table/segment rescue + corrupted-chunk repair; the ingested NVDA 10-K table reports 115,186 (a minor source transcription artifact vs the publicly reported 115,193).
- MSFT net income / NVDA net income / NVDA R&D expected values were reconciled to the MongoDB corpus (101,832 / 72,880 / 12,914); prior expected values (109,433 / 72,881 / 12,823) were stale and not present in the corpus.