🧪 Financial RAG System — Live Demo Test Script
================================================================================
👤 SECTION 1: USER ACCOUNT VIEW (حساب المستخدم العادي)
Goal: Showcasing Multi-Ticker Balancing, Memory, Guardrails, & Retrieval Precision

------------------------------------------------------------------------------
Test 1: Single-Ticker High Precision & Risk Extraction
Target: Hybrid Search (Dense + BM25) + Table Shield Compaction
Check: يختبر استخرج أرقام مالية دقيقة للقطاع والنصوص الكيفية (Risk Factors) لـ MSFT
------------------------------------------------------------------------------
What was Microsoft's Total Revenue and Intelligent Cloud segment revenue in FY2025?


------------------------------------------------------------------------------
Test 2: Multi-Ticker Quantitative & Balanced Retrieval (تحديد سنة كل شركة بدقة)
Target: _balanced_ticker_subretrievals & Multi-Company CFO Prompt
Check: يختبر استرجاع بيانات 3 شركات بالتوازي وتوازن الـ Chunks حسب سنة كل تقرير
------------------------------------------------------------------------------
Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025.


------------------------------------------------------------------------------
Test 3: Memory & Coreference Resolution
Target: Intent Router Coreference + Session Context Memory (K=3)
Check: يختبر فهم "first company" وترجمتها لـ (Apple) مع تحديد سنتها المالية (FY2025)
------------------------------------------------------------------------------
How much did the Second company spend on Research and Development in that same fiscal year?

------------------------------------------------------------------------------
Test 3: Guardrail & Out-of-Scope (OOC) Detection
Target: Safety Guardrail & Dataset Scope Masking (AMZN / TSLA Exclusion)
Check: يختبر منع الهلوسة ورفض الإجابة عن شركات خارج القاعدة الإرشادية
------------------------------------------------------------------------------
What is Amazon's net income for FY2025, and what is Tesla's autonomous driving roadmap?


================================================================================
🛠️ SECTION 2: ADMIN ACCOUNT VIEW (حساب الأدمن)
Goal: Live Ingestion, Real-Time Chunk Indexing, & System Observability
------------------------------------------------------------------------------
Action 1: Ingestion Tab Workflow
Target: Document Upload API, Table Parsing, & Qdrant/Mongo Dual Indexing
Action:
1. انتقل لتبويب Ingestion
2. ارفع ملف: NVDA_AI_Infrastructure_Expansion_FY2026.pdf
3. اختر Ticker: NVDA | Fiscal Year: 2026
4. اضغط Upload واعرض مؤشرات المعالجة (Extraction → Chunking → Embedding → Upsert)
------------------------------------------------------------------------------
------------------------------------------------------------------------------
Test 5: Immediate Verification in Chat (Post-Ingestion)
Target: Zero-Latency Indexing Verification
Check: يختبر استعلام الـ Chat على البيانات المرفوعة حديثاً بدون إعادة تشغيل للـ Backend
------------------------------------------------------------------------------
According to the newly uploaded supplemental filing, what was NVIDIA's Data Center revenue in FY2026, and what was the year-over-year growth percentage?

------------------------------------------------------------------------------
Action 2: Analytics Tab Review
Target: AnalyticsTab.tsx, Context Quality Distribution, & Live MongoDB Audit Stream
Action:
1. انتقل لتبويب Analytics
2. اعرض رسم Context Quality Distribution (High, Mid, Low)
3. اعرض معدلات الـ Latency ورسم الـ Throughput
4. استعرض سجلات الـ Audit Logs الحية القادمة من rag_audit_logs في MongoDB
------------------------------------------------------------------------------