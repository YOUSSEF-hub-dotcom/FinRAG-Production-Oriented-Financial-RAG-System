import sys
sys.path.insert(0, "/home/youssef/Financial_RAG/src/3_pre_retrieval")
from intent_router import RuleBasedIntentRouter

r = RuleBasedIntentRouter()

def test(query, history):
    out = r.route(query, history)
    print(f"Q: {query!r:50} H: {[h['content'][:30] for h in history]} -> ticker={out.ticker} year={out.fiscal_year} standalone={out.standalone_query[:60]!r}")

# Memory tests as in phase_7
history = [{"role":"user","content":"Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025."}]
test("How much did the first company spend on Research and Development in that same fiscal year?", history)
test("How much did the second company spend on Research and Development?", history)
test("What about the third company?", history)

history_b = [{"role":"user","content":"Compare the operating margins and total net revenue between NVIDIA, Microsoft, and Apple for FY2025."}]
test("What did the first company report?", history_b)

history_c = [{"role":"user","content":"What is Apple's net income for FY2025?"}]
test("What is its revenue?", history_c)
test("What is its revenue? FY2024 case", [{"role":"user","content":"What is Apple's net income for FY2024?"}])

# Also test fiscal year inheritance
history_fy = [{"role":"user","content":"Compare Apple, Microsoft, NVIDIA FY2025"}]
test("How much did the first company spend on R&D?", history_fy)
test("What is its revenue?", history_fy)
