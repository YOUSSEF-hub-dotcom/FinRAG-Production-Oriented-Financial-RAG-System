import sys
sys.path.insert(0, "/home/youssef/Financial_RAG/src/3_pre_retrieval")
from intent_router import RuleBasedIntentRouter
r = RuleBasedIntentRouter()

def test(q, h):
    out = r.route(q, [{"role":"user","content":x} for x in h])
    print(f"Q:{q!r:45} H:{h[0][:30]!r:30} -> ticker={out.ticker} year={out.fiscal_year}")

# Session A: Apple, Microsoft, NVIDIA order
h_a = ["Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025."]
test("How much did the first company spend on Research and Development in that same fiscal year?", h_a)
test("How much did the second company spend on Research and Development?", h_a)
test("What about the third company?", h_a)
# Session B: NVIDIA, Microsoft, Apple order
h_b = ["Compare the operating margins and total net revenue between NVIDIA, Microsoft, and Apple for FY2025."]
test("What did the first company report?", h_b)
test("How much did the second company spend on Research and Development?", h_b)
# Session C: Apple net income FY2025 then its revenue
h_c = ["What is Apple's net income for FY2025?"]
test("What is its revenue?", h_c)
# Session isolation: FY2024 case
h_c2 = ["What is Apple's net income for FY2024?"]
test("What is its revenue?", h_c2)
# Cross-check fiscal year inheritance for same fiscal year phrase
test("How much did the first company spend on R&D in that same fiscal year?", h_a)
