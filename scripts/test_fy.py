import sys
sys.path.insert(0, "/home/youssef/Financial_RAG/src/3_pre_retrieval")
from intent_router import RuleBasedIntentRouter
import re
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
r = RuleBasedIntentRouter()
history = [{"role":"user","content":"Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025."}]
text = "How much did the first company spend on Research and Development in that same fiscal year?"
print("text matches:", _YEAR_PATTERN.findall(text))
print("history text:", history[0]["content"])
print("history matches:", _YEAR_PATTERN.findall(history[0]["content"]))
# call directly
print("extract with history:", r._extract_fiscal_year(text, history))
print("extract without history:", r._extract_fiscal_year(text))
# also test route
out = r.route(text, history)
print("route fiscal_year:", out.fiscal_year)
# check what history param is passed correctly
# also test second case
text2 = "What is its revenue?"
history2 = [{"role":"user","content":"What is Apple's net income for FY2025?"}]
print("test2:", r._extract_fiscal_year(text2, history2), r.route(text2, history2).fiscal_year)
