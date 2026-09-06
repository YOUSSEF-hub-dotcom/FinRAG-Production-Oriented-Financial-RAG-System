#!/usr/bin/env python3
"""Minimal verification that the backend config (new Groq key) can make one
authenticated LLM request. Does NOT print the API key. NOT an audit."""
import config.settings as S  # loads .env via load_dotenv

from langchain_groq import ChatGroq

key_present = bool(S.GROQ_API_KEY)
print("KEY_PRESENT:", key_present)
print("KEY_SOURCE_LEN:", len(S.GROQ_API_KEY or ""))

llm = ChatGroq(
    groq_api_key=S.GROQ_API_KEY,
    model_name=S.GROQ_FALLBACK_MODEL,  # openai/gpt-oss-20b (lighter)
    max_tokens=12,
    temperature=0.0,
)

try:
    resp = llm.invoke("Reply with the single word: OK")
    print("REQUEST_SUCCEEDED: True")
    print("RESPONSE:", repr(getattr(resp, "content", resp))[:60])
    print("RATE_LIMITED: False")
except Exception as e:  # noqa: BLE001
    msg = str(e)
    is_rl = ("429" in msg) or ("rate_limit" in msg.lower()) or ("Rate limit" in msg)
    print("REQUEST_SUCCEEDED: False")
    print("RATE_LIMITED:", is_rl)
    print("ERROR_HEAD:", msg[:200])
