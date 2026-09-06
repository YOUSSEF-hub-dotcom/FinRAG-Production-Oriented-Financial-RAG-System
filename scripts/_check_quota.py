import httpx
from config.settings import GROQ_API_KEY

h = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
r = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    json={"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
    headers=h, timeout=30,
)
print("status", r.status_code)
for k, v in r.headers.items():
    if "ratelimit" in k.lower() or "limit" in k.lower():
        print(k, "=", v)
print(r.text[:300])
