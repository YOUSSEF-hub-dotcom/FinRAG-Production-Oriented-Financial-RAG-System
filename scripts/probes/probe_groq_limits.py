import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
want = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
for m in client.models.list().data:
    if m.id in want:
        print(m.id, "| context:", m.context_window, "| tpm:", getattr(m, "tpm_per_tier", None) or "?", "| active:", m.active)
