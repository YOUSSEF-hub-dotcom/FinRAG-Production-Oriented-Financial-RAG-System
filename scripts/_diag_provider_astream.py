import asyncio
import time
import sys
sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from config.settings import GROQ_API_KEY, GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL, \
    LLM_MAX_TOKENS, LLM_SEED, LLM_TEMPERATURE
from langchain_groq import ChatGroq


def build_llm(model_name):
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        model_kwargs={
            "seed": LLM_SEED,
            "response_format": {"type": "json_object"},
        },
    )


async def main():
    llm = build_llm(GROQ_PRIMARY_MODEL)
    msgs = [
        {"role": "system", "content": "You are a financial assistant. Respond ONLY with JSON containing an 'answer' key."},
        {"role": "user", "content": 'Return JSON: {"answer": "The quick brown fox jumps over the lazy dog."}'},
    ]
    t0 = time.time()
    first = None
    n = 0
    lens = []
    async for chunk in llm.astream(msgs):
        content = getattr(chunk, "content", None) or ""
        now = (time.time() - t0) * 1000
        if not content:
            continue
        if first is None:
            first = now
        n += 1
        lens.append(len(content))
    total = (time.time() - t0) * 1000
    print("provider direct astream (%s):" % GROQ_PRIMARY_MODEL)
    print("  chunks_yielded =", n)
    print("  first_chunk_ms =", round(first, 1), " total_ms =", round(total, 1))
    print("  chunk_lengths  =", lens)
    if n == 1:
        print("  => provider coalesced entire completion into ONE chunk (no token streaming)")

asyncio.run(main())
