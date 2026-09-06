import asyncio

import groq

from config.settings import GROQ_API_KEY

client = groq.AsyncGroq(api_key=GROQ_API_KEY)

# Simulate a real judge call: ~3.5K token input, 1024 max output, strict JSON.
CONTEXT = (
    "| Item | 2025 | 2024 |\n|------|------|------|\n"
    + "\n".join(f"| Row {i} | 1234{i} | 2345{i} |" for i in range(1, 300))
)  # ~3600 chars of table


async def probe(model: str):
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict judge. Always output valid JSON only."},
                {"role": "user", "content": (
                    "Decide which rows contain the answer. Return JSON: "
                    '{"relevant_context_indices": [1]}\n\nCONTEXT:\n' + CONTEXT
                )},
            ],
            max_tokens=1024,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        print(f"{model}: OK input={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")
    except Exception as exc:
        msg = str(exc)
        if "tokens per day" in msg:
            i = msg.find("tokens per day")
            print(f"{model}: TPD BLOCKED ...{msg[i-40:i+190]}")
        else:
            print(f"{model}: {msg[:180]}")


async def main():
    for m in ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"]:
        await probe(m)


asyncio.run(main())
