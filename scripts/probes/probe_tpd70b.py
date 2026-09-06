import asyncio

import groq

from config.settings import GROQ_API_KEY

client = groq.AsyncGroq(api_key=GROQ_API_KEY)

CONTEXT = (
    "| Item | 2025 | 2024 |\n|------|------|------|\n"
    + "\n".join(f"| Row {i} | 1234{i} | 2345{i} |" for i in range(1, 300))
)


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
        print(f"{model}: OK input={resp.usage.prompt_tokens}")
    except Exception as exc:
        msg = str(exc)
        if "tokens per day" in msg:
            i = msg.find("tokens per day")
            print(f"{model}: TPD BLOCKED ...{msg[i-40:i+190]}")
        else:
            print(f"{model}: {msg[:160]}")


asyncio.run(probe("openai/gpt-oss-20b"))
