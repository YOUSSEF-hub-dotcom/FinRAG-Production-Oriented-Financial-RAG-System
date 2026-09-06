import asyncio

import groq

from config.settings import GROQ_API_KEY

client = groq.AsyncGroq(api_key=GROQ_API_KEY)

BIG_TEXT = ("The consolidated income statement reports revenue, costs and "
           "expenses for the fiscal year. Net sales were $416,161 million "
           "with cost of sales of $257,641 million. " * 40)  # ~2000 tokens


async def probe(model: str):
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output JSON."},
                {"role": "user", "content": f"Repeat this text then return JSON {{'ok': true}}:\n{BIG_TEXT}"},
            ],
            max_tokens=50,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        print(f"{model}: OK ({resp.usage.total_tokens} tok)")
    except Exception as exc:
        msg = str(exc)
        if "tokens per day" in msg:
            i = msg.find("tokens per day")
            print(f"{model}: TPD BLOCKED ...{msg[i-40:i+200]}")
        else:
            print(f"{model}: {msg[:160]}")


async def main():
    for m in [
        "openai/gpt-oss-120b",
        "gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
    ]:
        await probe(m)


asyncio.run(main())
