import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import groq

from config.settings import GROQ_API_KEY

client = groq.AsyncGroq(api_key=GROQ_API_KEY)


async def probe(model: str):
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output JSON."},
                {"role": "user", "content": 'Return JSON: {"ok": true}'},
            ],
            max_tokens=10,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        print(f"{model}: OK -> {resp.choices[0].message.content[:40]!r}")
    except Exception as exc:
        print(f"{model}: {str(exc)[:220]}")


async def main():
    for m in ["qwen/qwen3.6-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        await probe(m)


asyncio.run(main())
