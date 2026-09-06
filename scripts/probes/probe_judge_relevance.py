import asyncio

import groq

from config.settings import GROQ_API_KEY

client = groq.AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM = """\
You are a strict, deterministic LLM-as-a-Judge for a financial RAG system that
answers questions about SEC 10-K filings (AAPL, MSFT, NVDA). You evaluate
retrieval-and-generation quality using the Ragas metric methodology.

Rules:
- Score strictly on the evidence provided; never assume facts outside the
  given question, answer, ground truth and contexts.
- Respond with ONLY valid JSON. No markdown, no code fences, no preamble.
- Scores must be floats between 0.0 and 1.0 inclusive.
"""

PROMPT = """\
TASK: Answer Relevance -- measure how well the ANSWER addresses the QUESTION.

Score the answer 0.0-1.0 (1.0 = fully relevant, directly answers the question;
0.0 = completely off-topic). Return JSON:
{"score": <float 0.0-1.0>, "reasoning": "<short rationale>"}

QUESTION:
What were Apple's total net sales in fiscal year 2025 (ended September 27, 2025)?

ANSWER:
$416161 million
"""


async def main():
    for model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": PROMPT},
                ],
                max_tokens=1024,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            print(f"=== {model} ===")
            print(repr(resp.choices[0].message.content))
        except Exception as exc:
            print(f"=== {model} ===\n{str(exc)[:200]}")


asyncio.run(main())
