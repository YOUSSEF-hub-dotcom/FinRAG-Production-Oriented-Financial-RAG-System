"""
Generation Engine — Groq-powered LLM with primary/fallback execution.

Handles:
  - XML context enclosure for retrieved documents
  - Conversation memory management (truncated to last K messages)
  - Primary-fallback execution with exponential-backoff retry
  - CFO-grade system prompt with zero-hallucination instructions
  - Token streaming support
  - MLflow experiment tracking for generation events
"""

import hashlib
import json
import re
import time
import uuid
from typing import Any, AsyncIterator

import mlflow
from langchain_core.exceptions import OutputParserException
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import GenerationChunk
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from config.logging_config import get_logger
from config.settings import (
    GROQ_API_KEY,
    GROQ_FALLBACK_MODEL,
    GROQ_PRIMARY_MODEL,
    LLM_HISTORY_K,
    LLM_MAX_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
)

logger = get_logger("generation.engine")

# ---------------------------------------------------------------------------
# System Prompt — CFO-grade zero-hallucination instructions
# ---------------------------------------------------------------------------
_CFO_SYSTEM_PROMPT = """\
You are a senior financial analyst assistant for SEC 10-K filings.

STRICT RULES — ZERO HALLUCINATION:
1. ONLY use numbers and facts present in the <CONTEXT> documents provided.
2. NEVER fabricate, estimate, or interpolate financial figures.
3. Preserve ALL financial notation exactly: $ for currency, % for percentages,
   parentheses (X) for negatives, and scale suffixes M (millions), B (billions), K (thousands).
4. When scaling is mentioned (e.g., "in millions"), compute the full number explicitly
   in your internal_thought before answering.
5. If the requested information is NOT available in the provided reports, respond with:
   "The requested financial information is not available in the provided reports."
6. Always provide sources in the format: "ticker - fiscal_year - section - page_number"

FINANCIAL LINE EXTRACTION RULES (CRITICAL):
7. FISCAL YEAR MATCHING: Before extracting any number, verify the fiscal year
   matches the one requested in the question. The <CONTEXT> documents may contain
   multiple years (e.g., FY2025 and FY2024). Always pick the row for the
   EXACT fiscal year the question asks about. Do NOT accidentally use a prior
   year\'s figure.
8. CONSOLIDATED STATEMENTS PRIORITY: When a line item appears in both the
   Consolidated Statements of Operations (Income Statement) AND in Notes or
   supplemental tables, ALWAYS use the figure from the Consolidated Statements
   as the authoritative source. The primary financial statements are the
   official reported numbers.
9. CROSS-COMPANY COMPARATIVE QUERIES: When the question explicitly asks to
   compare or contrast TWO different companies\' figures (e.g., "Apple vs
   Microsoft", "which is larger", "comparing X to Y"), and both companies\'
   data is present in the <CONTEXT> as separate <ENTITY> blocks, you MUST:
   a. Extract the relevant figure from EACH entity\'s documents.
   b. Present BOTH figures in the answer with the company name attached.
   c. State which is larger/smaller and by how much if asked.
   d. NEVER refuse such a query when data from both companies IS in the context.
   e. If one company\'s figure is missing from the context, state what is
      available and clearly note what is not.

OUTPUT FORMAT — You MUST respond with ONLY valid JSON. No markdown, no code fences,
no conversational preambles, no explanations outside the JSON. Output pure JSON
matching this exact schema:
{
  "internal_thought": "Step-by-step reasoning with number verification",
  "extracted_raw_data": "Exact verbatim facts from the context documents",
  "answer": "Executive-level answer using only extracted_raw_data",
  "sources": ["TICKER - YEAR - SECTION - PAGE"]
}
"""

# Appended to the system prompt when cross-company comparison intent is
# detected, so the LLM never scopes its answer to a single active ticker.
_MULTI_TICKER_INSTRUCTION = (
    "\nMULTI-COMPANY COMPARISON MODE (ACTIVE):\n"
    "You have context from multiple companies (e.g. AAPL, MSFT, NVDA). "
    "Answer the user's comparison question directly using all provided "
    "contexts. Do NOT output 'information not available'."
)

# ---------------------------------------------------------------------------
# Context Formatting
# ---------------------------------------------------------------------------

def format_context_xml(documents: list[dict]) -> str:
    """
    Enclose retrieved candidate documents in structured XML tags.

    Documents are grouped by ticker under <ENTITY ticker="..."> wrappers
    inside <CONTEXT>.  Each document is wrapped in <DOCUMENT> with metadata
    attributes.  When multiple entities are present (cross-company queries),
    explicit entity grouping ensures the LLM can distinguish which data
    belongs to which company.

    Args:
        documents: List of dicts with keys: text, metadata (ticker, fiscal_year,
                   section, contains_table, chunk_id, page_number).

    Returns:
        XML-formatted context string.
    """
    if not documents:
        return "<CONTEXT>\nNo documents retrieved.\n</CONTEXT>"

    from collections import OrderedDict
    groups: OrderedDict[str, list] = OrderedDict()
    for doc in documents:
        meta = doc.get("metadata", {})
        ticker = meta.get("ticker", "UNKNOWN")
        groups.setdefault(ticker, []).append(doc)

    multi_entity = len(groups) > 1
    parts: list[str] = ["<CONTEXT>"]
    doc_idx = 0
    for ticker_sym, ticker_docs in groups.items():
        if multi_entity:
            parts.append(f'<ENTITY ticker="{ticker_sym}">')
        for doc in ticker_docs:
            doc_idx += 1
            meta = doc.get("metadata", {})
            year = meta.get("fiscal_year", "UNKNOWN")
            section = meta.get("section", "General")
            has_table = meta.get("contains_table", False)
            page = meta.get("page_number", "N/A")
            chunk_id = doc.get("chunk_id", f"doc_{doc_idx}")

            parts.append(
                f'<DOCUMENT id="{chunk_id}" ticker="{ticker_sym}" '
                f'fiscal_year="{year}" section="{section}" '
                f'page="{page}" contains_table="{str(has_table).lower()}">'
            )
            parts.append(doc.get("text", ""))
            parts.append("</DOCUMENT>")
            parts.append("")
        if multi_entity:
            parts.append("</ENTITY>")
            parts.append("")

    parts.append("</CONTEXT>")
    return "\n".join(parts)

def _hash_query(query: str) -> str:
    """Deterministic short hash for cache keys."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Memory Management
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Sliding-window conversation memory.

    Retains the system prompt plus the last K (human, assistant) message pairs.
    Older messages are discarded to stay within token budgets.
    """

    def __init__(self, k: int = LLM_HISTORY_K):
        self._k = k
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def add_human(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_ai(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))

    def set_system(self, content: str) -> None:
        """Prepend or replace the system message."""
        if self._messages and isinstance(self._messages[0], SystemMessage):
            self._messages[0] = SystemMessage(content=content)
        else:
            self._messages.insert(0, SystemMessage(content=content))

    def truncate(self) -> list[BaseMessage]:
        """
        Return messages respecting the K-pair window.

        Always keeps the system prompt (index 0), then the last 2*K
        non-system messages.
        """
        if not self._messages:
            return []

        system_msgs = [m for m in self._messages if isinstance(m, SystemMessage)]
        non_system = [m for m in self._messages if not isinstance(m, SystemMessage)]

        # Keep last 2*K non-system messages (K pairs = K human + K AI)
        max_non_system = self._k * 2
        trimmed = non_system[-max_non_system:] if len(non_system) > max_non_system else non_system

        return system_msgs + trimmed

    def clear(self) -> None:
        self._messages.clear()


# ---------------------------------------------------------------------------
# Generation Engine
# ---------------------------------------------------------------------------

class FinancialRAGGenerator:
    """
    Groq-powered generation engine with primary/fallback, retry, and streaming.

    Flow:
        1. Build XML context from retrieved chunks
        2. Manage conversation memory (truncate to last K pairs)
        3. Call primary model with exponential-backoff retry
        4. On exhausted retries, silently fallback to secondary model
        5. Parse JSON output into ConsolidatedFinancialAnswer via Pydantic
        6. Log generation metrics to MLflow
    """

    def __init__(
        self,
        primary_model: str = GROQ_PRIMARY_MODEL,
        fallback_model: str = GROQ_FALLBACK_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        seed: int = LLM_SEED,
        history_k: int = LLM_HISTORY_K,
    ):
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._memory = ConversationMemory(k=history_k)
        self._run_id: str = ""
        self._mlflow_active = False

    # --- Internal helpers ---------------------------------------------------

    def _build_llm(self, model_name: str) -> ChatGroq:
        """Construct a ChatGroq instance for the given model."""
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            model_kwargs={
                "seed": self._seed,
                "response_format": {"type": "json_object"},
            },
        )

    def _build_structured_llm(self, model_name: str) -> Runnable:
        """Bind ChatGroq with .with_structured_output(ConsolidatedFinancialAnswer).

        LangChain enforces the ConsolidatedFinancialAnswer Pydantic schema
        server-side (json_mode, which Groq maps to response_format json_object)
        and returns a validated instance directly. Falling back to a raw
        ChatGroq (via _build_llm) is handled by the callers when the structured
        binding fails to produce a parseable object.
        """
        from schemas import ConsolidatedFinancialAnswer

        return self._build_llm(model_name).with_structured_output(
            ConsolidatedFinancialAnswer,
            method="json_mode",
        )

    def _start_mlflow(self) -> None:
        """Start an MLflow run if not already active."""
        try:
            mlflow.set_experiment("financial_rag_generation")
            self._run_id = str(uuid.uuid4())[:8]
            mlflow.start_run(run_name=f"gen_{self._run_id}", nested=True)
            mlflow.log_param("primary_model", self._primary_model)
            mlflow.log_param("fallback_model", self._fallback_model)
            mlflow.log_param("temperature", self._temperature)
            mlflow.log_param("max_tokens", self._max_tokens)
            mlflow.log_param("seed", self._seed)
            mlflow.log_param("history_k_truncated", self._memory._k)
            self._mlflow_active = True
        except Exception as exc:
            logger.warning("MLflow start failed (non-blocking): %s", exc)
            self._mlflow_active = False

    def _end_mlflow(
        self,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """End the active MLflow run, logging optional metrics."""
        if not self._mlflow_active:
            return
        try:
            if metrics:
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow end failed (non-blocking): %s", exc)
        finally:
            self._mlflow_active = False

    def _log_mlflow_param(self, key: str, value: Any) -> None:
        if not self._mlflow_active:
            return
        try:
            mlflow.log_param(key, value)
        except Exception:
            pass

    def _log_mlflow_metric(self, key: str, value: Any) -> None:
        if not self._mlflow_active:
            return
        try:
            mlflow.log_metric(key, value)
        except Exception:
            pass

    # --- Public API ---------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_docs: list[dict],
        stream: bool = False,
        multi_ticker: bool = False,
    ) -> dict[str, Any]:
        """
        Synchronous generation with primary/fallback and retry.

        Args:
            query: User question string.
            retrieved_docs: List of chunk dicts from vector retrieval.
            stream: If True, tokens are printed as they arrive (no JSON parsing).
            multi_ticker: Inject the cross-company comparison directive into
                the system prompt (set when the query compares 2+ companies).

        Returns:
            Dict with keys: raw_output, parsed (ConsolidatedFinancialAnswer or None),
            model_used, fallback_triggered, ttft_ms.
        """
        self._start_mlflow()
        t_start = time.time()
        fallback_triggered = False

        # Build context and memory
        context_xml = format_context_xml(retrieved_docs)
        system_prompt = _CFO_SYSTEM_PROMPT + (
            _MULTI_TICKER_INSTRUCTION if multi_ticker else ""
        )
        self._memory.set_system(system_prompt)
        # Store ONLY the question in persistent memory. The retrieved context
        # is re-fetched every turn and must NOT be replayed in history — doing
        # so previously pushed multi-turn prompts past the model token limit
        # (HTTP 413), which made follow-up turns fail entirely.
        self._memory.add_human(query)
        messages = self._memory.truncate()
        # Inject the retrieved context into the CURRENT turn's human message
        # only (not into stored memory) so this turn stays grounded.
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1] = HumanMessage(content=f"{context_xml}\n\nQuestion: {query}")

        # Try primary model with retry
        raw_output, model_used, ttft_ms = self._call_with_retry(
            self._primary_model, messages, stream=stream
        )

        # If primary returned error, try fallback
        if raw_output is None:
            logger.warning(
                "Primary model %s exhausted — falling back to %s",
                self._primary_model,
                self._fallback_model,
            )
            fallback_triggered = True
            raw_output, model_used, ttft_ms = self._call_with_retry(
                self._fallback_model, messages, stream=stream
            )

        # If both failed, return safe fallback
        if raw_output is None:
            safe_msg = (
                "The requested financial information is not available "
                "in the provided reports."
            )
            total_ms = (time.time() - t_start) * 1000
            self._memory.add_ai(safe_msg)
            self._log_mlflow_metric("fallback_triggered", 1)
            self._log_mlflow_metric("guardrail_passed", 0)
            self._log_mlflow_metric("ttft_ms", 0)
            self._log_mlflow_metric("total_generation_tokens", 0)
            self._end_mlflow()
            return {
                "raw_output": safe_msg,
                "parsed": None,
                "model_used": "none",
                "fallback_triggered": True,
                "ttft_ms": 0,
            }

        # Parse JSON from LLM output
        parsed = self._parse_json_output(raw_output)
        total_ms = (time.time() - t_start) * 1000
        total_tokens = self._estimate_tokens(raw_output)

        # Update memory
        self._memory.add_ai(raw_output)

        # Log to MLflow
        self._log_mlflow_metric("fallback_triggered", int(fallback_triggered))
        self._log_mlflow_metric("ttft_ms", round(ttft_ms, 2))
        self._log_mlflow_metric("total_generation_tokens", total_tokens)
        self._end_mlflow()

        logger.info(
            "Generation complete: model=%s fallback=%s ttft=%.0fms tokens=%d",
            model_used,
            fallback_triggered,
            ttft_ms,
            total_tokens,
        )

        return {
            "raw_output": raw_output,
            "parsed": parsed,
            "model_used": model_used,
            "fallback_triggered": fallback_triggered,
            "ttft_ms": round(ttft_ms, 2),
        }

    async def agenerate(
        self,
        query: str,
        retrieved_docs: list[dict],
        multi_ticker: bool = False,
    ) -> dict[str, Any]:
        """
        Async generation with primary/fallback and retry.

        Same logic as generate() but uses ainvoke for non-blocking execution.
        """
        self._start_mlflow()
        t_start = time.time()
        fallback_triggered = False

        context_xml = format_context_xml(retrieved_docs)
        system_prompt = _CFO_SYSTEM_PROMPT + (
            _MULTI_TICKER_INSTRUCTION if multi_ticker else ""
        )
        self._memory.set_system(system_prompt)
        self._memory.add_human(query)
        messages = self._memory.truncate()
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1] = HumanMessage(content=f"{context_xml}\n\nQuestion: {query}")

        # Primary model
        raw_output, model_used, ttft_ms = await self._acall_with_retry(
            self._primary_model, messages
        )

        if raw_output is None:
            logger.warning("Primary exhausted — fallback to %s", self._fallback_model)
            fallback_triggered = True
            raw_output, model_used, ttft_ms = await self._acall_with_retry(
                self._fallback_model, messages
            )

        if raw_output is None:
            safe_msg = (
                "The requested financial information is not available "
                "in the provided reports."
            )
            self._memory.add_ai(safe_msg)
            self._log_mlflow_metric("fallback_triggered", 1)
            self._log_mlflow_metric("guardrail_passed", 0)
            self._end_mlflow()
            return {
                "raw_output": safe_msg,
                "parsed": None,
                "model_used": "none",
                "fallback_triggered": True,
                "ttft_ms": 0,
            }

        parsed = self._parse_json_output(raw_output)
        total_ms = (time.time() - t_start) * 1000
        total_tokens = self._estimate_tokens(raw_output)

        self._memory.add_ai(raw_output)
        self._log_mlflow_metric("fallback_triggered", int(fallback_triggered))
        self._log_mlflow_metric("ttft_ms", round(ttft_ms, 2))
        self._log_mlflow_metric("total_generation_tokens", total_tokens)
        self._end_mlflow()

        return {
            "raw_output": raw_output,
            "parsed": parsed,
            "model_used": model_used,
            "fallback_triggered": fallback_triggered,
            "ttft_ms": round(ttft_ms, 2),
        }

    async def stream_tokens(
        self,
        query: str,
        retrieved_docs: list[dict],
        multi_ticker: bool = False,
    ) -> AsyncIterator[str]:
        """
        Async token-by-token streaming via astream.

        Args:
            query: User question string.
            retrieved_docs: List of chunk dicts from vector retrieval.
            multi_ticker: Inject the cross-company comparison directive into
                the system prompt (set when the query compares 2+ companies).

        Yields individual token strings as they are generated.
        """
        context_xml = format_context_xml(retrieved_docs)
        system_prompt = _CFO_SYSTEM_PROMPT + (
            _MULTI_TICKER_INSTRUCTION if multi_ticker else ""
        )
        self._memory.set_system(system_prompt)
        self._memory.add_human(query)
        messages = self._memory.truncate()
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1] = HumanMessage(content=f"{context_xml}\n\nQuestion: {query}")

        llm = self._build_llm(self._primary_model)
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    # --- Retry helpers ------------------------------------------------------

    @staticmethod
    def _coerce_structured_response(response: Any) -> str | None:
        """Serialize a .with_structured_output response into JSON text.

        Depending on the LangChain binding the invocation returns either a
        Pydantic ConsolidatedFinancialAnswer instance or a plain dict of the
        parsed JSON. Both are converted to a canonical JSON string so the rest
        of the pipeline (guardrail, cache) sees uniform raw output.
        """
        if hasattr(response, "model_dump_json"):
            try:
                return response.model_dump_json()
            except Exception:
                pass
        if isinstance(response, dict):
            try:
                import json as _json

                return _json.dumps(response)
            except Exception:
                return None
        text = getattr(response, "content", None)
        return text if isinstance(text, str) and text.strip() else None

    def _call_with_retry(
        self,
        model_name: str,
        messages: list[BaseMessage],
        max_retries: int = 3,
        stream: bool = False,
    ) -> tuple[str | None, str, float]:
        """
        Call the Groq API with exponential-backoff retry.

        Uses .with_structured_output(ConsolidatedFinancialAnswer) so every
        successful response is schema-bound JSON. The structured binding does
        not support token streaming, so when stream=True the raw ChatGroq
        (json_object) is invoked instead and parsed downstream.

        Returns (raw_output, model_name, ttft_ms) or (None, model_name, 0)
        on exhausted retries.
        """
        if stream:
            llm: Any = self._build_llm(model_name)
        else:
            llm = self._build_structured_llm(model_name)
        t_first = 0.0
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                try:
                    response = llm.invoke(messages)
                except OutputParserException as ope:
                    logger.warning(
                        "Structured output validation failed (%s): %s — falling back to raw JSON",
                        model_name,
                        ope,
                    )
                    response = self._build_llm(model_name).invoke(messages)
                t_first = (time.time() - t0) * 1000

                content = self._coerce_structured_response(response)
                if content and content.strip():
                    return content, model_name, t_first

                last_error = ValueError("Empty response from model")
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Groq call attempt %d/%d failed (%s): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    model_name,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error(
            "All %d retries exhausted for model %s: %s",
            max_retries,
            model_name,
            last_error,
        )
        return None, model_name, 0.0

    async def _acall_with_retry(
        self,
        model_name: str,
        messages: list[BaseMessage],
        max_retries: int = 3,
    ) -> tuple[str | None, str, float]:
        """Async variant of _call_with_retry (structured binding, no streaming)."""
        import asyncio

        llm: Any = self._build_structured_llm(model_name)
        t_first = 0.0
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                try:
                    response = await llm.ainvoke(messages)
                except OutputParserException as ope:
                    logger.warning(
                        "Async structured output validation failed (%s): %s — falling back to raw JSON",
                        model_name,
                        ope,
                    )
                    response = await self._build_llm(model_name).ainvoke(messages)
                t_first = (time.time() - t0) * 1000

                content = self._coerce_structured_response(response)
                if content and content.strip():
                    return content, model_name, t_first

                last_error = ValueError("Empty response from model")
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Async Groq attempt %d/%d failed (%s): %s — retry in %ds",
                    attempt + 1,
                    max_retries,
                    model_name,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        return None, model_name, 0.0

    # --- Output parsing -----------------------------------------------------

    def _parse_json_output(self, raw: str) -> "ConsolidatedFinancialAnswer | None":
        """
        Extract and validate JSON from the LLM's raw text output.

        Handles cases where the model wraps JSON in markdown code fences
        or thinking tags. Falls back to wrapping raw text into a default
        ConsolidatedFinancialAnswer when JSON parsing fails entirely.
        """
        from pydantic import ValidationError
        from schemas import ConsolidatedFinancialAnswer

        text = raw.strip()

        # Strip thinking tags (Qwen / DeepSeek style)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = text.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        def _validate(data) -> "ConsolidatedFinancialAnswer | None":
            # Recursively unwrap {"content": "<json>"} wrappers
            _data = data
            for _ in range(3):
                if isinstance(_data, dict) and isinstance(_data.get("content"), str):
                    try:
                        _data = json.loads(_data["content"])
                    except Exception:
                        break
            # Fill defaults for empty required fields
            if isinstance(_data, dict):
                if not _data.get("extracted_raw_data"):
                    _data["extracted_raw_data"] = _data.get("answer", "N/A")
                if not _data.get("sources"):
                    _data["sources"] = ["unknown"]
            try:
                return ConsolidatedFinancialAnswer.model_validate(_data)
            except (json.JSONDecodeError, ValidationError):
                pass
            except Exception:
                pass
            return None

        # Try direct parse first
        try:
            data = json.loads(text)
            parsed = _validate(data)
            if parsed is not None:
                return parsed
        except (json.JSONDecodeError, ValidationError):
            pass
        except Exception:
            pass

        # Try to find a JSON object in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                parsed = _validate(data)
                if parsed is not None:
                    return parsed
            except (json.JSONDecodeError, ValidationError):
                pass
            except Exception:
                pass

        logger.warning("Failed to parse JSON from LLM output (length=%d) — using raw-text fallback", len(raw))

        # Fallback: try to extract answer text from partial/escaped JSON
        answer_text = None
        # Try to find "answer" key in a partial JSON string
        ans_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if ans_match:
            try:
                answer_text = ans_match.group(1).encode().decode("unicode_escape")
            except Exception:
                answer_text = ans_match.group(1).replace("\\n", " ").replace('\"', '"')
        if not answer_text:
            # Try to find "content" key wrapping inner JSON with "answer"
            content_match = re.search(r'"content"\s*:\s*"(.*)"', raw, re.DOTALL)
            if content_match:
                inner = content_match.group(1).replace("\\n", "\n").replace('\"', '"')
                inner_ans = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', inner)
                if inner_ans:
                    try:
                        answer_text = inner_ans.group(1).encode().decode("unicode_escape")
                    except Exception:
                        answer_text = inner_ans.group(1).replace("\\n", " ").replace('\"', '"')
        if not answer_text:
            answer_text = raw.strip()
        if not answer_text:
            answer_text = "The requested financial information is not available in the provided reports."

        try:
            return ConsolidatedFinancialAnswer(
                internal_thought="1. Fallback: raw LLM output could not be parsed as structured JSON.",
                extracted_raw_data="No structured data was extracted from the LLM response.",
                answer=answer_text,
                sources=["unknown"],
            )
        except ValidationError:
            logger.error("Fallback ConsolidatedFinancialAnswer also failed validation")
            return None

    # --- Utility ------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (4 chars per token for English)."""
        return max(1, len(text) // 4)

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self._memory.clear()
