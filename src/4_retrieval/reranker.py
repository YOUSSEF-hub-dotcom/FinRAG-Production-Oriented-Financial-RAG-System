"""
Module 4 -- Retrieval Stage: Cross-Encoder Re-ranking (Post-Retrieval Part 1).

Re-ranks the top-40 chunks from Hybrid Search using a cross-encoder model to
improve precision by evaluating full query-chunk cross-attention.

Model: BAAI/bge-reranker-large (via sentence-transformers CrossEncoder).
Device: Auto-detect CUDA, fallback to CPU.
Dtype:  Configurable via RERANKER_DTYPE ("float32" legacy / "float16" CUDA
        optimized). On CPU the model always runs in float32. The requested
        dtype is applied to the actual model parameters (verified at load).
Output: Top 8 chunks with `rerank_score` attached.
"""

import os
import sys
import threading
from typing import Any

from config.logging_config import get_logger
from config.settings import CACHE_RERANKER_MODEL, RERANKER_DTYPE

logger = get_logger("retrieval.reranker")

# Force CUDA/CPU selection before torch import for predictable device placement.
# Respect explicit env var; otherwise auto-detect.
_DEVICE_OVERRIDE = os.getenv("RERANKER_DEVICE", "").strip().lower()
if _DEVICE_OVERRIDE in ("cuda", "cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if _DEVICE_OVERRIDE == "cuda" else ""
    _AUTO_DEVICE = _DEVICE_OVERRIDE
else:
    # Auto-detect: try CUDA, fall back to CPU
    try:
        import torch
        _AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _AUTO_DEVICE = "cpu"

# Default top-n after re-ranking (can be overridden per call)
DEFAULT_TOP_N = 8


class CrossEncoderReranker:
    """
    Cross-encoder re-ranker using BAAI/bge-reranker-large.

    Evaluates (query, chunk_text) pairs with full cross-attention and returns
    the top-n chunks sorted by relevance score, attaching `rerank_score` to each.

    Args:
        model_name: HuggingFace model identifier (default from settings).
        device: "cuda" or "cpu". Defaults to auto-detect.
        top_n: Number of chunks to return after re-ranking (default 8).
        batch_size: Batch size for cross-encoder inference.
        dtype: Inference precision: "float16" (CUDA optimized) or "float32"
            (legacy/rollback). Defaults to the RERANKER_DTYPE setting. On CPU
            the model always runs in float32 regardless of the requested dtype.
    """

    def __init__(
        self,
        model_name: str = CACHE_RERANKER_MODEL,
        device: str | None = None,
        top_n: int = DEFAULT_TOP_N,
        batch_size: int = 32,
        dtype: str | None = RERANKER_DTYPE,
    ):
        self._model_name = model_name
        self._device = device or _AUTO_DEVICE
        self._top_n = max(1, int(top_n))
        self._batch_size = max(1, int(batch_size))
        self._requested_dtype = (dtype or RERANKER_DTYPE).strip().lower()
        if self._requested_dtype not in ("float16", "float32"):
            raise ValueError(
                "Unsupported reranker dtype %r: expected 'float16' or 'float32'",
                self._requested_dtype,
            )
        # Effective dtype/resolution, populated at load time (CPU may fall back).
        self._effective_dtype: str | None = None
        self._dtype_fallback_reason: str | None = None
        self._model: Any = None
        self._initialised = False
        # Guards the shared CUDA CrossEncoder so concurrent threads never issue
        # simultaneous inference on the same GPU model.
        self._infer_lock = threading.Lock()

        logger.info(
            "CrossEncoderReranker configured: model=%s device=%s top_n=%d batch_size=%d dtype=%s",
            self._model_name,
            self._device,
            self._top_n,
            self._batch_size,
            self._requested_dtype,
        )

    def _lazy_init(self) -> None:
        """Load the CrossEncoder model on first use."""
        if self._initialised:
            return
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            logger.error("sentence-transformers not available: %s", exc)
            raise RuntimeError(
                "CrossEncoderReranker requires 'sentence-transformers' package"
            ) from exc

        try:
            import torch

            is_cuda = self._device.startswith("cuda")
            if self._requested_dtype == "float16" and not is_cuda:
                self._effective_dtype = "float32"
                self._dtype_fallback_reason = "CPU fallback"
                logger.warning(
                    "Reranker dtype fallback: requested_dtype=%s effective_dtype=%s "
                    "reason=%s on %s",
                    self._requested_dtype,
                    self._effective_dtype,
                    self._dtype_fallback_reason,
                    self._device,
                )
            else:
                self._effective_dtype = self._requested_dtype

            torch_dtype = (
                torch.float16 if self._effective_dtype == "float16" else torch.float32
            )
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                model_kwargs={"torch_dtype": torch_dtype},
            )
            # Ensure every parameter (including any fp32-held component such as
            # a classification head) is actually in the requested dtype -- never
            # merely store the dtype variable while leaving weights in fp32.
            if isinstance(self._model, torch.nn.Module):
                self._model.to(torch_dtype)
                actual_dtype = next(self._model.parameters()).dtype
                mismatched = [
                    p.dtype for p in self._model.parameters() if p.dtype != torch_dtype
                ]
                if actual_dtype != torch_dtype or mismatched:
                    logger.error(
                        "Reranker dtype verification failed: requested=%s got=%s "
                        "mismatched=%d",
                        self._effective_dtype,
                        actual_dtype,
                        len(mismatched),
                    )
                    raise RuntimeError(
                        f"Reranker model is not in requested dtype {self._effective_dtype}"
                    )
                logger.info(
                    "Reranker loaded: model=%s device=%s dtype=%s",
                    self._model_name,
                    self._model.device,
                    self._effective_dtype,
                )
            else:
                # Mock/hosted model (tests): trust upstream dtype handling.
                logger.debug(
                    "Reranker dtype verification skipped (non-torch model object)"
                )
            self._initialised = True
            logger.info("CrossEncoder model loaded: %s on %s", self._model_name, self._device)
        except Exception as exc:
            logger.error("Failed to load CrossEncoder model: %s", exc)
            raise

    def warm_up(self) -> None:
        """Pre-load the shared CrossEncoder so the first user request does not
        pay the one-time model-load cost (previously ~29s inside the first
        query's TTFT).

        This reuses the exact instance held by this reranker (never a second
        model copy) and runs a single tiny no-op rerank so the GPU load path is
        also warmed. Raises on failure so startup health can report honestly
        that required warm-up did not complete.
        """
        if self._initialised:
            logger.info("Reranker warm-up: already initialised -- skipping")
            return
        logger.info(
            "Reranker warm-up: loading %s on %s ...",
            self._model_name,
            self._device,
        )
        self._lazy_init()
        try:
            self.rerank(
                "financial report warmup",
                [{"text": "placeholder chunk to warm the cross-encoder"}],
                top_n=1,
            )
        except Exception as exc:
            logger.error("Reranker warm-up inference failed: %s", exc)
            raise
        logger.info("Reranker warm-up complete: %s on %s", self._model_name, self._device)

    def predict_scores(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[float]:
        """Run the cross-encoder predict over (query, chunk.text) pairs and
        return one score per chunk, aligned to `chunks` order.

        This is the GPU-bound hot path. It performs a SINGLE predict over all
        pairs so that multi-ticker retrieval can share one batched GPU call
        (the shared CUDA CrossEncoder must not be called concurrently from
        multiple threads). On predict failure it degrades to all-zero scores,
        preserving the pipeline's graceful-degradation behaviour.
        """
        scores: list[float]
        if not chunks:
            return []

        self._lazy_init()

        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        try:
            with self._infer_lock:
                raw = self._model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            scores = [float(s) for s in raw]
        except Exception as exc:
            logger.error("CrossEncoder prediction failed: %s", exc)
            scores = [0.0 for _ in chunks]
        return scores

    def rerank_with_scores(
        self,
        chunks: list[dict],
        scores: list[float],
        top_n: int | None = None,
    ) -> list[dict]:
        """Attach precomputed cross-encoder `scores` (aligned to `chunks`) to
        copies of the chunks, sort descending by score, and return top-n.

        This lets batched multi-ticker rerank share a single GPU predict and
        then apply the standard per-ticker selection faithfully (identical
        semantics to `rerank`, minus the predict step).
        """
        if not chunks:
            return []

        n = top_n if top_n is not None else self._top_n
        n = min(max(1, int(n)), len(chunks))

        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)  # Shallow copy to avoid mutating original
            chunk_copy["rerank_score"] = float(score)
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        top_chunks = scored_chunks[:n]

        logger.info(
            "Rerank complete: %d -> %d chunks, top_score=%.4f",
            len(chunks),
            len(top_chunks),
            top_chunks[0]["rerank_score"] if top_chunks else 0.0,
        )
        return top_chunks

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Re-rank chunks by cross-encoder relevance to the query.

        Args:
            query: The search query string.
            chunks: List of chunk dicts from HybridSearchEngine (each must have
                a 'text' field). Typically 40 chunks.
            top_n: Optional override for number of top chunks to return.

        Returns:
            List of top-n chunk dicts sorted by `rerank_score` (descending),
            each with an added `rerank_score` float field.
        """
        if not chunks:
            logger.warning("Rerank called with empty chunk list")
            return []

        self._lazy_init()

        n = top_n if top_n is not None else self._top_n
        n = min(max(1, int(n)), len(chunks))

        logger.debug("Reranking %d chunks for query: %s", len(chunks), query[:80])
        scores = self.predict_scores(query, chunks)
        return self.rerank_with_scores(chunks, scores, top_n=n)

    # Convenience sync wrapper (same as rerank)
    __call__ = rerank


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = DEFAULT_TOP_N,
    model_name: str = CACHE_RERANKER_MODEL,
    device: str | None = None,
) -> list[dict]:
    """
    Functional entry point for one-shot re-ranking.

    Creates a CrossEncoderReranker instance, re-ranks, and returns top-n chunks.

    Args:
        query: Search query string.
        chunks: List of chunk dicts with 'text' field.
        top_n: Number of top chunks to return (default 8).
        model_name: Cross-encoder model name.
        device: "cuda" or "cpu" (auto if None).

    Returns:
        Top-n chunks with `rerank_score` attached.
    """
    reranker = CrossEncoderReranker(model_name=model_name, device=device, top_n=top_n)
    return reranker.rerank(query, chunks, top_n=top_n)