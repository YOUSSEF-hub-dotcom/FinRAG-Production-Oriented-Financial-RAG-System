"""
Pre-Retrieval Stage (Module 3) -- Orchestrator.

Composes the intent router and the conditional query expander into the single
pre-retrieval decision the pipeline consumes: a `PreRetrievalResult` holding
the ready-to-search query list, the Qdrant pre-filtering metadata dictionary,
and the routing outcome.

Execution routes implemented here:
  Route 1  (is_safe == False)  -> halt: empty queries, empty filter.
  Route 2  (action == GENERAL) -> bypass: empty queries (pipeline sends the
                                  query straight to the final LLM).
  Route 3  (action == REWRITE) -> standalone query (+ optional expansion
                                  variants) plus the metadata pre-filter.
"""

from typing import Optional

from config.logging_config import get_logger
from config.settings import INTENT_MODEL, INTENT_PROVIDER
from intent_router import (
    BaseIntentRouter,
    IntentRouterError,
    RuleBasedIntentRouter,
    build_intent_router,
)
from query_expansion import QueryExpander
from pre_retrieval_schemas import PreRetrievalResult

logger = get_logger("pre_retrieval.orchestrator")


class PreRetrievalOrchestrator:
    """
    Single entry point for the pre-retrieval stage.

    Args:
        router: Intent router (default: built from settings via factory).
        embed_fn: Batch embedder (list[str] -> list[list[float]]) used by the
            expander for the single GPU embedding pass.
        expander: Query expander (default: rule-based QueryExpander).
    """

    def __init__(
        self,
        router: Optional[BaseIntentRouter] = None,
        embed_fn=None,
        expander: Optional[QueryExpander] = None,
        provider: str = INTENT_PROVIDER,
        model_name: str = INTENT_MODEL,
    ):
        if router is None:
            try:
                router = build_intent_router(provider=provider, model_name=model_name)
            except IntentRouterError as exc:
                logger.warning(
                    "Intent router factory failed (%s) -- using rule-based router", exc
                )
                router = RuleBasedIntentRouter()
        self._router = router
        self._expander = expander or QueryExpander()
        self._embed_fn = embed_fn
        logger.info(
            "PreRetrievalOrchestrator ready: router=%s expander=%s embed_fn=%s",
            type(router).__name__,
            type(self._expander).__name__,
            "yes" if embed_fn is not None else "no",
        )

    @property
    def router(self) -> BaseIntentRouter:
        return self._router

    def process(
        self,
        query: str,
        history: Optional[list[dict[str, str]]] = None,
    ) -> PreRetrievalResult:
        """
        Run the full pre-retrieval decision for a user query.

        Args:
            query: Current user query.
            history: Chat history as a list of {"role", "content"} dicts
                (most recent last), used for coreference resolution.

        Returns:
            PreRetrievalResult ready for the pipeline.
        """
        intent = self._router.route(query=query, history=history)
        logger.info(
            "Intent routed: is_safe=%s action=%s ticker=%s year=%s section=%s "
            "need_expansion=%s",
            intent.is_safe,
            intent.action,
            intent.ticker,
            intent.fiscal_year,
            intent.section,
            intent.need_expansion,
        )

        # Route 1: security violation -- halt without retrieval.
        if not intent.is_safe:
            return PreRetrievalResult(
                queries=[],
                metadata_filter={},
                is_safe=False,
                action=intent.action,
                standalone_query=intent.standalone_query,
            )

        # Route 2: general chitchat -- bypass vector DBs and RAG.
        if intent.action == "GENERAL":
            return PreRetrievalResult(
                queries=[],
                metadata_filter={},
                is_safe=True,
                action="GENERAL",
                standalone_query=intent.standalone_query,
            )

        # Route 3: financial rewrite -- expand + metadata pre-filter.
        expansion = self._expander.expand(
            intent.standalone_query,
            need_expansion=intent.need_expansion,
            embed_fn=self._embed_fn,
        )

        metadata_filter: dict = {}
        if intent.tickers and len(intent.tickers) > 1:
            metadata_filter["tickers"] = intent.tickers
            metadata_filter["ticker"] = intent.tickers[0]
        elif intent.ticker:
            metadata_filter["ticker"] = intent.ticker
        if intent.fiscal_year is not None:
            metadata_filter["fiscal_year"] = intent.fiscal_year
        if intent.section:
            metadata_filter["section"] = intent.section

        return PreRetrievalResult(
            queries=expansion.queries,
            metadata_filter=metadata_filter,
            is_safe=True,
            action="REWRITE",
            standalone_query=intent.standalone_query,
            embeddings=expansion.embeddings,
        )
