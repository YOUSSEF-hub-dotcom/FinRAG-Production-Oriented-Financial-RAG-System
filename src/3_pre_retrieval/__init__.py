"""
Module 3 -- Pre-Retrieval Stage (Intent Routing & Conditional Expansion).

Public API:
  - pre_retrieval_schemas.IntentAnalysisOutput / PreRetrievalResult /
    ExpansionResult
  - intent_router.build_intent_router / SECURITY_VIOLATION_RESPONSE
  - query_expansion.QueryExpander
  - orchestrator.PreRetrievalOrchestrator
"""

from intent_router import (
    SECURITY_VIOLATION_RESPONSE,
    BaseIntentRouter,
    IntentRouterError,
    build_intent_router,
)
from orchestrator import PreRetrievalOrchestrator
from pre_retrieval_schemas import ExpansionResult, IntentAnalysisOutput, PreRetrievalResult
from query_expansion import QueryExpander

__all__ = [
    "SECURITY_VIOLATION_RESPONSE",
    "BaseIntentRouter",
    "IntentRouterError",
    "PreRetrievalOrchestrator",
    "QueryExpander",
    "ExpansionResult",
    "IntentAnalysisOutput",
    "PreRetrievalResult",
    "build_intent_router",
]
