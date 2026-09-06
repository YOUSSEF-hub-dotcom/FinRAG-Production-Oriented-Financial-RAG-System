"""
Module 4 -- Retrieval Stage (Hybrid Dense + BM25 Search + Post-Retrieval Pipeline).

Public API:
  - hybrid_search.HybridSearchEngine      (sync search() + awaitable asearch())
  - reranker.CrossEncoderReranker         (BAAI/bge-reranker-large cross-encoder)
  - reranker.rerank_chunks                (functional one-shot re-ranking)
  - table_shield.TableShield              (Contextual Shredding & Table Shield)
  - table_shield.shield_chunks            (functional one-shot table shielding)
  - cylinder_reorder.cylinder_reorder     (Lost-in-the-Middle staggered reorder)
  - cylinder_reorder.CylinderReorderer    (callable reorderer class)
  - post_retrieval.PostRetrievalPipeline  (rerank -> shield -> cylinder entry)
  - post_retrieval.run_post_retrieval     (functional one-shot post-retrieval)
"""

from hybrid_search import HybridSearchEngine
from reranker import CrossEncoderReranker, rerank_chunks
from table_shield import TableShield, shield_chunks
from cylinder_reorder import cylinder_reorder, CylinderReorderer
from post_retrieval import PostRetrievalPipeline, run_post_retrieval

__all__ = [
    "HybridSearchEngine",
    "CrossEncoderReranker",
    "rerank_chunks",
    "TableShield",
    "shield_chunks",
    "cylinder_reorder",
    "CylinderReorderer",
    "PostRetrievalPipeline",
    "run_post_retrieval",
]