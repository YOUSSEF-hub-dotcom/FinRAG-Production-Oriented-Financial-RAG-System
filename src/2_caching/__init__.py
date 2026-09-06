"""
Module 2: Semantic Query Caching Layer.

Components:
  - redis_client.py   — centralized Redis connection manager (graceful degradation)
  - semantic_cache.py — two-tier semantic cache (fast embedding screening +
                        cross-encoder gray zone) with ticker/fiscal_year
                        namespacing and SEC-static TTL policy.

Note: `2_caching` starts with a digit, so it is NOT a valid Python package name.
Like the other numbered modules, its files are imported flat (the directory is
added to sys.path by the app/tests/pipeline bootstrap).
"""
