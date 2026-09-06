"""
Pytest session configuration for the Financial RAG API tests.

Disables rate limiting by default so the existing/baseline endpoint tests
(which issue many unauthenticated ``/api/v1/chat`` calls) are not throttled by
the Redis-backed slowapi counters. The advanced rate-limit tests
(``test_api_advanced.py``) explicitly re-enable the limiter for their scope.
"""

import os

# "false" unless an explicit override is provided in the environment.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
# Disable Super Admin bootstrap on every TestClient startup (the auth tests
# exercise seeding explicitly with an injected fake collection instead).
os.environ.setdefault("SEED_SUPERADMIN", "false")
# JWT access tokens must outlive the full test suite (~10 min).
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "120")
