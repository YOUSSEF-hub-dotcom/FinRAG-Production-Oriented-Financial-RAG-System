"""
Centralized configuration for the Financial RAG system.

Loads environment variables from .env and exposes typed constants
used across all modules.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Paths ---
PROJECT_ROOT: Path = _PROJECT_ROOT
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(Path.home() / "Financial_RAG" / "data")))
LOG_DIR: Path = PROJECT_ROOT / "logs"

# --- API Keys ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

# --- MongoDB ---
# Prefer canonical keys, then accept the legacy .env names (MONGO_URI,
# MONGO_DB_NAME) before falling back to the local default.
MONGODB_URI: str = (
    os.getenv("MONGODB_URI")
    or os.getenv("MONGO_URI")
    or "mongodb://localhost:27017/financial_rag"
)
MONGODB_DB: str = os.getenv("MONGODB_DB") or os.getenv("MONGO_DB_NAME") or "financial_rag"
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "raw_chunks")

# --- Qdrant ---
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "financial_vectors")
QDRANT_PATH: str = os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant_db"))
EMBEDDING_DIM: int = 768

# --- Redis ---
# Prefer REDIS_URL; otherwise build it from the legacy REDIS_HOST/REDIS_PORT
# pair used in .env before falling back to the local default.
REDIS_URL: str = (
    os.getenv("REDIS_URL")
    or f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0"
)

# --- Authentication & Authorization (Module 7: AuthN/AuthZ + RBAC) ---
# JWT signing (HS256). Override JWT_SECRET_KEY in production with a long,
# random secret. A development fallback is provided so the system boots without
# explicit configuration, but it MUST be replaced before any real deployment.
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-insecure-jwt-secret-change-me")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER: str = os.getenv("JWT_ISSUER", "financial-rag-api")
# Short-lived access token (minutes) and long-lived refresh token (days).
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# MongoDB collection that persists user accounts (Module 7 auth).
AUTH_USERS_COLLECTION: str = os.getenv("AUTH_USERS_COLLECTION", "users")
# Redis key prefix for the JWT revocation (logout) blacklist.
AUTH_BLACKLIST_PREFIX: str = os.getenv("AUTH_BLACKLIST_PREFIX", "auth:blacklist:")
# When "false" the lifespan bootstrap of the Super Admin account is skipped
# (used by the test-suite to avoid touching MongoDB / Redis on every startup).
SEED_SUPERADMIN: bool = os.getenv("SEED_SUPERADMIN", "true").lower() != "false"

# Bootstrap Super Admin credentials (Module 7). Seeded on startup when no
# admin account exists. Override via .env in any real environment.
SUPERADMIN_EMAIL: str = os.getenv("SUPERADMIN_EMAIL", "admin@financial-rag.local")
SUPERADMIN_PASSWORD: str = os.getenv("SUPERADMIN_PASSWORD", "SuperAdmin!123")
SUPERADMIN_FULL_NAME: str = os.getenv("SUPERADMIN_FULL_NAME", "System Super Admin")

# --- Background Tasks (Module 7: API) ---
# Default queue: FastAPI BackgroundTasks (in-process, zero infra).
# Set USE_ARQ_QUEUE=true to route async jobs (e.g. document ingestion)
# through the Arq worker pool on Redis instead. See app/api/worker.py.
USE_ARQ_QUEUE: bool = os.getenv("USE_ARQ_QUEUE", "false").lower() == "true"
ARQ_MAX_JOBS: int = int(os.getenv("ARQ_MAX_JOBS", "4"))
ARQ_JOB_TIMEOUT_SECONDS: int = int(os.getenv("ARQ_JOB_TIMEOUT_SECONDS", "900"))

# --- Caching (Module 2: Semantic Query Caching) ---
CACHE_NAMESPACE: str = "sem_cache"
CACHE_TTL_SECONDS: int = 3600              # Session TTL for ad-hoc / generic queries
CACHE_TTL_STATIC_SECONDS: int = 604800     # Long TTL (7d) for canonical SEC 10-K filings
CACHE_FAST_HIT_THRESHOLD: float = 0.96     # Tier A: score >= this -> immediate hit
CACHE_FAST_MISS_THRESHOLD: float = 0.60    # Tier A: score < this -> fast miss
CACHE_RERANK_THRESHOLD: float = 0.75       # Tier B: cross-encoder score >= this -> hit
CACHE_EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
CACHE_RERANKER_MODEL: str = "BAAI/bge-reranker-large"
# Cross-encoder inference precision. "float32" = legacy/rollback behaviour;
# "float16" = optimized CUDA mode (validated: identical top-8 ranking +
# evidence, ~67% faster on this GPU, verified via FP32-vs-FP16 regression +
# full E2E E1-E5). On CPU the model always runs in float32 regardless.
# Rollback: set RERANKER_DTYPE=float32 (config or env) to restore FP32.
RERANKER_DTYPE: str = os.getenv("RERANKER_DTYPE", "float16").strip().lower()

# --- Chunking ---
# Larger token-bounded chunks (768-1024) with 20% overlap capture wider
# semantic context so multi-line narrative statements survive retrieval
# intact; the higher overlap keeps boundary-crossing claims recoverable.
CHUNK_MIN_TOKENS: int = 768
CHUNK_MAX_TOKENS: int = 1024
CHUNK_OVERLAP_RATIO: float = 0.2  # 20% overlap (top of the 10-20% range)

# --- LLM Generation Models ---
GROQ_PRIMARY_MODEL: str = os.getenv("GROQ_PRIMARY_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE: float = 0.0
LLM_SEED: int = 42
LLM_MAX_TOKENS: int = 2048
LLM_HISTORY_K: int = 3

# --- Supported tickers ---
SUPPORTED_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA"]

# --- Retrieval (Module 4: Hybrid Dense + BM25 Search) ---
ENABLE_HYBRID_RETRIEVAL: bool = True  # Production: BM25+RRF hybrid over pre-filtered corpus
HYBRID_TOP_K: int = 40                    # Final fused result count (RRF cap)
HYBRID_RRF_K: int = 60                    # RRF constant k (rank dampening)
HYBRID_DENSE_WEIGHT: float = 1.0          # RRF weight for dense (semantic) lists
HYBRID_SPARSE_WEIGHT: float = 1.0         # RRF weight for sparse (BM25) lists
HYBRID_SPARSE_CORPUS_LIMIT: int = 500     # Max chunks fetched for the pre-filtered BM25 corpus

# --- Post-Retrieval (Module 4 Part 2: Reranker -> Table Shield -> Cylinder Reorder) ---
ENABLE_POST_RETRIEVAL: bool = True  # Production: reranker -> table shield -> cylinder reorder
POST_RETRIEVAL_INPUT_CHUNKS: int = 40     # Contract: hybrid search feeds 40 chunks in
POST_RETRIEVAL_RERANK_TOP_N: int = 8      # Cross-encoder keeps top 8
# Dynamic 8-chunk cylinder (1-based Rank 1,3,5,7,8,6,4,2): best at the head,
# second-best at the tail, remainder staggered through the middle so LLM
# attention lands on the extremes where it is strongest.
POST_RETRIEVAL_CYLINDER_PATTERN: tuple[int, ...] = (0, 2, 4, 6, 7, 5, 3, 1)

# --- Module 6: Evaluation & Quality Gate Engine (LLM-as-a-Judge) ---
# Strictly distinct from generation models to eliminate self-bias
JUDGE_PRIMARY_MODEL: str = os.getenv("JUDGE_PRIMARY_MODEL", "qwen/qwen3.6-27b")
JUDGE_FALLBACK_MODEL: str = os.getenv("JUDGE_FALLBACK_MODEL", "openai/gpt-oss-120b")
# Deterministic, reproducible judge responses.
JUDGE_TEMPERATURE: float = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
JUDGE_MAX_TOKENS: int = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))
JUDGE_MAX_RETRIES: int = int(os.getenv("JUDGE_MAX_RETRIES", "6"))
JUDGE_TIMEOUT_SECONDS: float = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "120.0"))
# Max characters per context chunk sent to the judge (keeps daily TPD budget
# within Groq free-tier limits while scoring all 25 samples).
JUDGE_MAX_CONTEXT_CHARS: int = int(os.getenv("JUDGE_MAX_CONTEXT_CHARS", "800"))

# Generation Models (Preserved for Reference) -- used by synthetic testset
# generation, kept distinct from the judge tier for cost control.
SYNTH_GENERATOR_MODEL: str = os.getenv("SYNTH_GENERATOR_MODEL", "qwen-2.5-32b")
SYNTH_FALLBACK_MODEL: str = os.getenv("SYNTH_FALLBACK_MODEL", "openai/gpt-oss-20b")

# Synthetic dataset -- STRICT 25-question cap to fit Groq free-tier TPM limits.
SYNTHETIC_TEST_SIZE: int = int(os.getenv("SYNTHETIC_TEST_SIZE", "25"))
SYNTHETIC_BATCH_CONCURRENCY: int = int(os.getenv("SYNTHETIC_BATCH_CONCURRENCY", "4"))
SYNTHETIC_MAX_DOC_CHARS: int = int(os.getenv("SYNTHETIC_MAX_DOC_CHARS", "60000"))

# --- Evaluation artifacts & pipeline tracing ---
ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
TEST_DATASET_PATH: Path = ARTIFACTS_DIR / "test_dataset.csv"
EVALUATION_RESULTS_PATH: Path = ARTIFACTS_DIR / "evaluation_results.csv"
EVALUATION_SCORES_PATH: Path = ARTIFACTS_DIR / "evaluation_scores.json"
EVALUATION_TRACE_COLUMNS: tuple[str, ...] = (
    "question",
    "ground_truth",
    "contexts",
    "answer",
    "model_used",
    "pipeline_run_id",
)

# --- Quality Gate thresholds (Module 6) ---
QA_GATE_FAITHFULNESS: float = float(os.getenv("QA_GATE_FAITHFULNESS", "0.85"))
QA_GATE_ANSWER_RELEVANCE: float = float(os.getenv("QA_GATE_ANSWER_RELEVANCE", "0.80"))
QA_GATE_CONTEXT_PRECISION: float = float(os.getenv("QA_GATE_CONTEXT_PRECISION", "0.75"))
QA_GATE_CONTEXT_RECALL: float = float(os.getenv("QA_GATE_CONTEXT_RECALL", "0.80"))

# --- MLflow evaluation experiment & model registry ---
MLFLOW_EVAL_EXPERIMENT: str = os.getenv("MLFLOW_EVAL_EXPERIMENT", "Financial_RAG_Evaluation")
MLFLOW_EVAL_MODEL_NAME: str = os.getenv("MLFLOW_EVAL_MODEL_NAME", "financial_rag_pipeline")
MLFLOW_EVAL_REGISTRY_URI: str = os.getenv("MLFLOW_EVAL_REGISTRY_URI", "sqlite:////home/youssef/Financial_RAG/mlflow.db")

# --- MLflow Production Model Loading ---
# When enabled, the backend fetches the active Production pipeline from the
# MLflow Model Registry alias (models:/<name>@Production) instead of
# instantiating FinancialRAGPipeline directly. Set to "false" to disable
# and fall back to direct instantiation (useful for tests/offline dev).
MLFLOW_MODEL_LOADING: bool = os.getenv("MLFLOW_MODEL_LOADING", "true").lower() == "true"
MLFLOW_PRODUCTION_ALIAS: str = os.getenv("MLFLOW_PRODUCTION_ALIAS", "Production")

# The 4 core Ragas metrics evaluated by the judge tier.
CORE_RAGAS_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
)

# --- Pre-Retrieval (Module 3: Intent Routing & Conditional Expansion) ---
ENABLE_PRE_RETRIEVAL: bool = False  # Opt-in; API/legacy tests keep defaults
INTENT_PROVIDER: str = os.getenv("INTENT_PROVIDER", "groq")
INTENT_MODEL: str = os.getenv("INTENT_MODEL", "qwen2.5-3b-instruct")
INTENT_TEMPERATURE: float = float(os.getenv("INTENT_TEMPERATURE", "0.0"))
INTENT_MAX_TOKENS: int = int(os.getenv("INTENT_MAX_TOKENS", "512"))
INTENT_SEED: int = LLM_SEED
INTENT_MAX_RETRIES: int = int(os.getenv("INTENT_MAX_RETRIES", "2"))
INTENT_OLLAMA_BASE_URL: str = os.getenv("INTENT_OLLAMA_BASE_URL", "")
INTENT_VLLM_BASE_URL: str = os.getenv("INTENT_VLLM_BASE_URL", "")
EXPANSION_MIN_WORDS: int = int(os.getenv("EXPANSION_MIN_WORDS", "4"))
EXPANSION_MAX_VARIANTS: int = int(os.getenv("EXPANSION_MAX_VARIANTS", "4"))
