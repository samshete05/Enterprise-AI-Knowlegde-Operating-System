from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8001"))
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")

# Faster multilingual embedding model for better startup/indexing speed.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-small",
)

# Keep a multilingual reranker with a good speed/quality tradeoff.
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

# Local QA stays multilingual, but remains configurable if you want a smaller model later.
LOCAL_QA_MODEL_NAME = os.getenv(
    "LOCAL_QA_MODEL_NAME",
    "alon-albalak/xlm-roberta-base-xquad",
)

# Smaller remote fallback model to reduce latency when fallback is enabled.
HF_REMOTE_LLM_MODEL = os.getenv(
    "HF_REMOTE_LLM_MODEL",
    "google/flan-t5-base",
)

HF_INFERENCE_API_TOKEN = os.getenv("HF_INFERENCE_API_TOKEN", "")

# Disable remote fallback by default because the network roundtrip is the slowest path.
ENABLE_REMOTE_FALLBACK = os.getenv("ENABLE_REMOTE_FALLBACK", "false").lower() == "true"

# Slightly larger chunks reduce total chunk count and indexing time.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
