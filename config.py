from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8001"))
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")

# Slightly larger multilingual embedding model for stronger retrieval accuracy.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-large",
)

# Stronger multilingual reranker for cross-language and long-document relevance.
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3",
)

# Larger multilingual extractive QA model for stronger span selection.
LOCAL_QA_MODEL_NAME = os.getenv(
    "LOCAL_QA_MODEL_NAME",
    "deepset/xlm-roberta-large-squad2",
)

# Remote model used for third-party fallback answers and translation requests.
HF_REMOTE_LLM_MODEL = os.getenv(
    "HF_REMOTE_LLM_MODEL",
    "Qwen/Qwen2.5-7B-Instruct",
)

HF_INFERENCE_API_TOKEN = os.getenv("HF_INFERENCE_API_TOKEN", "")

# Enable remote fallback by default when a token is configured.
ENABLE_REMOTE_FALLBACK = os.getenv("ENABLE_REMOTE_FALLBACK", "true").lower() == "true"
ENABLE_REMOTE_TRANSLATION = os.getenv("ENABLE_REMOTE_TRANSLATION", "true").lower() == "true"
STRICT_DOCUMENT_GROUNDED = os.getenv("STRICT_DOCUMENT_GROUNDED", "true").lower() == "true"

# Accuracy-focused retrieval defaults.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
OCR_ENGINE_PREFERENCE = os.getenv(
    "OCR_ENGINE_PREFERENCE",
    "easyocr,paddleocr,pytesseract",
)
EASYOCR_LANGUAGES = os.getenv(
    "EASYOCR_LANGUAGES",
    "en,hi,ar,ch_sim,ja,ko",
)
PADDLEOCR_LANGUAGE = os.getenv("PADDLEOCR_LANGUAGE", "en")
OCR_MIN_QUALITY_SCORE = float(os.getenv("OCR_MIN_QUALITY_SCORE", "0.25"))
SCANNED_PDF_TEXT_THRESHOLD = int(os.getenv("SCANNED_PDF_TEXT_THRESHOLD", "120"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.56"))
OUT_OF_SCOPE_RELEVANCE_THRESHOLD = float(os.getenv("OUT_OF_SCOPE_RELEVANCE_THRESHOLD", "0.66"))
MIN_LEXICAL_OVERLAP = float(os.getenv("MIN_LEXICAL_OVERLAP", "0.08"))
OUT_OF_SCOPE_LEXICAL_THRESHOLD = float(os.getenv("OUT_OF_SCOPE_LEXICAL_THRESHOLD", "0.12"))
MIN_QA_SCORE = float(os.getenv("MIN_QA_SCORE", "0.20"))
OUT_OF_SCOPE_QA_THRESHOLD = float(os.getenv("OUT_OF_SCOPE_QA_THRESHOLD", "0.35"))
MIN_ANSWER_GROUNDING_SCORE = float(os.getenv("MIN_ANSWER_GROUNDING_SCORE", "0.12"))
MIN_ANSWER_EVIDENCE_SIMILARITY = float(os.getenv("MIN_ANSWER_EVIDENCE_SIMILARITY", "0.62"))
MIN_QUERY_ANSWER_SIMILARITY = float(os.getenv("MIN_QUERY_ANSWER_SIMILARITY", "0.45"))
LOW_CONFIDENCE_REJECT_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_REJECT_THRESHOLD", "0.20"))
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.78"))
CONFIDENCE_MEDIUM_THRESHOLD = float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.50"))
