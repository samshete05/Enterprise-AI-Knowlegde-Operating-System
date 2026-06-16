from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# Force Transformers/SentenceTransformers to avoid TensorFlow imports.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import pipeline as hf_pipeline

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    ENABLE_REMOTE_FALLBACK,
    HF_INFERENCE_API_TOKEN,
    HF_REMOTE_LLM_MODEL,
    LOCAL_QA_MODEL_NAME,
    RERANKER_MODEL_NAME,
)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional runtime dependency
    BeautifulSoup = None

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional runtime dependency
    DocxDocument = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional runtime dependency
    pytesseract = None


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".docx",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".go",
    ".rs",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".css",
    ".sql",
    ".sh",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
HTML_EXTENSIONS = {".html", ".htm"}
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".go",
    ".rs",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".css",
    ".sql",
    ".sh",
}

_embedding_model: HuggingFaceEmbeddings | None = None
_reranker: CrossEncoder | None = None
_qa_pipeline: Any | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def get_qa_pipeline() -> Any | None:
    global _qa_pipeline
    if not LOCAL_QA_MODEL_NAME:
        return None
    if _qa_pipeline is None:
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model=LOCAL_QA_MODEL_NAME,
            tokenizer=LOCAL_QA_MODEL_NAME,
        )
    return _qa_pipeline


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def detect_language_hint(text: str) -> str:
    if not text.strip():
        return "unknown"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    return "en"


def validate_supported_file(file_path: Path) -> None:
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Supported types: {allowed}")


def extract_pdf_documents(file_path: Path) -> list[Document]:
    documents = PyPDFLoader(str(file_path)).load()
    extracted: list[Document] = []

    for index, doc in enumerate(documents, start=1):
        text = doc.page_content.strip()
        if not text:
            continue
        extracted.append(
            Document(
                page_content=text,
                metadata={
                    "section": index,
                    "section_label": f"page {index}",
                    "extraction_method": "pdf_text",
                },
            )
        )

    return extracted


def extract_image_documents(file_path: Path) -> list[Document]:
    if Image is None:
        raise ValueError(
            "Image support requires Pillow. Install dependencies from requirements.txt to process screenshots and images."
        )
    if pytesseract is None:
        raise ValueError(
            "Image OCR requires pytesseract. Install pytesseract and the Tesseract OCR binary to process screenshots and images."
        )

    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    text = text.strip()
    if not text:
        raise ValueError("No readable text was detected in the uploaded image.")

    return [
        Document(
            page_content=text,
            metadata={
                "section": 1,
                "section_label": "image ocr",
                "extraction_method": "ocr",
            },
        )
    ]


def extract_docx_documents(file_path: Path) -> list[Document]:
    if DocxDocument is None:
        raise ValueError(
            "DOCX support requires python-docx. Install dependencies from requirements.txt to process DOCX files."
        )
    document = DocxDocument(str(file_path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    if not paragraphs:
        raise ValueError("No readable text was found in the uploaded DOCX file.")

    return [
        Document(
            page_content="\n".join(paragraphs),
            metadata={
                "section": 1,
                "section_label": "document body",
                "extraction_method": "docx_text",
            },
        )
    ]


def extract_html_documents(file_path: Path) -> list[Document]:
    if BeautifulSoup is None:
        raise ValueError(
            "HTML support requires beautifulsoup4. Install dependencies from requirements.txt to process HTML files."
        )
    html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No readable text was found in the uploaded HTML file.")

    return [
        Document(
            page_content="\n".join(lines),
            metadata={
                "section": 1,
                "section_label": "html body",
                "extraction_method": "html_text",
            },
        )
    ]


def extract_text_documents(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("The uploaded file is empty after text extraction.")

    return [
        Document(
            page_content=text,
            metadata={
                "section": 1,
                "section_label": "text body",
                "extraction_method": "plain_text",
            },
        )
    ]


def extract_documents(file_path: Path) -> list[Document]:
    validate_supported_file(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_documents(file_path)
    if suffix in IMAGE_EXTENSIONS:
        return extract_image_documents(file_path)
    if suffix == ".docx":
        return extract_docx_documents(file_path)
    if suffix in HTML_EXTENSIONS:
        return extract_html_documents(file_path)
    if suffix in TEXT_EXTENSIONS:
        return extract_text_documents(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def build_chunks(file_path: Path) -> list[Document]:
    # in building the chunks according to doc extension the chunks is done
    documents = extract_documents(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["document_name"] = file_path.name
        chunk.metadata["document_type"] = file_path.suffix.lower()
        chunk.metadata["language_hint"] = detect_language_hint(chunk.page_content)

    return chunks


def build_bm25(chunks: list[Document]) -> BM25Okapi:
    corpus = [tokenize(doc.page_content) for doc in chunks]
    return BM25Okapi(corpus)


def bm25_search(
    query: str,
    chunks: list[Document],
    bm25: BM25Okapi,
    top_k: int = 5,
) -> list[Document]:
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[idx] for idx in top_indices]


def hybrid_search(
    query: str,
    chunks: list[Document],
    vectorstore: Chroma,
    bm25: BM25Okapi,
    top_k: int = 5,
) -> list[Document]:
    semantic_docs = vectorstore.similarity_search(query, k=top_k)
    bm25_docs = bm25_search(query, chunks, bm25, top_k=top_k)

    unique_docs: list[Document] = []
    seen_chunk_ids: set[Any] = set()

    for doc in semantic_docs + bm25_docs:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        unique_docs.append(doc)

    return unique_docs[:top_k]


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 4,
) -> tuple[list[Document], list[float]]:
    if not documents:
        return [], []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    ranked_docs = [documents[idx] for idx in ranked_indices]
    ranked_scores = [float(scores[idx]) for idx in ranked_indices]
    return ranked_docs, ranked_scores


def expand_query(query: str) -> list[str]:
    expansions = {
        "claim": ["claim settlement", "insurance reimbursement", "claim process"],
        "policy": ["insurance coverage", "coverage rules", "policy document"],
        "exclusions": ["not covered", "limitations", "exceptions"],
    }

    expanded_queries = [query]
    lowered_query = query.lower()

    for keyword, variants in expansions.items():
        if keyword in lowered_query:
            expanded_queries.extend(variants)

    return expanded_queries


def normalize_rerank_score(score: float) -> float:
    return 1 / (1 + math.exp(-score))


def lexical_overlap_ratio(query: str, documents: list[Document]) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens or not documents:
        return 0.0

    context_tokens = set()
    for doc in documents:
        context_tokens.update(tokenize(doc.page_content))

    return len(query_tokens & context_tokens) / max(len(query_tokens), 1)


def calculate_confidence(
    query: str,
    documents: list[Document],
    rerank_scores: list[float],
    qa_score: float,
    used_external_fallback: bool,
) -> dict[str, Any]:
    if not documents:
        return {
            "score": 0.1 if used_external_fallback else 0.0,
            "label": "low",
            "reason": "No grounded evidence was found in the uploaded file.",
        }

    semantic_strength = max((normalize_rerank_score(score) for score in rerank_scores), default=0.0)
    lexical_strength = lexical_overlap_ratio(query, documents)
    evidence_density = min(len(documents) / 3, 1.0)
    answer_strength = max(min(qa_score, 1.0), 0.0)

    confidence = (
        0.4 * semantic_strength
        + 0.2 * lexical_strength
        + 0.25 * evidence_density
        + 0.15 * answer_strength
    )

    if used_external_fallback:
        confidence *= 0.65

    label = "high" if confidence >= 0.75 else "medium" if confidence >= 0.45 else "low"
    reason = (
        "Confidence is based on reranking strength, lexical overlap, source coverage, and answer extraction quality."
    )

    return {
        "score": round(confidence, 3),
        "label": label,
        "reason": reason,
    }


def answer_from_documents(query: str, documents: list[Document]) -> tuple[str, float]:
    if not documents:
        return "", 0.0

    qa = get_qa_pipeline()
    if qa is None:
        return "", 0.0

    best_answer = ""
    best_score = 0.0

    for doc in documents:
        try:
            result = qa(question=query, context=doc.page_content)
        except Exception:
            continue

        answer = str(result.get("answer", "")).strip()
        score = float(result.get("score", 0.0))
        if not answer or answer.lower() in {"", "[cls]"}:
            continue
        if score > best_score:
            best_answer = answer
            best_score = score

    return best_answer, best_score


def build_grounded_answer(query: str, documents: list[Document], extracted_answer: str) -> str:
    if not documents:
        return "I could not find relevant information in the uploaded file for that question."

    best_excerpt = documents[0].page_content.strip()
    short_excerpt = best_excerpt[:700].strip()

    if extracted_answer:
        return (
            f"Answer: {extracted_answer}\n\n"
            f"Grounding: The answer was extracted from the uploaded file based on the most relevant retrieved chunk.\n\n"
            f"Top evidence excerpt:\n{short_excerpt}"
        )

    evidence = "\n\n".join(
        doc.page_content.strip()[:500]
        for doc in documents[:3]
        if doc.page_content.strip()
    )
    return (
        f"I found relevant evidence for the question '{query}', but I could not extract a precise short answer with high confidence.\n\n"
        f"Most relevant evidence from the uploaded file:\n{evidence}"
    )


def call_remote_llm(query: str, documents: list[Document]) -> str:
    if not ENABLE_REMOTE_FALLBACK or not HF_INFERENCE_API_TOKEN:
        return ""

    evidence = "\n\n".join(doc.page_content.strip()[:800] for doc in documents[:3] if doc.page_content.strip())
    prompt = (
        "You are a multilingual assistant.\n"
        "Answer in the same language as the user's question.\n"
        "If the evidence is sufficient, prioritize it.\n"
        "If the evidence is insufficient, answer from general knowledge and clearly mention that the answer used external reasoning.\n\n"
        f"Question: {query}\n\n"
        f"Evidence from uploaded file:\n{evidence or 'No reliable evidence found in the uploaded file.'}"
    )

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_REMOTE_LLM_MODEL}",
        headers={
            "Authorization": f"Bearer {HF_INFERENCE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.2,
                "return_full_text": False,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list) and payload:
        return str(payload[0].get("generated_text", "")).strip()
    if isinstance(payload, dict):
        return str(payload.get("generated_text", "")).strip()
    return ""


def build_workflow_steps(
    file_path: Path,
    documents: list[Document],
    rerank_scores: list[float],
    qa_score: float,
    used_external_fallback: bool,
) -> list[str]:
    top_score = max((normalize_rerank_score(score) for score in rerank_scores), default=0.0)
    return [
        f"Accepted `{file_path.suffix.lower()}` and selected the matching extraction strategy.",
        f"Extracted {len(documents)} searchable chunks from `{file_path.name}`.",
        f"Built multilingual embeddings with `{EMBEDDING_MODEL_NAME}` and hybrid lexical search.",
        f"Reranked retrieved chunks with `{RERANKER_MODEL_NAME}`. Top normalized rerank score: {top_score:.2f}.",
        f"Ran local multilingual answer extraction with `{LOCAL_QA_MODEL_NAME}`. Best QA score: {qa_score:.2f}.",
        "Used external Hugging Face fallback because grounded evidence was weak."
        if used_external_fallback
        else "Answered from the uploaded file because grounded evidence was strong enough.",
    ]


def prepare_pipeline(file_path: Path) -> dict[str, Any]:
    validate_supported_file(file_path)
    chunks = build_chunks(file_path)
    if not chunks:
        raise ValueError("No readable text chunks were found in the uploaded file.")

    vectorstore_dir = Path(tempfile.mkdtemp(prefix="enterprise_multiformat_chroma_"))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=str(vectorstore_dir),
    )

    combined_text = "\n".join(chunk.page_content for chunk in chunks[:10])

    return {
        "document_name": file_path.name,
        "document_path": str(file_path),
        "document_type": file_path.suffix.lower(),
        "language_hint": detect_language_hint(combined_text),
        "chunks": chunks,
        "bm25": build_bm25(chunks),
        "vectorstore": vectorstore,
        "vectorstore_dir": str(vectorstore_dir),
    }


def run_query(pipeline: dict[str, Any], query: str) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Please enter a question before submitting.")

    aggregated_docs: list[Document] = []
    seen_chunk_ids: set[Any] = set()

    for variant in expand_query(clean_query):
        matches = hybrid_search(
            variant,
            pipeline["chunks"],
            pipeline["vectorstore"],
            pipeline["bm25"],
            top_k=5,
        )
        for doc in matches:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            aggregated_docs.append(doc)

    top_documents, rerank_scores = rerank_documents(clean_query, aggregated_docs, top_k=4)
    extracted_answer, qa_score = answer_from_documents(clean_query, top_documents)

    used_external_fallback = False
    answer = build_grounded_answer(clean_query, top_documents, extracted_answer)

    confidence = calculate_confidence(
        clean_query,
        top_documents,
        rerank_scores,
        qa_score,
        used_external_fallback=False,
    )

    if confidence["label"] == "low":
        remote_answer = call_remote_llm(clean_query, top_documents)
        if remote_answer:
            used_external_fallback = True
            answer = remote_answer
            confidence = calculate_confidence(
                clean_query,
                top_documents,
                rerank_scores,
                qa_score,
                used_external_fallback=True,
            )

    return {
        "query": clean_query,
        "answer": answer,
        "sources": [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "document_name": doc.metadata.get("document_name", pipeline["document_name"]),
                "preview": doc.page_content.strip()[:700],
                "section_label": doc.metadata.get("section_label", "section"),
                "language_hint": doc.metadata.get("language_hint", pipeline["language_hint"]),
            }
            for doc in top_documents
        ],
        "confidence": confidence,
        "used_external_fallback": used_external_fallback,
        "workflow_steps": build_workflow_steps(
            Path(pipeline["document_path"]),
            top_documents,
            rerank_scores,
            qa_score,
            used_external_fallback,
        ),
    }
