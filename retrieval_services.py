from __future__ import annotations

import math
import os
import re
import tempfile
from io import BytesIO
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
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    EASYOCR_LANGUAGES,
    ENABLE_REMOTE_FALLBACK,
    ENABLE_REMOTE_TRANSLATION,
    HF_INFERENCE_API_TOKEN,
    HF_REMOTE_LLM_MODEL,
    LOCAL_QA_MODEL_NAME,
    LOW_CONFIDENCE_REJECT_THRESHOLD,
    MIN_ANSWER_EVIDENCE_SIMILARITY,
    MIN_ANSWER_GROUNDING_SCORE,
    MIN_LEXICAL_OVERLAP,
    MIN_QA_SCORE,
    MIN_QUERY_ANSWER_SIMILARITY,
    MIN_RELEVANCE_SCORE,
    OCR_MIN_QUALITY_SCORE,
    OCR_ENGINE_PREFERENCE,
    OUT_OF_SCOPE_LEXICAL_THRESHOLD,
    OUT_OF_SCOPE_QA_THRESHOLD,
    OUT_OF_SCOPE_RELEVANCE_THRESHOLD,
    PADDLEOCR_LANGUAGE,
    RERANK_TOP_K,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K,
    SCANNED_PDF_TEXT_THRESHOLD,
    STRICT_DOCUMENT_GROUNDED,
)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional runtime dependency
    BeautifulSoup = None

try:
    import fitz
except ImportError:  # pragma: no cover - optional runtime dependency
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional runtime dependency
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional runtime dependency
    DocxDocument = None

try:
    from PIL import Image, ImageFilter, ImageOps
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageFilter = None
    ImageOps = None

try:
    import easyocr
except ImportError:  # pragma: no cover - optional runtime dependency
    easyocr = None

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional runtime dependency
    PaddleOCR = None

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
_easyocr_reader: Any | None = None
_paddleocr_reader: Any | None = None

LANGUAGE_NAME_TO_CODE = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "japanese": "ja",
    "korean": "ko",
    "marathi": "mr",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}

LANGUAGE_CODE_TO_NAME = {code: name.title() for name, code in LANGUAGE_NAME_TO_CODE.items()}


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
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=1024)
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


def get_easyocr_reader() -> Any | None:
    global _easyocr_reader
    if easyocr is None:
        return None
    if _easyocr_reader is None:
        languages = [lang.strip() for lang in EASYOCR_LANGUAGES.split(",") if lang.strip()]
        _easyocr_reader = easyocr.Reader(languages or ["en"], gpu=False)
    return _easyocr_reader


def get_paddleocr_reader() -> Any | None:
    global _paddleocr_reader
    if PaddleOCR is None:
        return None
    if _paddleocr_reader is None:
        _paddleocr_reader = PaddleOCR(use_angle_cls=True, lang=PADDLEOCR_LANGUAGE, show_log=False)
    return _paddleocr_reader


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def detect_language_hint(text: str) -> str:
    if not text.strip():
        return "unknown"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gu"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u3040-\u30FF]", text):
        return "ja"
    if re.search(r"[\uAC00-\uD7AF]", text):
        return "ko"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    return "en"


def can_use_remote_fallback() -> bool:
    return ENABLE_REMOTE_FALLBACK and bool(HF_INFERENCE_API_TOKEN)


def can_use_remote_translation() -> bool:
    return ENABLE_REMOTE_TRANSLATION and bool(HF_INFERENCE_API_TOKEN)


def language_name(code: str) -> str:
    return LANGUAGE_CODE_TO_NAME.get(code, code or "unknown")


def parse_target_language(query: str) -> str | None:
    lowered = query.lower()
    for language, code in LANGUAGE_NAME_TO_CODE.items():
        pattern = (
            rf"(?:answer|respond|reply|summarize|summary|translate|convert|give|provide)"
            rf"(?:\s+the\s+answer)?(?:\s+it)?\s+(?:in|into|to)\s+{language}\b"
        )
        if re.search(pattern, lowered):
            return code

    if re.search(r"(?:same language as (?:the )?(?:file|document)|in the document language)", lowered):
        return "document"

    return None


def strip_language_instruction(query: str) -> str:
    cleaned = query
    for language in LANGUAGE_NAME_TO_CODE:
        cleaned = re.sub(
            rf"(?i)[,.;]?\s*(?:please\s+)?(?:answer|respond|reply|summarize|summary|translate|convert|give|provide)"
            rf"(?:\s+the\s+answer)?(?:\s+it)?\s+(?:in|into|to)\s+{language}\b",
            "",
            cleaned,
        )

    cleaned = re.sub(
        r"(?i)[,.;]?\s*(?:please\s+)?(?:answer|respond|reply)\s+in\s+the\s+(?:same\s+)?language\s+as\s+(?:the\s+)?(?:file|document)\b",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;")
    return cleaned or query.strip()


def parse_query_preferences(query: str, document_language: str) -> dict[str, str | bool]:
    query_language = detect_language_hint(query)
    requested_target_language = parse_target_language(query)
    cleaned_query = strip_language_instruction(query)

    if requested_target_language == "document":
        target_language = document_language if document_language != "unknown" else query_language
    elif requested_target_language:
        target_language = requested_target_language
    else:
        target_language = query_language

    return {
        "clean_query": cleaned_query,
        "query_language": query_language,
        "target_language": target_language,
        "translation_requested": target_language != query_language or cleaned_query != query.strip(),
    }


def validate_supported_file(file_path: Path) -> None:
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Supported types: {allowed}")


def format_table_rows(rows: list[list[Any]]) -> str:
    formatted_rows: list[str] = []
    for row in rows:
        cleaned_cells = [normalize_whitespace(str(cell or "")) for cell in row]
        cleaned_cells = [cell for cell in cleaned_cells if cell]
        if cleaned_cells:
            formatted_rows.append(" | ".join(cleaned_cells))
    return "\n".join(formatted_rows)


def render_pdf_page_to_image(file_path: Path, page_index: int) -> Any | None:
    if fitz is None or Image is None:
        return None

    pdf = fitz.open(str(file_path))
    try:
        page = pdf.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return Image.open(BytesIO(pixmap.tobytes("png")))
    finally:
        pdf.close()


def extract_scanned_pdf_page(file_path: Path, page_index: int) -> tuple[str, float, str]:
    rendered_image = render_pdf_page_to_image(file_path, page_index)
    if rendered_image is None:
        return "", 0.0, "render-unavailable"

    best_text = ""
    best_score = 0.0
    best_variant = "rendered-original"

    for variant_name, variant_image in build_ocr_candidates(rendered_image):
        for engine_name in get_ocr_engine_order():
            if engine_name == "easyocr":
                text, average_confidence = run_easyocr_on_variant(variant_image)
                candidate_score = score_ocr_result(text, average_confidence)
                if candidate_score > best_score:
                    best_text = text
                    best_score = candidate_score
                    best_variant = f"{variant_name}/easyocr"
            elif engine_name == "paddleocr":
                text, average_confidence = run_paddleocr_on_variant(variant_image)
                candidate_score = score_ocr_result(text, average_confidence)
                if candidate_score > best_score:
                    best_text = text
                    best_score = candidate_score
                    best_variant = f"{variant_name}/paddleocr"
            elif engine_name == "pytesseract" and pytesseract is not None:
                for page_segmentation_mode in (4, 6, 11):
                    tesseract_config = f"--oem 3 --psm {page_segmentation_mode}"
                    text, average_confidence = run_ocr_on_variant(variant_image, tesseract_config)
                    candidate_score = score_ocr_result(text, average_confidence)
                    if candidate_score > best_score:
                        best_text = text
                        best_score = candidate_score
                        best_variant = f"{variant_name}/tesseract-psm{page_segmentation_mode}"

    return best_text, best_score, best_variant


def extract_pdf_documents(file_path: Path) -> list[Document]:
    extracted: list[Document] = []

    if pdfplumber is not None:
        with pdfplumber.open(str(file_path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                page_parts: list[str] = []
                raw_text = normalize_whitespace(page.extract_text() or "")
                if raw_text:
                    page_parts.append(raw_text)

                table_blocks: list[str] = []
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for table_index, table in enumerate(tables, start=1):
                    formatted_table = format_table_rows(table)
                    if formatted_table:
                        table_blocks.append(f"Table {table_index}:\n{formatted_table}")

                if table_blocks:
                    page_parts.append("\n\n".join(table_blocks))

                page_text = normalize_whitespace("\n\n".join(page_parts))
                extraction_method = "pdf_text_table"
                ocr_quality_score = None
                ocr_variant = None

                if len(page_text) < SCANNED_PDF_TEXT_THRESHOLD:
                    ocr_text, ocr_quality_score, ocr_variant = extract_scanned_pdf_page(file_path, index - 1)
                    if ocr_text and ocr_quality_score >= OCR_MIN_QUALITY_SCORE:
                        page_text = normalize_whitespace("\n\n".join(part for part in [page_text, ocr_text] if part))
                        extraction_method = "pdf_text_table_ocr"

                if not page_text:
                    continue

                metadata: dict[str, Any] = {
                    "section": index,
                    "section_label": f"page {index}",
                    "extraction_method": extraction_method,
                }
                if table_blocks:
                    metadata["contains_table"] = True
                if ocr_quality_score is not None:
                    metadata["ocr_quality_score"] = round(ocr_quality_score, 3)
                if ocr_variant is not None:
                    metadata["ocr_variant"] = ocr_variant

                extracted.append(Document(page_content=page_text, metadata=metadata))

        if extracted:
            return extracted

    documents = PyPDFLoader(str(file_path)).load()
    for index, doc in enumerate(documents, start=1):
        text = normalize_whitespace(doc.page_content)
        if not text:
            continue
        extracted.append(
            Document(
                page_content=text,
                metadata={
                    "section": index,
                    "section_label": f"page {index}",
                    "extraction_method": "pdf_loader_text",
                },
            )
        )

    return extracted


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def build_ocr_candidates(image: Any) -> list[tuple[str, Any]]:
    variants: list[tuple[str, Any]] = [("original", image.convert("RGB"))]
    grayscale = ImageOps.grayscale(image)
    variants.append(("grayscale", grayscale))
    variants.append(("autocontrast", ImageOps.autocontrast(grayscale)))

    threshold = ImageOps.autocontrast(grayscale).point(lambda px: 255 if px > 170 else 0)
    variants.append(("threshold", threshold))

    enlarged = ImageOps.autocontrast(grayscale).resize(
        (max(grayscale.width * 2, 1), max(grayscale.height * 2, 1))
    )
    variants.append(("enlarged", enlarged))

    sharpened = enlarged.filter(ImageFilter.SHARPEN)
    variants.append(("sharpened", sharpened))
    return variants


def run_ocr_on_variant(image: Any, tesseract_config: str) -> tuple[str, float]:
    data = pytesseract.image_to_data(
        image,
        config=tesseract_config,
        output_type=pytesseract.Output.DICT,
    )
    words: list[str] = []
    confidences: list[float] = []

    for word, confidence in zip(data.get("text", []), data.get("conf", [])):
        cleaned_word = str(word).strip()
        if not cleaned_word:
            continue
        words.append(cleaned_word)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence_value >= 0:
            confidences.append(confidence_value)

    text = normalize_whitespace(" ".join(words))
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, average_confidence


def run_easyocr_on_variant(image: Any) -> tuple[str, float]:
    reader = get_easyocr_reader()
    if reader is None:
        return "", 0.0

    results = reader.readtext(np.array(image), detail=1, paragraph=False)
    texts: list[str] = []
    confidences: list[float] = []

    for result in results:
        if len(result) < 3:
            continue
        text = normalize_whitespace(str(result[1]))
        confidence = float(result[2])
        if not text:
            continue
        texts.append(text)
        confidences.append(confidence)

    merged_text = normalize_whitespace("\n".join(texts))
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return merged_text, average_confidence


def run_paddleocr_on_variant(image: Any) -> tuple[str, float]:
    reader = get_paddleocr_reader()
    if reader is None:
        return "", 0.0

    results = reader.ocr(np.array(image), cls=True)
    texts: list[str] = []
    confidences: list[float] = []

    for block in results or []:
        for line in block or []:
            if not line or len(line) < 2:
                continue
            recognition = line[1]
            if not recognition or len(recognition) < 2:
                continue
            text = normalize_whitespace(str(recognition[0]))
            confidence = float(recognition[1])
            if not text:
                continue
            texts.append(text)
            confidences.append(confidence)

    merged_text = normalize_whitespace("\n".join(texts))
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return merged_text, average_confidence


def score_ocr_result(text: str, average_confidence: float) -> float:
    if not text:
        return 0.0

    tokens = tokenize(text)
    unique_token_ratio = len(set(tokens)) / max(len(tokens), 1)
    alnum_ratio = sum(char.isalnum() for char in text) / max(len(text), 1)
    length_score = min(len(tokens) / 40, 1.0)
    confidence_score = min(max(average_confidence / 100.0, 0.0), 1.0)

    return (
        0.45 * confidence_score
        + 0.25 * unique_token_ratio
        + 0.20 * alnum_ratio
        + 0.10 * length_score
    )


def get_ocr_engine_order() -> list[str]:
    engines = [engine.strip().lower() for engine in OCR_ENGINE_PREFERENCE.split(",") if engine.strip()]
    return engines or ["easyocr", "paddleocr", "pytesseract"]


def extract_best_ocr_text(file_path: Path) -> tuple[str, float, str]:
    image = Image.open(file_path)
    best_text = ""
    best_score = 0.0
    best_variant = "original"

    for variant_name, variant_image in build_ocr_candidates(image):
        for engine_name in get_ocr_engine_order():
            if engine_name == "easyocr":
                text, average_confidence = run_easyocr_on_variant(variant_image)
                candidate_score = score_ocr_result(text, average_confidence)
                if candidate_score > best_score:
                    best_text = text
                    best_score = candidate_score
                    best_variant = f"{variant_name}/easyocr"
            elif engine_name == "paddleocr":
                text, average_confidence = run_paddleocr_on_variant(variant_image)
                candidate_score = score_ocr_result(text, average_confidence)
                if candidate_score > best_score:
                    best_text = text
                    best_score = candidate_score
                    best_variant = f"{variant_name}/paddleocr"
            elif engine_name == "pytesseract":
                for page_segmentation_mode in (6, 11):
                    tesseract_config = f"--oem 3 --psm {page_segmentation_mode}"
                    text, average_confidence = run_ocr_on_variant(variant_image, tesseract_config)
                    candidate_score = score_ocr_result(text, average_confidence)
                    if candidate_score > best_score:
                        best_text = text
                        best_score = candidate_score
                        best_variant = f"{variant_name}/tesseract-psm{page_segmentation_mode}"

    return best_text, best_score, best_variant


def extract_image_documents(file_path: Path) -> list[Document]:
    if Image is None:
        raise ValueError(
            "Image support requires Pillow. Install dependencies from requirements.txt to process screenshots and images."
        )
    if pytesseract is None:
        raise ValueError(
            "Image OCR requires pytesseract. Install pytesseract and the Tesseract OCR binary to process screenshots and images."
        )

    text, ocr_quality_score, ocr_variant = extract_best_ocr_text(file_path)
    if not text:
        raise ValueError("No readable text was detected in the uploaded image.")
    if ocr_quality_score < 0.25:
        raise ValueError(
            "OCR could not confidently read the uploaded image. Please upload a clearer image or higher-resolution screenshot."
        )

    return [
        Document(
            page_content=text,
            metadata={
                "section": 1,
                "section_label": "image ocr",
                "extraction_method": "ocr",
                "ocr_quality_score": round(ocr_quality_score, 3),
                "ocr_variant": ocr_variant,
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
    table_blocks: list[str] = []
    for table_index, table in enumerate(document.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        formatted_table = format_table_rows(rows)
        if formatted_table:
            table_blocks.append(f"Table {table_index}:\n{formatted_table}")

    combined_parts = paragraphs + table_blocks
    if not combined_parts:
        raise ValueError("No readable text was found in the uploaded DOCX file.")

    return [
        Document(
            page_content="\n\n".join(combined_parts),
            metadata={
                "section": 1,
                "section_label": "document body",
                "extraction_method": "docx_text_table",
                "contains_table": bool(table_blocks),
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
    top_k: int = RETRIEVAL_TOP_K,
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
    top_k: int = RERANK_TOP_K,
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


def answer_grounding_ratio(answer: str, documents: list[Document]) -> float:
    answer_tokens = set(tokenize(answer))
    if not answer_tokens or not documents:
        return 0.0

    best_overlap = 0.0
    for doc in documents:
        doc_tokens = set(tokenize(doc.page_content))
        if not doc_tokens:
            continue
        overlap = len(answer_tokens & doc_tokens) / max(len(answer_tokens), 1)
        best_overlap = max(best_overlap, overlap)

    return best_overlap


def lexical_overlap_ratio(query: str, documents: list[Document]) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens or not documents:
        return 0.0

    context_tokens = set()
    for doc in documents:
        context_tokens.update(tokenize(doc.page_content))

    return len(query_tokens & context_tokens) / max(len(query_tokens), 1)


def sentence_relevance_score(query: str, sentence: str) -> float:
    query_tokens = set(tokenize(query))
    sentence_tokens = set(tokenize(sentence))
    if not query_tokens or not sentence_tokens:
        return 0.0
    overlap = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1)
    coverage = len(query_tokens & sentence_tokens) / max(len(sentence_tokens), 1)
    return 0.7 * overlap + 0.3 * coverage


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    array_a = np.array(vector_a)
    array_b = np.array(vector_b)
    denominator = np.linalg.norm(array_a) * np.linalg.norm(array_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(array_a, array_b) / denominator)


def embedding_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0

    embedding_model = get_embedding_model()
    vector_a = embedding_model.embed_query(text_a)
    vector_b = embedding_model.embed_query(text_b)
    return cosine_similarity(vector_a, vector_b)


def collect_supporting_sentences(query: str, documents: list[Document], limit: int = 4) -> list[str]:
    scored_sentences: list[tuple[float, str]] = []
    for doc in documents:
        for sentence in sentence_split(doc.page_content):
            score = sentence_relevance_score(query, sentence)
            if score <= 0:
                continue
            scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    seen = set()
    for _, sentence in scored_sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def verify_answer_against_evidence(
    query: str,
    answer: str,
    documents: list[Document],
) -> tuple[bool, dict[str, float], str]:
    if not answer.strip():
        return False, {"answer_evidence_similarity": 0.0, "query_answer_similarity": 0.0}, "The answer was empty."

    supporting_sentences = collect_supporting_sentences(query, documents, limit=6)
    if not supporting_sentences:
        return False, {"answer_evidence_similarity": 0.0, "query_answer_similarity": 0.0}, (
            "No supporting evidence sentences were found for the answer."
        )

    answer_evidence_similarity = max(
        embedding_similarity(answer, sentence) for sentence in supporting_sentences
    )
    query_answer_similarity = embedding_similarity(query, answer)
    grounding_strength = answer_grounding_ratio(answer, documents)

    if answer_evidence_similarity < MIN_ANSWER_EVIDENCE_SIMILARITY:
        return False, {
            "answer_evidence_similarity": round(answer_evidence_similarity, 3),
            "query_answer_similarity": round(query_answer_similarity, 3),
            "grounding_strength": round(grounding_strength, 3),
        }, "The answer was not semantically close enough to the retrieved evidence."

    if query_answer_similarity < MIN_QUERY_ANSWER_SIMILARITY:
        return False, {
            "answer_evidence_similarity": round(answer_evidence_similarity, 3),
            "query_answer_similarity": round(query_answer_similarity, 3),
            "grounding_strength": round(grounding_strength, 3),
        }, "The answer did not stay close enough to the user question."

    if grounding_strength < MIN_ANSWER_GROUNDING_SCORE:
        return False, {
            "answer_evidence_similarity": round(answer_evidence_similarity, 3),
            "query_answer_similarity": round(query_answer_similarity, 3),
            "grounding_strength": round(grounding_strength, 3),
        }, "The answer did not overlap enough with the retrieved document evidence."

    return True, {
        "answer_evidence_similarity": round(answer_evidence_similarity, 3),
        "query_answer_similarity": round(query_answer_similarity, 3),
        "grounding_strength": round(grounding_strength, 3),
    }, "The answer passed semantic and grounding verification."


def calculate_confidence(
    query: str,
    documents: list[Document],
    rerank_scores: list[float],
    qa_score: float,
    answer: str,
    used_external_fallback: bool,
) -> dict[str, Any]:
    if not documents:
        return {
            "score": 0.1 if used_external_fallback else 0.0,
            "label": "low",
            "reason": "No grounded evidence was found in the uploaded file.",
        }

    semantic_strength = max((normalize_rerank_score(score) for score in rerank_scores), default=0.0)
    average_semantic_strength = (
        sum(normalize_rerank_score(score) for score in rerank_scores) / len(rerank_scores)
        if rerank_scores
        else 0.0
    )
    lexical_strength = lexical_overlap_ratio(query, documents)
    evidence_density = min(len(documents) / 3, 1.0)
    answer_strength = max(min(qa_score, 1.0), 0.0)
    grounding_strength = answer_grounding_ratio(answer, documents)
    relevance_gate = 1.0 if semantic_strength >= MIN_RELEVANCE_SCORE else 0.55

    confidence = (
        0.28 * semantic_strength
        + 0.16 * average_semantic_strength
        + 0.14 * lexical_strength
        + 0.14 * evidence_density
        + 0.14 * answer_strength
        + 0.14 * grounding_strength
    )
    confidence *= relevance_gate

    if used_external_fallback:
        confidence *= 0.65

    label = (
        "high"
        if confidence >= CONFIDENCE_HIGH_THRESHOLD
        else "medium"
        if confidence >= CONFIDENCE_MEDIUM_THRESHOLD
        else "low"
    )
    reason = (
        "Confidence is based on chunk relevance, overlap with the query, evidence coverage, QA score, and whether the extracted answer is grounded in the retrieved chunks."
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
        if not answer or answer.lower() in {"", "[cls]"} or len(answer) < 4:
            continue
        if score > best_score:
            best_answer = answer
            best_score = score

    return best_answer, best_score


def build_local_grounded_answer(query: str, documents: list[Document], extracted_answer: str) -> str:
    if not documents:
        return "I could not find relevant information in the uploaded file for that question."

    supporting_sentences = collect_supporting_sentences(query, documents, limit=4)
    evidence_block = "\n".join(f"- {sentence}" for sentence in supporting_sentences if sentence)

    if extracted_answer and supporting_sentences:
        return (
            f"Answer: {extracted_answer}\n\n"
            f"Grounding: The answer was extracted from the uploaded file and checked against the most relevant evidence.\n\n"
            f"Top evidence:\n{evidence_block}"
        )

    if supporting_sentences:
        return (
            f"I found relevant evidence for the question '{query}'.\n\n"
            f"Most relevant evidence from the uploaded file:\n{evidence_block}"
        )

    best_excerpt = documents[0].page_content.strip()
    return (
        f"I found some related content for the question '{query}', but I could not extract a precise grounded answer.\n\n"
        f"Closest evidence from the uploaded file:\n{best_excerpt[:700].strip()}"
    )


def synthesize_grounded_answer(
    query: str,
    documents: list[Document],
    extracted_answer: str,
    target_language: str,
) -> str:
    if not documents:
        return "I could not find relevant information in the uploaded file for that question."

    supporting_sentences = collect_supporting_sentences(query, documents, limit=6)
    evidence = "\n".join(f"- {sentence}" for sentence in supporting_sentences if sentence)
    if not evidence:
        evidence = "\n".join(
            f"- {doc.page_content.strip()[:300]}"
            for doc in documents[:3]
            if doc.page_content.strip()
        )

    if can_use_remote_fallback():
        prompt = (
            "You are a document question-answering assistant.\n"
            f"Answer in {language_name(target_language)}.\n"
            "Use only the evidence from the uploaded file.\n"
            "If the evidence does not answer the question, say clearly that the answer was not found in the uploaded file.\n"
            "Do not invent facts.\n\n"
            f"Question: {query}\n"
            f"Candidate extracted answer: {extracted_answer or 'none'}\n"
            f"Evidence from uploaded file:\n{evidence}"
        )
        grounded_answer = remote_generation_request(prompt)
        if grounded_answer:
            return grounded_answer

    return build_local_grounded_answer(query, documents, extracted_answer)


def remote_generation_request(prompt: str) -> str:
    if not HF_INFERENCE_API_TOKEN:
        return ""

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


def maybe_translate_answer(
    answer: str,
    source_language: str,
    target_language: str,
    original_query: str,
) -> str:
    if not answer or target_language == source_language:
        return answer
    if not can_use_remote_translation():
        return (
            f"{answer}\n\n"
            f"Note: Translation to {language_name(target_language)} was requested, but remote translation is not available in the current configuration."
        )

    prompt = (
        "You are a multilingual assistant.\n"
        f"Translate the answer below into {language_name(target_language)}.\n"
        "Keep the meaning exact, keep important entities unchanged, and do not add new facts.\n"
        f"User question: {original_query}\n"
        f"Answer to translate:\n{answer}"
    )
    translated = remote_generation_request(prompt)
    return translated or answer


def call_remote_llm(query: str, documents: list[Document], target_language: str) -> str:
    if not can_use_remote_fallback():
        return ""

    evidence = "\n\n".join(doc.page_content.strip()[:800] for doc in documents[:3] if doc.page_content.strip())
    prompt = (
        "You are a multilingual assistant.\n"
        f"Answer in {language_name(target_language)}.\n"
        "Use only the evidence from the uploaded file.\n"
        "Do not answer from general knowledge.\n"
        "If the evidence is insufficient or unrelated to the question, say exactly: 'The answer was not found in the uploaded file.'\n\n"
        f"Question: {query}\n\n"
        f"Evidence from uploaded file:\n{evidence or 'No reliable evidence found in the uploaded file.'}"
    )
    return remote_generation_request(prompt)


def should_use_external_fallback(
    query: str,
    documents: list[Document],
    rerank_scores: list[float],
    qa_score: float,
    extracted_answer: str,
) -> tuple[bool, str]:
    if not documents:
        return True, "No relevant chunks were retrieved from the uploaded file."

    normalized_scores = [normalize_rerank_score(score) for score in rerank_scores]
    top_relevance = max(normalized_scores, default=0.0)
    average_relevance = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
    lexical_strength = lexical_overlap_ratio(query, documents)
    grounding_strength = answer_grounding_ratio(extracted_answer, documents)
    has_short_or_empty_answer = len(extracted_answer.strip()) < 4

    if top_relevance < MIN_RELEVANCE_SCORE:
        return True, (
            f"All retrieved chunks scored below the relevance threshold of {MIN_RELEVANCE_SCORE:.2f}."
        )
    if (
        top_relevance < OUT_OF_SCOPE_RELEVANCE_THRESHOLD
        and average_relevance < OUT_OF_SCOPE_RELEVANCE_THRESHOLD
        and lexical_strength < OUT_OF_SCOPE_LEXICAL_THRESHOLD
        and qa_score < OUT_OF_SCOPE_QA_THRESHOLD
    ):
        return True, "The question appears out of scope for the uploaded file because semantic match, lexical overlap, and QA confidence were all weak."
    if lexical_strength < MIN_LEXICAL_OVERLAP and qa_score < MIN_QA_SCORE:
        return True, "The retrieved chunks had weak overlap with the user question and the QA score was too low."
    if has_short_or_empty_answer and qa_score < OUT_OF_SCOPE_QA_THRESHOLD:
        return True, "The model could not extract a meaningful grounded answer from the retrieved chunks."
    if extracted_answer and grounding_strength < MIN_ANSWER_GROUNDING_SCORE and qa_score < MIN_QA_SCORE:
        return True, "The extracted answer was not grounded strongly enough in the retrieved chunks."

    return False, "The retrieved chunks were relevant enough to answer from the uploaded file."


def build_workflow_steps(
    file_path: Path,
    document_language: str,
    clean_query: str,
    query_language: str,
    target_language: str,
    documents: list[Document],
    rerank_scores: list[float],
    qa_score: float,
    used_external_fallback: bool,
    fallback_reason: str,
) -> list[str]:
    top_score = max((normalize_rerank_score(score) for score in rerank_scores), default=0.0)
    return [
        f"Accepted `{file_path.suffix.lower()}` and selected the matching extraction strategy.",
        f"Detected document language as `{language_name(document_language)}`.",
        f"Normalized the user request to `{clean_query}` for retrieval. Query language: `{language_name(query_language)}`. Target answer language: `{language_name(target_language)}`.",
        f"Extracted {len(documents)} searchable chunks from `{file_path.name}`.",
        f"Built multilingual embeddings with `{EMBEDDING_MODEL_NAME}` and hybrid lexical search.",
        f"Reranked retrieved chunks with `{RERANKER_MODEL_NAME}`. Top normalized rerank score: {top_score:.2f}.",
        f"Ran local multilingual answer extraction with `{LOCAL_QA_MODEL_NAME}`. Best QA score: {qa_score:.2f}.",
        f"Fallback decision: {fallback_reason}",
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
    original_query = query.strip()
    clean_query = original_query
    if not clean_query:
        raise ValueError("Please enter a question before submitting.")

    query_preferences = parse_query_preferences(clean_query, str(pipeline["language_hint"]))
    clean_query = str(query_preferences["clean_query"]).strip()
    query_language = str(query_preferences["query_language"])
    target_language = str(query_preferences["target_language"])
    document_language = str(pipeline["language_hint"])

    aggregated_docs: list[Document] = []
    seen_chunk_ids: set[Any] = set()

    for variant in expand_query(clean_query):
        matches = hybrid_search(
            variant,
            pipeline["chunks"],
            pipeline["vectorstore"],
            pipeline["bm25"],
            top_k=RETRIEVAL_TOP_K,
        )
        for doc in matches:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            aggregated_docs.append(doc)

    top_documents, rerank_scores = rerank_documents(clean_query, aggregated_docs, top_k=RERANK_TOP_K)
    extracted_answer, qa_score = answer_from_documents(clean_query, top_documents)

    used_external_fallback = False
    fallback_reason = "The retrieved chunks were relevant enough to answer from the uploaded file."
    answer = synthesize_grounded_answer(clean_query, top_documents, extracted_answer, target_language)
    answer_verified, verification_metrics, verification_reason = verify_answer_against_evidence(
        clean_query,
        answer,
        top_documents,
    )

    should_fallback, fallback_reason = should_use_external_fallback(
        clean_query,
        top_documents,
        rerank_scores,
        qa_score,
        extracted_answer,
    )

    confidence = calculate_confidence(
        clean_query,
        top_documents,
        rerank_scores,
        qa_score,
        answer,
        used_external_fallback=False,
    )
    should_reject_local_answer = confidence["score"] < LOW_CONFIDENCE_REJECT_THRESHOLD
    if not answer_verified:
        should_reject_local_answer = True
        fallback_reason = f"{fallback_reason} Verification failed: {verification_reason}"

    if should_fallback or should_reject_local_answer:
        fallback_reason = (
            fallback_reason
            if should_fallback
            else (
                "The grounded answer confidence score "
                f"({confidence['score']:.3f}) was below the rejection threshold "
                f"({LOW_CONFIDENCE_REJECT_THRESHOLD:.2f}), so the local answer was rejected."
            )
        )
        answer = ""
        remote_answer = call_remote_llm(clean_query, top_documents, target_language)
        if remote_answer:
            used_external_fallback = True
            answer = remote_answer
            if STRICT_DOCUMENT_GROUNDED and answer.strip() != "The answer was not found in the uploaded file.":
                remote_verified, _, remote_verification_reason = verify_answer_against_evidence(
                    clean_query,
                    answer,
                    top_documents,
                )
                if not remote_verified:
                    answer = "The answer was not found in the uploaded file."
                    fallback_reason = (
                        f"{fallback_reason} Remote synthesis was also rejected: {remote_verification_reason}"
                    )
            confidence = calculate_confidence(
                clean_query,
                top_documents,
                rerank_scores,
                qa_score,
                answer,
                used_external_fallback=True,
            )
        else:
            answer = (
                "I rejected the answer from the uploaded file because its confidence score was below the allowed threshold, and the external fallback model is not available."
            )
    elif target_language != document_language or bool(query_preferences["translation_requested"]):
        answer = maybe_translate_answer(answer, document_language, target_language, original_query)

    return {
        "query": original_query,
        "normalized_query": clean_query,
        "answer": answer,
        "query_language": query_language,
        "target_language": target_language,
        "document_language": document_language,
        "verification_reason": verification_reason,
        "verification_metrics": verification_metrics,
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
        "fallback_reason": fallback_reason,
        "workflow_steps": build_workflow_steps(
            Path(pipeline["document_path"]),
            document_language,
            clean_query,
            query_language,
            target_language,
            top_documents,
            rerank_scores,
            qa_score,
            used_external_fallback,
            fallback_reason,
        ),
    }
