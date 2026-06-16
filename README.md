# Enterprise AI Knowledge Operating System

This project is a multilingual document question-answering app built with FastAPI. You upload a file, ask a question about it, and the system answers from the uploaded file only. It supports cross-language questions and can translate the final grounded answer into the user's requested language.

## What This Project Can Do

- Accept multiple file types instead of only PDFs
- Read screenshots and images using OCR
- Read scanned PDFs using OCR fallback when direct PDF text is weak
- Extract text and table content from PDFs and DOCX files
- Read `docx`, `html`, text files, and common code files
- Support multilingual questions and multilingual document content
- Retrieve the most relevant parts of the uploaded file
- Generate answers from retrieved evidence
- Show a confidence score for the answer
- Verify the final answer against document evidence before showing it
- Use a strict external synthesis model only for document-grounded reformulation

## Supported File Types

The app currently supports:

- `pdf`
- `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`, `webp`
- `docx`
- `html`, `htm`
- `txt`, `md`
- `py`, `js`, `ts`, `tsx`, `jsx`
- `java`, `c`, `cpp`, `cs`, `go`, `rs`
- `json`, `yaml`, `yml`, `xml`, `css`, `sql`, `sh`

## How It Works

1. Upload a file.
2. The app extracts text from that file.
3. The app uses EasyOCR, PaddleOCR, and Tesseract OCR when needed for images and scanned PDFs, plus table-aware extraction for PDFs and DOCX files.
4. The text is split into chunks.
5. The chunks are converted into embeddings.
6. The app stores chunk embeddings in Chroma and also builds a BM25 index.
7. The app retrieves the most relevant chunks with embedding search and BM25 together.
8. Those chunks are reranked to improve relevance.
9. The best reranked chunks are used for answer generation.
10. A confidence score is calculated.
11. The answer is verified against the retrieved evidence.
12. If the answer is not grounded strongly enough, the app rejects it and returns a strict document-only fallback response.

## Main Files

- `main.py` handles the FastAPI routes
- `retrieval_services.py` contains extraction, indexing, retrieval, reranking, verification, OCR, and answering logic
- `config.py` contains model names and runtime settings
- `ui.py` renders the HTML pages
- `state.py` stores the current in-memory pipeline
- `run.py` starts the app locally

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the App

```powershell
python run.py
```

By default, the app runs at:

```text
http://127.0.0.1:8001
```

## Configuration

You can change the behavior of the app through environment variables in `config.py`.

Important settings include:

- `APP_HOST`
- `APP_PORT`
- `UPLOAD_DIR`
- `EMBEDDING_MODEL_NAME`
- `RERANKER_MODEL_NAME`
- `LOCAL_QA_MODEL_NAME`
- `HF_REMOTE_LLM_MODEL`
- `HF_INFERENCE_API_TOKEN`
- `ENABLE_REMOTE_FALLBACK`
- `ENABLE_REMOTE_TRANSLATION`
- `STRICT_DOCUMENT_GROUNDED`
- `OCR_ENGINE_PREFERENCE`
- `EASYOCR_LANGUAGES`
- `PADDLEOCR_LANGUAGE`
- `OCR_MIN_QUALITY_SCORE`
- `SCANNED_PDF_TEXT_THRESHOLD`
- `LOW_CONFIDENCE_REJECT_THRESHOLD`
- `MIN_ANSWER_EVIDENCE_SIMILARITY`
- `MIN_QUERY_ANSWER_SIMILARITY`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

## OCR Notes

If you want to extract text from screenshots, scanned PDFs, or image files, the app can now try EasyOCR, PaddleOCR, and Tesseract OCR depending on what is installed and how `OCR_ENGINE_PREFERENCE` is configured.

Without optional OCR engines:

- the app can still start
- direct PDF text, DOCX, HTML, and text/code files still work
- scanned PDFs and image uploads may fall back to the next available OCR engine or show a clear error if none are available

## Current Limitations

- Only one active uploaded file is stored at a time
- State is kept in memory, so it is not persistent across restarts
- OCR quality still depends on image clarity and scan quality
- Remote synthesis and translation need a valid Hugging Face token

## Recent Accuracy Improvements

- stronger multilingual embedding default
- OCR cascade with EasyOCR -> PaddleOCR -> Tesseract
- scanned PDF OCR fallback
- PDF and DOCX table extraction
- strict document-grounded remote synthesis
- answer-vs-evidence semantic verification
- low-confidence answer rejection before final output
