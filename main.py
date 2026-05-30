from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from config import UPLOAD_DIR
from retrieval_services import SUPPORTED_EXTENSIONS, prepare_pipeline, run_query
from state import get_active_pipeline, set_active_pipeline
from ui import render_home_page, render_results_page


app = FastAPI(title="Enterprise Multi-Format Retrieval API", version="2.0.0")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    active_pipeline = get_active_pipeline()
    active_document = active_pipeline["document_name"] if active_pipeline else ""
    return render_home_page(active_document=active_document)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_class=HTMLResponse)
async def upload_document(file: UploadFile = File(...)) -> str:
    if not file.filename:
        return render_home_page("Please upload a supported file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return render_home_page(f"Unsupported file type. Supported types: {allowed}")

    safe_name = Path(file.filename).name
    saved_path = UPLOAD_DIR / safe_name

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pipeline = prepare_pipeline(saved_path)
        set_active_pipeline(pipeline)
    except Exception as exc:
        return render_home_page(f"Processing failed: {exc}")

    return render_home_page(
        message="File uploaded successfully. Multilingual embeddings and retrieval index are ready.",
        active_document=safe_name,
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask_document(query: str = Form(...)) -> str:
    active_pipeline = get_active_pipeline()
    if active_pipeline is None:
        return render_home_page("Please upload a file before asking a question.")

    try:
        payload = run_query(active_pipeline, query)
    except Exception as exc:
        return render_home_page(
            message=f"Query failed: {exc}",
            active_document=active_pipeline["document_name"],
        )

    return render_results_page(active_pipeline["document_name"], payload)
