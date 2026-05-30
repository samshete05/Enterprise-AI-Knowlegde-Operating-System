from __future__ import annotations

from html import escape
from typing import Any


def _page_shell(body: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Enterprise Multi-Format Retrieval</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                margin: 0;
                padding: 40px 16px;
                color: #1f2937;
            }}
            .card {{
                max-width: 760px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 16px;
                padding: 28px;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
            }}
            h1, h2 {{
                color: #0f172a;
            }}
            h1 {{
                margin-top: 0;
            }}
            p {{
                line-height: 1.6;
            }}
            label {{
                display: block;
                margin: 18px 0 8px;
                font-weight: 700;
            }}
            input[type="file"],
            input[type="text"] {{
                width: 100%;
                padding: 12px;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                box-sizing: border-box;
            }}
            button, .link-button {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 18px;
                border: 0;
                border-radius: 10px;
                background: #0f766e;
                color: #fff;
                font-weight: 700;
                cursor: pointer;
                text-decoration: none;
            }}
            .message {{
                margin-top: 16px;
                padding: 12px 14px;
                border-radius: 10px;
                background: #f8fafc;
                border: 1px solid #cbd5e1;
            }}
            .active-doc {{
                margin: 20px 0;
                padding: 14px 16px;
                border-radius: 10px;
                background: #ecfeff;
                border: 1px solid #99f6e4;
            }}
            .answer {{
                margin-top: 20px;
                padding: 16px;
                border-radius: 12px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                white-space: pre-wrap;
            }}
            .source {{
                margin-top: 16px;
                padding: 14px 16px;
                border-radius: 12px;
                background: #fff;
                border: 1px solid #e2e8f0;
            }}
            .source-title {{
                font-weight: 700;
                margin-bottom: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            {body}
        </div>
    </body>
    </html>
    """


def render_home_page(message: str = "", active_document: str = "") -> str:
    active_block = ""
    if active_document:
        active_block = f"""
            <div class="active-doc">
                <strong>Active File:</strong> {escape(active_document)}<br />
                Multilingual retrieval is ready. You can now ask a prompt in your preferred language.
            </div>
            <form action="/ask" method="post">
                <label for="query">Ask a Prompt</label>
                <input id="query" name="query" type="text" placeholder="Example: Summarize the key exclusions or ask in your own language" required />
                <button type="submit">Ask Document</button>
            </form>
        """

    message_block = f'<div class="message">{escape(message)}</div>' if message else ""

    return _page_shell(
        f"""
        <h1>Enterprise AI Multi-Format Retrieval</h1>
        <p>Upload a PDF, screenshot image, DOCX, HTML, or text/code file. The app extracts text, builds multilingual embeddings, and then answers prompts grounded in the uploaded file.</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="file">Upload File</label>
            <input id="file" name="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.docx,.html,.htm,.txt,.md,.py,.js,.ts,.tsx,.jsx,.java,.c,.cpp,.cs,.go,.rs,.json,.yaml,.yml,.xml,.css,.sql,.sh" required />
            <button type="submit">Upload And Index File</button>
        </form>
        {message_block}
        {active_block}
        """
    )


def render_results_page(active_document: str, payload: dict[str, Any]) -> str:
    sources = payload.get("sources", [])
    workflow_steps = payload.get("workflow_steps", [])
    confidence = payload.get("confidence", {})
    fallback_note = ""

    if payload.get("used_external_fallback"):
        fallback_note = """
        <div class="message">
            Grounded evidence was weak, so the app also used the configured external Hugging Face fallback model.
        </div>
        """

    source_blocks = ""
    for source in sources:
        source_blocks += f"""
        <div class="source">
            <div class="source-title">
                Source chunk {escape(str(source.get("chunk_id", "n/a")))} from {escape(str(source.get("document_name", active_document)))}
            </div>
            <div><strong>Section:</strong> {escape(str(source.get("section_label", "section")))}</div>
            <div><strong>Language hint:</strong> {escape(str(source.get("language_hint", "unknown")))}</div>
            <div>{escape(str(source.get("preview", "")))}</div>
        </div>
        """

    if not source_blocks:
        source_blocks = '<div class="message">No source snippets were returned for this query.</div>'

    workflow_html = ""
    for step in workflow_steps:
        workflow_html += f"<li>{escape(str(step))}</li>"

    return _page_shell(
        f"""
        <h1>Enterprise AI Multi-Format Retrieval</h1>
        <div class="active-doc">
            <strong>Active File:</strong> {escape(active_document)}
        </div>
        <h2>Question</h2>
        <p>{escape(str(payload.get("query", "")))}</p>
        <h2>Answer</h2>
        <div class="answer">{escape(str(payload.get("answer", "")))}</div>
        <h2>Confidence</h2>
        <div class="message">
            <strong>{escape(str(confidence.get("label", "unknown")).upper())}</strong>
            ({escape(str(confidence.get("score", "0")))})<br />
            {escape(str(confidence.get("reason", "")))}
        </div>
        {fallback_note}
        <h2>Workflow</h2>
        <div class="source">
            <ol>{workflow_html}</ol>
        </div>
        <h2>Source Chunks</h2>
        {source_blocks}
        <a class="link-button" href="/">Ask Another Question</a>
        """
    )
