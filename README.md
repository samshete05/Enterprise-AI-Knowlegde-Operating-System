# Enterprise AI Knowledge Operating System
This project is a simple multilingual document question-answering app built with FastAPI. You upload a file, ask a question about it, and the system tries to answer using the content of that file first. If needed, it can also fall back to an external Hugging Face model.
The goal is to make it easier to work with different kinds of documents, not just PDFs. Also if the user upload the file in differnt language and ask question in differnt language then also it will work and give answer in that user target language.

## What This Project Can Do

- Accept multiple file types instead of only PDFs
- Read screenshots and images using OCR
- Read `docx`, `html`, text files, and common code files
- Support multilingual questions and multilingual document content
- Retrieve the most relevant parts of the uploaded file
- Generate an answer based on the retrieved content
- Show a confidence score for the answer
- Use an external fallback model when local evidence is weak

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

The flow is straightforward:

1. Upload a file.
2. The app extracts text from that file.
3. The text is split into chunks.
4. The chunks are converted into embeddings.
5. The app retrieves the most relevant chunks for the user’s question.
6. Those chunks are reranked to improve relevance.
7. A local model tries to answer the question from the uploaded content.
8. A confidence score is calculated.
9. If fallback is enabled and the confidence is low, an external model can be used.

## Main Files

- `main.py` handles the FastAPI routes
- `retrieval_services.py` contains extraction, indexing, retrieval, reranking, and answering logic
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

Some important settings are:

- `APP_HOST`
- `APP_PORT`
- `UPLOAD_DIR`
- `EMBEDDING_MODEL_NAME`
- `RERANKER_MODEL_NAME`
- `LOCAL_QA_MODEL_NAME`
- `HF_REMOTE_LLM_MODEL`
- `HF_INFERENCE_API_TOKEN`
- `ENABLE_REMOTE_FALLBACK`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

## Performance Notes

The project has already been tuned to reduce startup and retrieval time:

- a smaller multilingual embedding model is used by default
- remote fallback is disabled by default because network calls are slow
- chunk settings are adjusted to reduce unnecessary processing

If you want even faster results, the next best improvement would be optimizing `retrieval_services.py` further by reducing reranking load and query expansion.

## OCR Note For Image Files

If you want to extract text from screenshots or image files, Python packages alone are not enough. You also need Tesseract OCR installed on your system.

Without Tesseract:
- the app can still start
- PDFs, text files, HTML, and code files still work
- image uploads will show a clear error message instead of crashing


### Keras or TensorFlow error from `transformers`

The project is already configured to avoid that TensorFlow import path. If you still see the old error, stop the running process and start the app again.


## Current Limitations

- Only one active uploaded file is stored at a time
- State is kept in memory, so it is not persistent across restarts
- OCR quality depends on image clarity
- External fallback needs a valid Hugging Face token

## Future Improvements

- support multiple uploaded documents at once
- save vector indexes persistently
- improve language detection
- make large-file processing asynchronous
- improve reranking speed further
