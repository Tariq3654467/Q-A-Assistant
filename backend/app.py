# backend/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil

from rag import get_qa_chain, ingest_paths

app = FastAPI(title="RAG Q/A API")

# --- CORS (dev-friendly; tighten in prod) ---
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5500",  # VS Code Live Server (optional)
    "null",                   # file:// during dev (optional)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Storage ---
DATA_DIR = Path(__file__).parent.parent / "data" / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- RAG chain cache ---
qa_chain = None

class AskBody(BaseModel):
    question: str

@app.on_event("startup")
def _startup():
    global qa_chain
    qa_chain = get_qa_chain()

@app.get("/health")
def health():
    return {"status": "ok"}

# Optional helper so preflight always succeeds (CORS should already cover it)
@app.options("/ask")
def options_ask():
    return Response(status_code=200)

@app.post("/ask")
def ask(body: AskBody):
    global qa_chain
    if qa_chain is None:
        qa_chain = get_qa_chain()

    result = qa_chain({"query": body.question})
    answer = result.get("result", "")

    sources = []
    for i, doc in enumerate(result.get("source_documents", []), start=1):
        meta = doc.metadata or {}
        src = meta.get("source") or meta.get("file_path") or meta.get("doc_id") or f"chunk_{i}"
        sources.append({"id": f"S{i}", "source": src})

    return {"answer": answer, "sources": sources}

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved_paths = []
    allowed = {".pdf", ".txt", ".md"}

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed:
            continue
        dest = DATA_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(str(dest))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files (pdf/txt/md).")

    result = ingest_paths(saved_paths)

    # refresh retriever so new docs are used immediately
    global qa_chain
    qa_chain = get_qa_chain()

    return {"saved": [Path(p).name for p in saved_paths], **result}
