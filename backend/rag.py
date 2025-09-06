# backend/rag.py
import os, json
from pathlib import Path
from uuid import uuid4
from typing import List

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data"
INDEX_PATH = BASE_DIR / "storage"
MANIFEST   = INDEX_PATH / "manifest.json"

GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5")

__all__ = ["build_vectorstore", "load_vectorstore", "ingest_paths", "get_qa_chain"]

# ---------- Embeddings / helpers ----------
def _emb():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

def _splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

def _load_manifest() -> dict:
    try:
        with MANIFEST.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_manifest(m: dict):
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

# ---------- Vector store ----------
def build_vectorstore(data_dir: Path | str = DATA_DIR):
    """Full rebuild from /data (PDF/TXT/MD)."""
    data_dir = Path(data_dir)

    pdf_loader = DirectoryLoader(str(data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    txt_loader = DirectoryLoader(str(data_dir), glob="**/*.[tT][xX][tT]", loader_cls=TextLoader, show_progress=True)
    md_loader  = DirectoryLoader(str(data_dir), glob="**/*.md", loader_cls=TextLoader, show_progress=True)

    docs = []
    for loader in (pdf_loader, txt_loader, md_loader):
        try:
            docs.extend(loader.load())
        except Exception:
            pass

    if not docs:
        # create an empty index so the app still runs
        vectordb = FAISS.from_texts([""], _emb())
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vectordb.save_local(str(INDEX_PATH))
        return

    chunks = _splitter().split_documents(docs)
    # track file names for nicer sources
    for c in chunks:
        src = c.metadata.get("source") or c.metadata.get("file_path")
        if src:
            c.metadata["doc_id"] = Path(src).name

    vectordb = FAISS.from_documents(chunks, _emb())
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(INDEX_PATH))

def load_vectorstore() -> FAISS:
    """Load existing FAISS (create empty if missing)."""
    try:
        return FAISS.load_local(str(INDEX_PATH), _emb(), allow_dangerous_deserialization=True)
    except Exception:
        # create a fresh empty store
        vectordb = FAISS.from_texts([""], _emb())
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vectordb.save_local(str(INDEX_PATH))
        return vectordb

def ingest_paths(paths: List[str] | List[Path]) -> dict:
    """Incrementally add specific files to the index."""
    docs = []
    for p in paths:
        p = Path(p)
        ext = p.suffix.lower()
        if ext == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif ext in {".txt", ".md"}:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        else:
            continue

    if not docs:
        return {"added": 0, "docs": []}

    chunks = _splitter().split_documents(docs)
    for c in chunks:
        src = c.metadata.get("source") or c.metadata.get("file_path") or ""
        c.metadata["doc_id"] = Path(src).name if src else p.name

    ids = [str(uuid4()) for _ in chunks]

    try:
        vectordb = load_vectorstore()
    except Exception:
        vectordb = FAISS.from_texts([""], _emb())
        vectordb.docstore._dict.clear()
        vectordb.index.reset()

    vectordb.add_documents(chunks, ids=ids)
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(INDEX_PATH))

    m = _load_manifest()
    for c, _id in zip(chunks, ids):
        doc_id = c.metadata.get("doc_id", "unknown")
        m.setdefault(doc_id, []).append(_id)
    _save_manifest(m)

    return {"added": len(ids), "docs": sorted({c.metadata.get("doc_id") for c in chunks if c.metadata.get("doc_id")})}

# ---------- LLM + Chain ----------
def _llm():
    return ChatGroq(model_name=GROQ_MODEL, temperature=0.1, max_tokens=None)

def get_qa_chain():
    """Build a RetrievalQA chain over the current FAISS index."""
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    system_prompt = (
        "You are a precise Q&A assistant. Answer using the provided context.\n"
        "Cite sources as [S1], [S2], etc. from document metadata. "
        "If the answer is not in the context, say you don't have enough information."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    return RetrievalQA.from_chain_type(
        llm=_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
