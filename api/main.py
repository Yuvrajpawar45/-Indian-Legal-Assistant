"""
FastAPI backend for the Indian Legal RAG system.

POST /query  {"question": "..."}  -> {"answer", "citations", "valid"}
GET  /health -> liveness check
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow `import src.*`

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.pipeline import run_query

app = FastAPI(title="Indian Legal RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    act_name: Optional[str] = None
    section_no: Optional[str] = None
    source_file: Optional[str] = None
    page_num: Optional[int] = None
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    valid: bool
    query_metadata: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = run_query(req.question)
    return result
