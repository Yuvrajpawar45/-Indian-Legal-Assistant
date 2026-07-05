"""
Orchestrates the full pipeline:

INGEST mode:  PDFs -> parse -> clean -> chunk -> embed -> Qdrant + BM25 index
QUERY mode:   question -> hybrid retrieval -> rerank -> validate -> Qwen3 generate
"""
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.parser import parse_all_pdfs
from src.cleaner import clean_and_annotate
from src.chunker import chunk_pages
from src.embedder import embed_texts
from src.vector_store import upload_chunks
from src.retriever import hybrid_search, build_bm25_index, TOP_K_DENSE
from src.reranker import rerank, TOP_K_RERANKED
from src.validator import validate_context
from src.generator import generate_answer

RAW_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")


def run_ingest():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Stage 1: Parsing PDFs ===")
    pages = parse_all_pdfs(RAW_DIR)
    (PROCESSED_DIR / "parsed_pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    print("\n=== Stage 2: Cleaning + Metadata ===")
    cleaned = clean_and_annotate(pages)
    (PROCESSED_DIR / "cleaned_pages.json").write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")

    print("\n=== Stage 3: Legal-aware Chunking ===")
    chunks = chunk_pages(cleaned)
    (PROCESSED_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Total chunks: {len(chunks)}")

    print("\n=== Stage 4: BGE-M3 Embeddings ===")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    print("\n=== Stage 5: Upload to Qdrant ===")
    upload_chunks(chunks, vectors)

    print("\n=== Stage 5b: Build BM25 sparse index ===")
    build_bm25_index(chunks)

    print("\n✅ Ingestion complete.")


def run_query(question: str) -> dict:
    t0 = time.time()
    print(f"\n=== Stage 6: Hybrid Retrieval (Dense + BM25) === \nQ: {question}")
    candidates = hybrid_search(question, top_k_each=TOP_K_DENSE)
    print(f"Fused candidates: {len(candidates)} | took {time.time()-t0:.2f}s")

    t1 = time.time()
    print("\n=== Stage 7: BGE Reranking ===")
    reranked = rerank(question, candidates, top_k=TOP_K_RERANKED)
    print(f"Top {len(reranked)} chunks after rerank | took {time.time()-t1:.2f}s")

    t2 = time.time()
    print("\n=== Stage 8: Context Validation ===")
    validation = validate_context(reranked)
    print(f"Validation took {time.time()-t2:.2f}s")
    if not validation["valid"]:
        return {"answer": validation["reason"], "citations": [], "valid": False}

    t3 = time.time()
    print("\n=== Stage 9: Qwen3 Generation ===")
    result = generate_answer(question, validation["chunks"])
    print(f"Generation took {time.time()-t3:.2f}s")
    result["valid"] = True
    print(f"TOTAL: {time.time()-t0:.2f}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Run full ingestion pipeline")
    parser.add_argument("--query", type=str, help="Run a single query end-to-end")
    args = parser.parse_args()

    if args.ingest:
        run_ingest()
    elif args.query:
        result = run_query(args.query)
        print("\n--- ANSWER ---")
        print(result["answer"])
        print("\n--- CITATIONS ---")
        for c in result["citations"]:
            print(c)
    else:
        parser.print_help() 