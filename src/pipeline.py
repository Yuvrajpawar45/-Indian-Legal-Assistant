"""
Orchestrates the full pipeline:

INGEST mode: PDFs -> parse -> clean -> chunk -> embed -> Qdrant + BM25 index
QUERY mode: question -> rewrite/filter -> hybrid retrieval -> rerank -> validate -> generate
"""
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.act_filter import display_names_for_filter, extract_act_filter
from src.chunker import chunk_pages
from src.cleaner import clean_and_annotate
from src.embedder import embed_texts
from src.generator import generate_answer
from src.parser import parse_all_pdfs
from src.query_rewriter import rewrite_query
from src.reranker import TOP_K_RERANKED, rerank
from src.retriever import TOP_K_DENSE, build_bm25_index, hybrid_search
from src.validator import validate_context
from src.vector_store import upload_chunks
import os

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

RAW_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")


def run_ingest():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Stage 1: Parsing PDFs ===")
    pages = parse_all_pdfs(RAW_DIR)
    (PROCESSED_DIR / "parsed_pages.json").write_text(
        json.dumps(pages, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Stage 2: Cleaning + Metadata ===")
    cleaned = clean_and_annotate(pages)
    (PROCESSED_DIR / "cleaned_pages.json").write_text(
        json.dumps(cleaned, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Stage 3: Legal-aware Chunking ===")
    chunks = chunk_pages(cleaned)
    (PROCESSED_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Total chunks: {len(chunks)}")

    print("\n=== Stage 4: BGE-M3 Embeddings ===")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    print("\n=== Stage 5: Upload to Qdrant ===")
    upload_chunks(chunks, vectors)

    print("\n=== Stage 5b: Build BM25 sparse index ===")
    build_bm25_index(chunks)

    print("\nIngestion complete.")


def run_query(question: str) -> dict:
    t0 = time.time()
    rewritten_question = rewrite_query(question)
    act_filter = extract_act_filter(question)
    act_names = display_names_for_filter(act_filter)

    print("\n=== V2: Query Rewrite + Metadata Filter ===")
    print(f"Original: {question}")
    print(f"Rewritten: {rewritten_question}")
    print(f"Act filter: {act_names or 'None'}")

    print(f"\n=== Stage 6: Hybrid Retrieval (Dense + BM25) === \nQ: {rewritten_question}")
    candidates = hybrid_search(
        rewritten_question,
        top_k_each=TOP_K_DENSE,
        act_filter=act_filter,
    )
    print(f"Fused candidates: {len(candidates)} | took {time.time()-t0:.2f}s")

    t1 = time.time()
    if RERANK_ENABLED:
        print("\n=== Stage 7: BGE Reranking ===")
        reranked = rerank(rewritten_question, candidates, top_k=TOP_K_RERANKED)
        print(f"Top {len(reranked)} chunks after rerank | took {time.time()-t1:.2f}s")
    else:
        print("\n=== Stage 7: BGE Reranking (SKIPPED — RERANK_ENABLED=false) ===")
        reranked = candidates[:TOP_K_RERANKED]
        print(f"Top {len(reranked)} chunks from hybrid fusion (no rerank) | took {time.time()-t1:.2f}s")
    t2 = time.time()
    print("\n=== Stage 8: Context Validation ===")
    validation = validate_context(reranked)
    print(f"Validation took {time.time()-t2:.2f}s")
    if not validation["valid"]:
        return {
            "answer": validation["reason"],
            "citations": [],
            "valid": False,
            "query_metadata": {
                "original_question": question,
                "rewritten_question": rewritten_question,
                "act_filter": act_filter or [],
                "act_filter_display": act_names,
            },
        }

    t3 = time.time()
    print("\n=== Stage 9: Qwen3 Generation ===")
    result = generate_answer(question, validation["chunks"])
    print(f"Generation took {time.time()-t3:.2f}s")
    result["valid"] = True
    result["query_metadata"] = {
        "original_question": question,
        "rewritten_question": rewritten_question,
        "act_filter": act_filter or [],
        "act_filter_display": act_names,
    }
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
        for citation in result["citations"]:
            print(citation)
        print("\n--- QUERY METADATA ---")
        print(json.dumps(result.get("query_metadata", {}), indent=2))
    else:
        parser.print_help()
