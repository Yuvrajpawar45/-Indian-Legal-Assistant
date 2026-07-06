"""
Stage 6: Hybrid Retrieval (Dense + BM25)
Combines Qdrant dense vector search with a BM25 sparse index over the same
corpus, then fuses results via Reciprocal Rank Fusion (RRF) before reranking.
Supports optional act_filter to scope retrieval to specific Acts.
"""
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from rank_bm25 import BM25Okapi
from qdrant_client.http import models as qmodels

from src.act_filter import source_files_for_filter
from src.embedder import embed_texts
from src.vector_store import get_client, COLLECTION

TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "20"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "20"))
BM25_INDEX_PATH = Path("data/processed/bm25_index.pkl")

_bm25 = None
_bm25_corpus_meta = None  # list of {id, text, payload...} aligned to bm25 corpus order


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25_index(chunks: List[Dict]):
    """Build and persist a BM25 index over all chunk texts. Run once during ingest."""
    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    meta = [{"id": c["id"], **{k: c.get(k) for k in
             ["text", "act_name", "section_no", "heading", "source_file", "page_num"]}} for c in chunks]
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "meta": meta}, f)
    print(f"[retriever] BM25 index built over {len(chunks)} chunks -> {BM25_INDEX_PATH}")


def _load_bm25():
    global _bm25, _bm25_corpus_meta
    if _bm25 is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _bm25_corpus_meta = data["meta"]
    return _bm25, _bm25_corpus_meta


def _build_qdrant_filter(act_filter: Optional[List[str]]) -> Optional[qmodels.Filter]:
    source_files = source_files_for_filter(act_filter)
    if not source_files:
        return None

    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="source_file",
                match=qmodels.MatchAny(any=source_files),
            )
        ]
    )


def dense_search(
    query: str,
    top_k: int = TOP_K_DENSE,
    act_filter: Optional[List[str]] = None,
) -> List[Dict]:
    client = get_client()
    qvec = embed_texts([query])[0]
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=qvec.tolist(),
        limit=top_k,
        query_filter=_build_qdrant_filter(act_filter),
    )
    results = []
    for h in hits:
        results.append({"id": str(h.id), "score": h.score, **h.payload})
    return results


def bm25_search(
    query: str,
    top_k: int = TOP_K_BM25,
    act_filter: Optional[List[str]] = None,
) -> List[Dict]:
    bm25, meta = _load_bm25()
    scores = bm25.get_scores(_tokenize(query))
    source_files = source_files_for_filter(act_filter)
    if source_files:
        candidate_idx = [
            i for i, item in enumerate(meta)
            if item.get("source_file") in source_files
        ]
    else:
        candidate_idx = list(range(len(scores)))

    ranked_idx = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in ranked_idx:
        results.append({"id": meta[idx]["id"], "score": float(scores[idx]), **meta[idx]})
    return results


def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Fuse multiple ranked lists by RRF: score = sum(1 / (k + rank))."""
    fused_scores: Dict[str, float] = {}
    doc_payload: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_payload[doc_id] = item

    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_payload[doc_id], "fused_score": score} for doc_id, score in fused]


def hybrid_search(
    query: str,
    top_k_each: int = TOP_K_DENSE,
    act_filter: Optional[List[str]] = None,
) -> List[Dict]:
    dense_results = dense_search(query, top_k_each, act_filter=act_filter)
    bm25_results = bm25_search(query, top_k_each, act_filter=act_filter)
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    return fused


if __name__ == "__main__":
    chunks = json.loads(Path("data/processed/chunks.json").read_text(encoding="utf-8"))
    build_bm25_index(chunks)
