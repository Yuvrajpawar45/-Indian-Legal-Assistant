"""
Stage 7: BGE Reranker
Re-scores the fused hybrid candidates using BAAI/bge-reranker-v2-m3
(cross-encoder) for precise query-passage relevance, then keeps the top N.
"""
import os
from typing import List, Dict
from FlagEmbedding import FlagReranker

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", "5"))

_reranker = None


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        print(f"[reranker] Loading {RERANK_MODEL} ...")
        _reranker = FlagReranker(RERANK_MODEL, use_fp16=True)
    return _reranker


def rerank(query: str, candidates: List[Dict], top_k: int = TOP_K_RERANKED) -> List[Dict]:
    """Cross-encode (query, chunk_text) pairs and return the top_k highest scoring chunks."""
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
