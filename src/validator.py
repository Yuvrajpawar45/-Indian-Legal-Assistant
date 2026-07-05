"""
Stage 8: Context Validation
Sanity-checks the top reranked chunks before they're handed to the LLM:
- drops chunks below a minimum relevance score
- drops near-duplicate chunks
- flags if no chunk clears the confidence bar (to avoid hallucinated answers)
"""
import os
from typing import List, Dict

MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.15"))
DEDUP_OVERLAP_THRESHOLD = 0.85


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def validate_context(chunks: List[Dict]) -> Dict:
    """
    Returns:
        {"valid": bool, "chunks": [...filtered...], "reason": str | None}
    """
    if not chunks:
        return {"valid": False, "chunks": [], "reason": "No chunks retrieved."}

    confident = [c for c in chunks if c.get("rerank_score", 0) >= MIN_RERANK_SCORE]
    if not confident:
        return {
            "valid": False,
            "chunks": [],
            "reason": "No retrieved passage met the minimum relevance threshold; "
                      "the question may fall outside the indexed Acts.",
        }

    deduped: List[Dict] = []
    for c in confident:
        if all(_token_overlap(c["text"], d["text"]) < DEDUP_OVERLAP_THRESHOLD for d in deduped):
            deduped.append(c)

    return {"valid": True, "chunks": deduped, "reason": None}
