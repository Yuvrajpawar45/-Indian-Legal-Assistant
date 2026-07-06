"""
V2: Lightweight legal query rewriting.

This is intentionally deterministic and offline. It expands common user terms
into statute-friendly wording so retrieval has better lexical and semantic
signals without changing the user's actual question.
"""
from __future__ import annotations

import re
from typing import Dict, List


LEGAL_EXPANSIONS: Dict[str, List[str]] = {
    "hacking": ["unauthorised access", "computer resource", "Section 66"],
    "hack": ["unauthorised access", "computer resource", "Section 66"],
    "data theft": ["computer data", "identity theft", "computer resource"],
    "privacy": ["personal information", "sensitive personal data"],
    "social media": ["intermediary", "IT Rules 2021", "due diligence"],
    "consumer complaint": ["consumer dispute", "defect", "deficiency in service"],
    "refund": ["consumer dispute", "deficiency in service", "unfair trade practice"],
    "bail": ["release on bail", "bond", "criminal procedure"],
    "arrest": ["arrest procedure", "police officer", "criminal procedure"],
    "fir": ["first information report", "information to police"],
    "cheating": ["dishonestly", "fraudulently", "deception"],
}


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def rewrite_query(question: str) -> str:
    """
    Expand a natural-language legal question for retrieval.

    The original question is still sent to the generator; this rewritten string
    is only used for retrieval and reranking.
    """
    clean_question = " ".join(question.strip().split())
    expansions: List[str] = []

    for phrase, phrase_expansions in LEGAL_EXPANSIONS.items():
        if _contains_phrase(clean_question, phrase):
            expansions.extend(phrase_expansions)

    unique_expansions = []
    seen = set()
    for item in expansions:
        key = item.lower()
        if key not in seen and not _contains_phrase(clean_question, item):
            seen.add(key)
            unique_expansions.append(item)

    if not unique_expansions:
        return clean_question

    return f"{clean_question} Related legal terms: {', '.join(unique_expansions)}."
