"""
V2: Act-level metadata filtering.

Detects when a user question mentions a specific statute and maps that mention
to stable source_file values stored in Qdrant/BM25 payloads.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


ACT_CATALOG: Dict[str, Dict[str, List[str] | str]] = {
    "it_act": {
        "display_name": "Information Technology Act, 2000",
        "source_files": ["IT_Act_2000.pdf", "A2000-21 (1).pdf"],
        "aliases": [
            "information technology act",
            "it act",
            "it act 2000",
            "cyber law",
            "hacking",
            "section 66",
            "section 67",
        ],
    },
    "it_rules": {
        "display_name": "IT Rules, 2021",
        "source_files": ["IT_Rules_2021.pdf"],
        "aliases": [
            "it rules",
            "intermediary guidelines",
            "digital media ethics code",
            "social media intermediary",
            "significant social media intermediary",
        ],
    },
    "bns": {
        "display_name": "Bharatiya Nyaya Sanhita, 2023",
        "source_files": ["BNS_2023.pdf"],
        "aliases": [
            "bharatiya nyaya sanhita",
            "bns",
            "new penal code",
            "criminal offence",
            "punishment under bns",
        ],
    },
    "bnss": {
        "display_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "source_files": ["BNSS_2023.pdf"],
        "aliases": [
            "bharatiya nagarik suraksha sanhita",
            "bnss",
            "criminal procedure",
            "procedure under bnss",
            "arrest",
            "bail",
            "fir",
        ],
    },
    "consumer_protection": {
        "display_name": "Consumer Protection Act, 2019",
        "source_files": ["Consumer_Protection_Act_2019.pdf"],
        "aliases": [
            "consumer protection act",
            "consumer complaint",
            "consumer dispute",
            "deficiency in service",
            "product liability",
            "unfair trade practice",
        ],
    },
}


def _contains_alias(question: str, alias: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, question.lower()) is not None


def extract_act_filter(question: str) -> Optional[List[str]]:
    """Return canonical act keys detected in the question, or None for broad search."""
    matches = []
    for act_key, config in ACT_CATALOG.items():
        aliases = config["aliases"]
        if any(_contains_alias(question, alias) for alias in aliases):  # type: ignore[arg-type]
            matches.append(act_key)
    return matches or None


def source_files_for_filter(act_filter: Optional[List[str]]) -> Optional[List[str]]:
    """Convert canonical act keys into source_file payload values."""
    if not act_filter:
        return None

    source_files = []
    for act_key in act_filter:
        config = ACT_CATALOG.get(act_key)
        if not config:
            continue
        source_files.extend(config["source_files"])  # type: ignore[arg-type]

    return sorted(set(source_files)) or None


def display_names_for_filter(act_filter: Optional[List[str]]) -> List[str]:
    """Human-readable names for logs/API metadata."""
    if not act_filter:
        return []
    return [
        str(ACT_CATALOG[act_key]["display_name"])
        for act_key in act_filter
        if act_key in ACT_CATALOG
    ]
