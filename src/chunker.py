"""
Stage 3: Legal-aware Chunking
Splits cleaned page text into chunks aligned to legal structure (Section /
Article / Clause boundaries) rather than naive fixed-size windows, so each
chunk maps cleanly to a citable section number.
"""
import re
import json
import uuid
from pathlib import Path
from typing import List, Dict

# Matches the START of a new legal unit, e.g. "302. Punishment for murder.—"
# or "Section 8. Definitions" at the start of a line.
UNIT_START_PATTERN = re.compile(
    r"(?m)^(?:Section\s+|Sec\.?\s+|Article\s+|Art\.?\s+|Clause\s+|Rule\s+)?"
    r"(\d+[A-Za-z]*)\.\s*([A-Z][^\n]{0,120})"
)

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150


def _split_by_legal_units(text: str) -> List[Dict]:
    """Split text on detected Section/Article boundaries; fall back to whole text."""
    matches = list(UNIT_START_PATTERN.finditer(text))
    if not matches:
        return [{"section_no": None, "heading": None, "text": text}]

    units = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        unit_text = text[start:end].strip()
        units.append({
            "section_no": m.group(1),
            "heading": m.group(2).strip(),
            "text": unit_text,
        })
    return units


def _subchunk(unit_text: str) -> List[str]:
    """If a legal unit is still too long, split with overlap on sentence/newline boundaries."""
    if len(unit_text) <= MAX_CHUNK_CHARS:
        return [unit_text]

    chunks = []
    start = 0
    while start < len(unit_text):
        end = min(start + MAX_CHUNK_CHARS, len(unit_text))
        # try to break on a sentence boundary
        boundary = unit_text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        else:
            boundary += 1
        chunks.append(unit_text[start:boundary].strip())
        start = max(boundary - OVERLAP_CHARS, boundary)
        if start >= len(unit_text):
            break
    return [c for c in chunks if c]


def chunk_pages(cleaned_pages: List[Dict]) -> List[Dict]:
    """Produce final citable chunks: {id, text, act_name, section_no, heading, source_file, page_num}."""
    chunks = []
    for page in cleaned_pages:
        units = _split_by_legal_units(page["text"])
        for unit in units:
            for sub in _subchunk(unit["text"]):
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": sub,
                    "act_name": page.get("act_name"),
                    "section_no": unit["section_no"],
                    "heading": unit["heading"],
                    "source_file": page["source_file"],
                    "page_num": page["page_num"],
                })
    return chunks


if __name__ == "__main__":
    in_path = Path("data/processed/cleaned_pages.json")
    out_path = Path("data/processed/chunks.json")
    pages = json.loads(in_path.read_text(encoding="utf-8"))
    chunks = chunk_pages(pages)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[chunker] Produced {len(chunks)} legal-aware chunks -> {out_path}")
