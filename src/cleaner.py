"""
Stage 2: Cleaning + Metadata
Cleans raw extracted text (headers/footers/page-noise) and attaches legal
metadata: Act name, Section/Article/Clause numbers detected via regex.
"""
import re
import json
from pathlib import Path
from typing import List, Dict

# Common noise patterns seen in India Code PDF exports
NOISE_PATTERNS = [
    r"^\s*Page \d+ of \d+\s*$",
    r"^\s*\d+\s*$",                       # bare page numbers
    r"THE GAZETTE OF INDIA.*",
    r"^\s*www\.indiacode\.nic\.in.*$",
]

# Detects "Section 302.", "Sec. 302", "Article 21", "Clause (a)" style headers
SECTION_PATTERN = re.compile(
    r"\b(Section|Sec\.?|Article|Art\.?|Clause|Rule|Order)\s+(\d+[A-Za-z]*(?:\(\d+\))?)\b",
    re.IGNORECASE,
)

# Detects the Act/Code title, e.g. "THE INDIAN PENAL CODE, 1860" or
# "Goods and Services Tax Act, 2017". Matches both "...Act, YYYY" and "...CODE, YYYY" forms.
ACT_TITLE_PATTERN = re.compile(
    r"((?:THE\s+)?[A-Z][A-Za-z,&\s]+?(?:Act|Code|Constitution),?\s*\d{4}?)",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Strip boilerplate noise lines and normalize whitespace."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(pat, stripped, re.IGNORECASE) for pat in NOISE_PATTERNS):
            continue
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def extract_metadata(text: str, source_file: str) -> Dict:
    """Pull Act name + section markers found on this page for downstream chunking."""
    act_match = ACT_TITLE_PATTERN.search(text)
    act_name = act_match.group(1).strip() if act_match else source_file.replace(".pdf", "")

    section_hits = SECTION_PATTERN.findall(text)
    sections = sorted({f"{kind.title()} {num}" for kind, num in section_hits})

    return {
        "act_name": act_name,
        "sections_on_page": sections,
    }


def clean_and_annotate(pages: List[Dict]) -> List[Dict]:
    """Apply cleaning + metadata extraction to every parsed page record."""
    out = []
    for rec in pages:
        cleaned = clean_text(rec["text"])
        if not cleaned:
            continue
        meta = extract_metadata(cleaned, rec["source_file"])
        out.append({
            **rec,
            "text": cleaned,
            **meta,
        })
    return out


if __name__ == "__main__":
    in_path = Path("data/processed/parsed_pages.json")
    out_path = Path("data/processed/cleaned_pages.json")
    pages = json.loads(in_path.read_text(encoding="utf-8"))
    cleaned = clean_and_annotate(pages)
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cleaner] Cleaned {len(cleaned)} pages -> {out_path}")
