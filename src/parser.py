"""
Stage 1: PyMuPDF Parser
Extracts raw text page-by-page from Indian Law PDFs (India Code), preserving
page numbers for citation purposes.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict


def parse_pdf(pdf_path: Path) -> List[Dict]:
    """
    Parse a single PDF into a list of page-level records.

    Returns:
        List[{"page_num": int, "text": str, "source_file": str}]
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({
            "page_num": page_num,
            "text": text,
            "source_file": pdf_path.name,
        })
    doc.close()
    return pages


def parse_all_pdfs(raw_dir: Path) -> List[Dict]:
    """Parse every PDF in raw_dir. Returns a flat list of page records."""
    all_pages = []
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[parser] No PDFs found in {raw_dir}. Add India Code PDFs there.")
    for pdf_path in pdf_files:
        print(f"[parser] Parsing {pdf_path.name} ...")
        pages = parse_pdf(pdf_path)
        all_pages.extend(pages)
        print(f"[parser]   -> {len(pages)} pages extracted")
    return all_pages


if __name__ == "__main__":
    import json
    raw_dir = Path("data/raw_pdfs")
    out_path = Path("data/processed/parsed_pages.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages = parse_all_pdfs(raw_dir)
    out_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[parser] Wrote {len(pages)} page records to {out_path}")
