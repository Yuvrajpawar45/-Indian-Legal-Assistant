"""
V2 retrieval evaluation dashboard.

Runs a small query set through rewriting, act filtering, hybrid retrieval, and
reranking. Prints a compact table and writes a Markdown report to snapshots/.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

from src.act_filter import display_names_for_filter, extract_act_filter
from src.query_rewriter import rewrite_query
from src.retriever import TOP_K_DENSE, hybrid_search

DEFAULT_EVAL_PATH = Path("data/eval_queries.json")
DEFAULT_REPORT_PATH = Path("snapshots/retrieval_eval_report.md")
DEFAULT_TOP_K = 5


def _term_hit(text: str, terms: List[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def evaluate_case(case: Dict, top_k: int, use_reranker: bool) -> Dict:
    question = case["question"]
    rewritten = rewrite_query(question)
    act_filter = extract_act_filter(question)
    candidates = hybrid_search(rewritten, top_k_each=TOP_K_DENSE, act_filter=act_filter)
    if use_reranker:
        from src.reranker import rerank

        ranked = rerank(rewritten, candidates, top_k=top_k)
    else:
        ranked = candidates[:top_k]

    expected_files = set(case.get("expected_source_files", []))
    expected_terms = case.get("expected_terms", [])
    retrieved_files = [item.get("source_file") for item in ranked]
    joined_text = "\n".join(item.get("text", "") for item in ranked)

    source_hit = any(source_file in expected_files for source_file in retrieved_files)
    term_hit = _term_hit(joined_text, expected_terms) if expected_terms else True

    return {
        "id": case["id"],
        "question": question,
        "rewritten_question": rewritten,
        "act_filter": display_names_for_filter(act_filter),
        "source_hit": source_hit,
        "term_hit": term_hit,
        "passed": source_hit and term_hit,
        "top_sources": retrieved_files,
        "top_sections": [item.get("section_no") for item in ranked],
        "reranker_used": use_reranker,
    }


def build_report(results: List[Dict]) -> str:
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    source_hits = sum(1 for result in results if result["source_hit"])
    term_hits = sum(1 for result in results if result["term_hit"])

    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"- Cases: {total}",
        f"- Passed: {passed}/{total}",
        f"- Source hit rate: {source_hits}/{total}",
        f"- Term hit rate: {term_hits}/{total}",
        f"- Reranker used: {'yes' if any(result['reranker_used'] for result in results) else 'no'}",
        "",
        "| ID | Pass | Source Hit | Term Hit | Act Filter | Top Sources |",
        "|---|---:|---:|---:|---|---|",
    ]

    for result in results:
        lines.append(
            "| {id} | {passed} | {source_hit} | {term_hit} | {act_filter} | {top_sources} |".format(
                id=result["id"],
                passed="yes" if result["passed"] else "no",
                source_hit="yes" if result["source_hit"] else "no",
                term_hit="yes" if result["term_hit"] else "no",
                act_filter=", ".join(result["act_filter"]) or "None",
                top_sources=", ".join(str(item) for item in result["top_sources"][:5]),
            )
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--skip-rerank", action="store_true", help="Evaluate fused retrieval without loading the reranker")
    args = parser.parse_args()

    cases = json.loads(args.eval_file.read_text(encoding="utf-8"))
    results = [
        evaluate_case(case, top_k=args.top_k, use_reranker=not args.skip_rerank)
        for case in cases
    ]
    report = build_report(results)

    print(report)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(report, encoding="utf-8")
    print(f"[eval] Wrote report to {args.report_file}")


if __name__ == "__main__":
    main()
