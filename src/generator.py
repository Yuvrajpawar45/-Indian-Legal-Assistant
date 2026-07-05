"""
Stage 9: Qwen3 Generation
Builds a grounded legal-answer prompt from the validated top chunks and
generates an answer with citations using a local model served by Ollama.
"""
import os
import re
from typing import List, Dict
import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")

SYSTEM_PROMPT = (
    "You are a precise Indian legal research assistant. Answer ONLY using the "
    "provided context chunks from Indian statutes. Every claim must be tied to a "
    "specific Section/Article number and Act name from the context. If the context "
    "does not contain the answer, say so explicitly — do not invent law. "
    "Cite as: (Act name, Section X, p. Y) after each claim. "
    "Be concise — answer in 3-5 sentences maximum."
)


def _build_context_block(chunks: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Chunk {i}] Act: {c.get('act_name')} | Section: {c.get('section_no')} | "
            f"Heading: {c.get('heading')} | Source: {c.get('source_file')} p.{c.get('page_num')}\n"
            f"{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: List[Dict]) -> Dict:
    context_block = _build_context_block(chunks)
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer in 3-5 sentences maximum. Cite Section numbers and Act names."
    )

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "num_predict": 400,   # short answers = faster
            "temperature": 0.1,   # deterministic for legal
            "num_ctx": 4096,
        },
    )
    answer_text = response["message"]["content"]

    # Strip any <think>...</think> reasoning block (Qwen3 specific)
    answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

    citations = [
        {
            "act_name": c.get("act_name"),
            "section_no": c.get("section_no"),
            "source_file": c.get("source_file"),
            "page_num": c.get("page_num"),
            "rerank_score": c.get("rerank_score"),
        }
        for c in chunks
    ]

    return {"answer": answer_text, "citations": citations}