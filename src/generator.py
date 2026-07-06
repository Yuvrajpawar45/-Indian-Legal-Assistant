"""
Stage 9: Qwen3 Generation
Builds a structured, grounded legal-answer prompt from validated chunks
and generates an answer with citations using a local model served by Ollama.
"""
import os
import re
from typing import Dict, List

import ollama

from src.prompt_builder import build_prompt

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")


def generate_answer(question: str, chunks: List[Dict]) -> Dict:
    prompt = build_prompt(question, chunks)

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 600,   # structured answers need a little more room
            "temperature": 0.1,   # deterministic for legal
            "num_ctx": 4096,
        },
    )
    answer_text = response["message"]["content"]

    # Strip any <think>...</think> reasoning block (Qwen3 specific).
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
