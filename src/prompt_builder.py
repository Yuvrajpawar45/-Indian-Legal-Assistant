"""
V2: Prompt builder for grounded legal answers.

Keeps prompt formatting separate from the Ollama client so answer structure can
evolve without touching model-serving code.
"""
from __future__ import annotations

from typing import Dict, List


SYSTEM_INSTRUCTIONS = """You are a precise Indian legal research assistant.
Use only the provided statute context.
Do not invent sections, punishments, procedures, or legal rules.
If the context does not support the answer, say that the indexed corpus does not contain enough information.
Every legal claim must include a citation in this form: (Act name, Section/Rule X, p. Y).
Keep the answer concise and practical."""


def build_context_block(chunks: List[Dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        section = chunk.get("section_no") or "unknown"
        heading = chunk.get("heading") or "No heading"
        blocks.append(
            f"[Context {i}]\n"
            f"Act: {chunk.get('act_name')}\n"
            f"Section/Rule: {section}\n"
            f"Heading: {heading}\n"
            f"Source: {chunk.get('source_file')} p.{chunk.get('page_num')}\n"
            f"Rerank score: {chunk.get('rerank_score')}\n"
            f"Text:\n{chunk.get('text', '')}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: List[Dict]) -> str:
    context = build_context_block(chunks)
    return f"""{SYSTEM_INSTRUCTIONS}

Context:
{context}

Question:
{question}

Answer format:
1. Direct answer in 3-5 sentences.
2. Mention the controlling Act and section/rule wherever available.
3. End with a short "Citations used" line listing the sources.
"""
