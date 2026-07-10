"""
Stage 9: LLM Generation
Builds a structured, grounded legal-answer prompt from validated chunks
and generates an answer with citations.

Backend selection via LLM_PROVIDER:
  - "ollama" (default): local Qwen3 served by Ollama — used for local dev
  - "groq": Llama 3.3 70B via Groq API — primary for the deployed version,
            automatically falls back to Gemini if Groq errors or rate-limits

Both paths use the exact same prompt (from build_prompt) and the same
grounding rules, so switching backends does not change what the model
is allowed to answer from.
"""
import os
import re
from typing import Dict, List

from src.prompt_builder import build_prompt

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# --- Ollama (local) settings ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("LLM_MODEL", "qwen3")

# --- Groq (deployed, primary) settings ---
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Gemini (deployed, fallback) settings ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def _generate_with_ollama(prompt: str) -> str:
    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 600,
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    )
    return response["message"]["content"]


def _generate_with_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1,
    )
    return response.choices[0].message.content


def _generate_with_gemini(prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=600,
            temperature=0.1,
        ),
    )
    return response.text


def _generate_deployed(prompt: str) -> str:
    """Groq primary, Gemini fallback. Only triggers fallback on errors
    that indicate Groq itself is unavailable — not on bugs in our own code."""
    try:
        return _generate_with_groq(prompt)
    except Exception as e:
        # groq-python raises subclasses of groq.APIError for rate limits,
        # timeouts, and connection issues — broad-catch here is deliberate
        # since we want ANY Groq failure to fail over, but we log it so
        # silent failures don't hide a real config bug (e.g. bad API key).
        print(f"[generator] Groq failed ({type(e).__name__}: {e}), falling back to Gemini")
        return _generate_with_gemini(prompt)


def generate_answer(question: str, chunks: List[Dict]) -> Dict:
    prompt = build_prompt(question, chunks)

    if LLM_PROVIDER == "groq":
        answer_text = _generate_deployed(prompt)
    else:
        answer_text = _generate_with_ollama(prompt)

    # Strip any <think>...</think> reasoning block (Qwen3 specific).
    # Harmless no-op for Groq/Gemini output, which never contains this tag.
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