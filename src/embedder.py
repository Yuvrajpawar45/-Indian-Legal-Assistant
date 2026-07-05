"""
Stage 4: BGE-M3 Embeddings
Encodes chunk text into dense vectors using BAAI/bge-m3 (1024-dim, multilingual,
handles long legal passages well).
"""
import os
import json
from pathlib import Path
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading {EMBED_MODEL} ...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs)


if __name__ == "__main__":
    in_path = Path("data/processed/chunks.json")
    out_path = Path("data/processed/embeddings.npy")
    chunks = json.loads(in_path.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    np.save(out_path, vecs)
    print(f"[embedder] Embedded {len(texts)} chunks -> {out_path} (shape={vecs.shape})")
