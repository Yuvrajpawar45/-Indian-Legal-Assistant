"""
Stage 4: BGE-M3 Embeddings
Encodes chunk text into dense vectors using BAAI/bge-m3 (1024-dim, multilingual,
handles long legal passages well).

Uses FastEmbed (ONNX runtime) via a custom model registration, since BGE-M3
isn't natively in fastembed's built-in registry as of this version. We point
it at Xenova/bge-m3 (a community ONNX export of the same weights) rather than
loading the PyTorch model directly — this is what keeps memory low enough for
constrained deploy environments (e.g. Railway).

NOTE: BGE-M3's ONNX export splits weights across two files — model.onnx (the
graph structure) and model.onnx_data (the actual weight tensors, too large for
a single ONNX file). Both must be listed or FastEmbed silently downloads only
the structure file and crashes at inference time with a missing-file error.
"""
import os
import json
from pathlib import Path
from typing import List
import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

_model = None
_CUSTOM_MODEL_NAME = "Xenova/bge-m3"

# Register once, before any TextEmbedding instance is created.
# CLS pooling + normalization matches BGE-M3's documented dense-embedding method
# (confirmed in Xenova/bge-m3's own usage example) — do not change to MEAN.
TextEmbedding.add_custom_model(
    model=_CUSTOM_MODEL_NAME,
    pooling=PoolingType.CLS,
    normalization=True,
    sources=ModelSource(hf=_CUSTOM_MODEL_NAME),
    dim=1024,
    model_file="onnx/model.onnx",
    additional_files=["onnx/model.onnx_data"],  # <-- NEW: the actual weights file
)


def get_embedder() -> TextEmbedding:
    global _model
    if _model is None:
        print(f"[embedder] Loading {EMBED_MODEL} via FastEmbed custom ONNX ({_CUSTOM_MODEL_NAME}) ...")
        _model = TextEmbedding(model_name=_CUSTOM_MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    model = get_embedder()
    vecs = list(model.embed(texts, batch_size=batch_size))
    vecs = np.asarray(vecs)

    # belt-and-suspenders normalization even though normalization=True above
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    vecs = vecs / norms

    return vecs


if __name__ == "__main__":
    in_path = Path("data/processed/chunks.json")
    out_path = Path("data/processed/embeddings.npy")
    chunks = json.loads(in_path.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    np.save(out_path, vecs)
    print(f"[embedder] Embedded {len(texts)} chunks -> {out_path} (shape={vecs.shape})")