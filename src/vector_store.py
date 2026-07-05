"""
Stage 5: Qdrant Vector DB
Creates/upserts the collection holding chunk vectors + payload (act, section,
heading, source file, page) for citation.
"""
import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "indian_law")

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection(vector_size: int):
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
        print(f"[qdrant] Created collection '{COLLECTION}' (dim={vector_size})")
    else:
        print(f"[qdrant] Collection '{COLLECTION}' already exists")


def upload_chunks(chunks: List[Dict], vectors: np.ndarray, batch_size: int = 256):
    client = get_client()
    ensure_collection(vectors.shape[1])

    points = []
    for chunk, vec in zip(chunks, vectors):
        points.append(qmodels.PointStruct(
            id=chunk["id"],
            vector=vec.tolist(),
            payload={
                "text": chunk["text"],
                "act_name": chunk.get("act_name"),
                "section_no": chunk.get("section_no"),
                "heading": chunk.get("heading"),
                "source_file": chunk.get("source_file"),
                "page_num": chunk.get("page_num"),
            },
        ))

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)
        print(f"[qdrant] Upserted {i + len(batch)}/{len(points)}")


if __name__ == "__main__":
    chunks_path = Path("data/processed/chunks.json")
    vecs_path = Path("data/processed/embeddings.npy")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    vectors = np.load(vecs_path)
    upload_chunks(chunks, vectors)
    print("[qdrant] Done.")
