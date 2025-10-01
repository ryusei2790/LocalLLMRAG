# -*- coding: utf-8 -*-
import os
import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from config import EmbeddingCfg, QdrantCfg, ChunkCfg
from utils_chunk import greedy_chunk_by_tokens

EMB = EmbeddingCfg()
QDR = QdrantCfg()
CH  = ChunkCfg()

def load_text_from_file(path: str) -> str:
    if path.lower().endswith((".txt", ".md")):
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if path.lower().endswith(".pdf"):
        out = []
        reader = PdfReader(path)
        for page in reader.pages:
            out.append(page.extract_text() or "")
        return "\n".join(out)
    return ""

def discover_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md", ".pdf")):
                files.append(os.path.join(dirpath, fn))
    return files

def ensure_collection(client: QdrantClient, dim: int, name: str):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def embedder():
    model = SentenceTransformer(EMB.model_name)
    if EMB.normalize:
        # SentenceTransformers は encode(normalize_embeddings=True) でもOK
        model.encode(["test"], normalize_embeddings=True)
    return model

def upsert_chunks(
    client: QdrantClient,
    collection: str,
    model: SentenceTransformer,
    chunks: List[Dict],
):
    texts = [c["text"] for c in chunks]
    vecs  = model.encode(texts, batch_size=EMB.batch_size, show_progress_bar=True, normalize_embeddings=EMB.normalize)
    points = []
    for v, meta in zip(vecs, chunks):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=v.tolist(),
            payload=meta
        ))
    client.upsert(collection_name=collection, points=points)

def main():
    src_dir = "docs"   # ← 学習・検索対象の文書ディレクトリ
    files = discover_files(src_dir)
    if not files:
        print("No files found in ./docs. Put .txt/.md/.pdf files there.")
        return

    model = embedder()
    dim = model.encode(["dim_check"], normalize_embeddings=EMB.normalize).shape[-1]
    client = QdrantClient(host=QDR.host, port=QDR.port)
    ensure_collection(client, dim, QDR.collection)

    all_chunks = []
    for fp in files:
        text = load_text_from_file(fp)
        if not text.strip():
            continue
        chunks = greedy_chunk_by_tokens(
            text,
            target_tokens=CH.target_tokens,
            overlap_tokens=CH.overlap_tokens,
            min_chars=CH.min_chars,
        )
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "text": ch,
                "source": os.path.relpath(fp, start=os.getcwd()),
                "chunk_id": i
            })
        print(f"[INGEST] {fp} -> {len(chunks)} chunks")

    upsert_chunks(client, QDR.collection, model, all_chunks)
    print(f"Done. Total chunks: {len(all_chunks)}")

if __name__ == "__main__":
    main()
