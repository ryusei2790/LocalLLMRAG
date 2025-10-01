# -*- coding: utf-8 -*-
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import EmbeddingCfg, QdrantCfg, LLMCfg

EMB = EmbeddingCfg()
QDR = QdrantCfg()
LLM = LLMCfg()

def load_embedder():
    model = SentenceTransformer(EMB.model_name)
    return model

def search(
    client: QdrantClient,
    emb_model: SentenceTransformer,
    query: str,
    top_k: int = 5,
    source_filter: str = None
) -> List[Tuple[float, dict]]:
    qvec = emb_model.encode([query], normalize_embeddings=EMB.normalize)[0]
    flt = None
    if source_filter:
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])
    hits = client.search(
        collection_name=QDR.collection,
        query_vector=qvec.tolist(),
        limit=top_k,
        with_payload=True,
        query_filter=flt
    )
    return [(h.score, h.payload) for h in hits]

def build_prompt(query: str, contexts: List[dict]) -> List[dict]:
    # Chatテンプレ：根拠を示しつつ回答
    context_strs = []
    for i, c in enumerate(contexts, 1):
        context_strs.append(f"[{i}] {c['text']}\n(出典: {c.get('source','')}, chunk {c.get('chunk_id','')})")
    ctx = "\n\n".join(context_strs)

    system = (
        "あなたは事実に忠実なアシスタントです。お客様の質問に対して以下のコンテキストに根拠を求め、"
        "わからない場合は無理に作らず『不明』と述べてください。最後に参照した出典番号を列挙してください。"
    )
    user = (
        f"# 質問\n{query}\n\n"
        f"# コンテキスト（出典付き）\n{ctx}\n\n"
        "指示: 上のコンテキストの範囲で回答してください。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def load_llm():
    dtype = None
    if LLM.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif LLM.dtype == "float16":
        dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(LLM.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM.model_path,
        device_map="auto",
        torch_dtype=dtype,
    )
    return tok, model

def chat(model, tok, messages: List[dict]) -> str:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=LLM.max_new_tokens,
            do_sample=True,
            temperature=LLM.temperature,
            top_p=LLM.top_p,
        )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    query_text = input("質問を入力してください: ").strip()
    emb = load_embedder()
    client = QdrantClient(host=QDR.host, port=QDR.port)

    hits = search(client, emb, query_text, top_k=5)
    contexts = [p for _, p in hits]

    msgs = build_prompt(query_text, contexts)
    tok, model = load_llm()
    answer = chat(model, tok, msgs)
    print("\n=== 回答 ===\n")
    print(answer)

    # print("\n=== 参照コンテキスト（Top-5） ===")
    # for i, (score, payload) in enumerate(hits, 1):
    #     print(f"\n[{i}] score={score:.4f} | {payload.get('source','')}, chunk={payload.get('chunk_id')}\n{payload['text'][:300]}...")

if __name__ == "__main__":
    main()
