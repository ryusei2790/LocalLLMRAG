# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import re
import time
import os

from config import EmbeddingCfg, QdrantCfg, LLMCfg

EMB = EmbeddingCfg()
QDR = QdrantCfg()
LLM = LLMCfg()

# =========================================
# デバイス/共通ユーティリティ
# =========================================
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dtype():
    if LLM.dtype == "bfloat16":
        return torch.bfloat16
    if LLM.dtype == "float16":
        return torch.float16
    return None  # auto

def now_ms() -> int:
    return int(time.time() * 1000)

# =========================================
# 埋め込みモデル読み込み
# =========================================
def load_embedder() -> SentenceTransformer:
    device = pick_device()
    model = SentenceTransformer(EMB.model_name, device=device)
    return model

# =========================================
# キーワードブースト用: クエリから素朴キーワード抽出
# =========================================
_WORD = re.compile(r"[A-Za-z0-9一-龥ぁ-んァ-ンー]+")

def extract_keywords(text: str, max_kw: int = 6) -> List[str]:
    words = _WORD.findall(text)
    # 短い語や助詞を雑に間引く
    stopish = set(["の","に","が","は","を","と","で","や","から","より","and","or","the","a","an","of","to","in"])
    cand = [w for w in words if len(w) >= 2 and w not in stopish]
    # 出現順でユニーク
    seen, out = set(), []
    for w in cand:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(w)
        if len(out) >= max_kw:
            break
    return out

# =========================================
# Qdrant 検索 + MMR 多様化（再ランキング）
# =========================================
def cosine(a: List[float], b: List[float]) -> float:
    # a, b は正規化済み想定（EMB.normalize=True）でも安全のため計算
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-12
    nb = math.sqrt(sum(y*y for y in b)) + 1e-12
    return dot / (na * nb)

def mmr_select(
    query_vec: List[float],
    cand_vecs: List[List[float]],
    k: int,
    lambda_div: float = 0.7
) -> List[int]:
    """
    MMRで候補インデックスを選ぶ。lambda_div が高いほど関連性重視、低いほど多様性重視。
    """
    selected: List[int] = []
    remaining = set(range(len(cand_vecs)))
    while remaining and len(selected) < k:
        best_i, best_score = None, -1e9
        for i in list(remaining):
            rel = cosine(query_vec, cand_vecs[i])
            div = 0.0
            if selected:
                div = max(cosine(cand_vecs[i], cand_vecs[j]) for j in selected)
            score = lambda_div * rel - (1 - lambda_div) * div
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)  # type: ignore
        remaining.remove(best_i) # type: ignore
    return selected

def search(
    client: QdrantClient,
    emb_model: SentenceTransformer,
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
    mmr_lambda: float = 0.7,
    hybrid_boost: float = 0.15,
    timeout: int = 5,
) -> List[Tuple[float, Dict]]:
    """
    - ベクトル検索 (top_k*3) で粗取り
    - クエリ/候補のコサインからMMRで多様化して上位 top_k を選出
    - ハイブリッド風: payloadの title/text/source にキーワード命中で微ブースト
    """
    t0 = now_ms()
    qvec = emb_model.encode([query], normalize_embeddings=EMB.normalize)[0].tolist()
    flt = None
    if source_filter:
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])

    # まずは十分大きく取得して MMR
    rough_k = max(top_k * 3, 12)
    hits = client.search(
        collection_name=QDR.collection,
        query_vector=qvec,
        limit=rough_k,
        with_payload=True,
        query_filter=flt,
        timeout=timeout,
        with_vectors=True,  # MMR用にベクトルを取り出す
    )

    if not hits:
        return []

    # 候補ベクトルとMMR
    cand_vecs = [h.vector for h in hits]  # type: ignore
    selected_idx = mmr_select(qvec, cand_vecs, k=top_k, lambda_div=mmr_lambda)

    # 簡易ハイブリッド: キーワード命中で微ブースト
    kws = extract_keywords(query)
    out: List[Tuple[float, Dict]] = []
    for i in selected_idx:
        h = hits[i]
        base = float(h.score)
        pay = h.payload or {}
        boost = 0.0
        hay = " ".join([
            str(pay.get("title","")),
            str(pay.get("text","")),
            str(pay.get("source","")),
        ]).lower()
        for kw in kws:
            if kw.lower() in hay:
                boost += hybrid_boost
        out.append((base + boost, pay))

    # スコア降順で整列
    out.sort(key=lambda x: x[0], reverse=True)

    # 重複source/chunk抑制（同一テキストなど）
    seen_keys = set()
    deduped: List[Tuple[float, Dict]] = []
    for sc, p in out:
        key = (p.get("source",""), p.get("chunk_id",""), p.get("text","")[:64])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append((sc, p))
    return deduped[:top_k]

# =========================================
# プロンプト生成（トークン予算に合わせて圧縮）
# =========================================
def summarize_for_context(txt: str, max_chars: int = 800) -> str:
    """段落圧縮（乱暴だが高速）"""
    s = re.sub(r"\s+", " ", txt).strip()
    if len(s) <= max_chars:
        return s
    # 文単位で切る
    parts = re.split(r"(?<=[。．！？!?])\s*", s)
    out = []
    total = 0
    for seg in parts:
        if not seg: 
            continue
        if total + len(seg) > max_chars:
            break
        out.append(seg)
        total += len(seg)
    if not out:
        return s[:max_chars]
    return " ".join(out)

def build_prompt(query: str, contexts: List[Dict], tok: AutoTokenizer, ctx_token_budget: int = 2300) -> List[Dict]:
    """
    - LLMのコンテキスト長に合わせてcontextを切り詰め
    - 重要メタ（source/title/page/chunk_id）を明示
    """
    # 1チャンクあたりの目安（ざっくり）
    per_ctx_chars = 900
    blocks = []
    for i, c in enumerate(contexts, 1):
        title = c.get("title","")
        source = c.get("source","")
        page = c.get("page","")
        chunk_id = c.get("chunk_id","")
        body = summarize_for_context(str(c.get("text","")), max_chars=per_ctx_chars)
        block = f"[{i}] {body}\n(出典: {source} | タイトル: {title} | page: {page} | chunk: {chunk_id})"
        blocks.append(block)

    # トークン予算を超えないように後方から削る（新しい/上位を優先）
    sys_base = (
        "あなたは事実に忠実なアシスタントです。回答は以下のコンテキストに厳密に基づき、"
        "不明な点は『不明』と答えてください。推測や脚色はしないでください。"
        "最終行に参照した出典番号（例: [1],[3]）を列挙してください。"
    )
    header = (
        f"# 質問\n{query}\n\n"
        "# コンテキスト（出典付き）\n"
    )

    # 先に粗結合しトークン数を見ながら調整
    joined = header + "\n\n".join(blocks) + "\n\n指示: コンテキストの範囲で箇条書きを用いながら簡潔に回答。最後に参照出典番号を列挙。"
    def count_tokens(s: str) -> int:
        return len(tok(s, add_special_tokens=False).input_ids)

    # 予算 = ctx_token_budget（回答・システム分の余白は別途残す）
    while count_tokens(joined) > ctx_token_budget and blocks:
        blocks.pop()  # 末尾から間引く
        joined = header + "\n\n".join(blocks) + "\n\n指示: コンテキストの範囲で箇条書きを用いながら簡潔に回答。最後に参照出典番号を列挙。"

    return [
        {"role": "system", "content": sys_base},
        {"role": "user", "content": joined},
    ]

# =========================================
# LLM 読み込み
# =========================================
def load_llm():
    if LLM.model_type == "openai":
        # OpenAI APIの場合はNoneを返す（chat関数で直接APIを呼び出す）
        return None, None
    
    # ローカルモデルの場合
    tok = AutoTokenizer.from_pretrained(LLM.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM.model_path,
        device_map="auto",
        torch_dtype=get_dtype(),
    )
    return tok, model

# =========================================
# チャット生成
# =========================================
def chat(model, tok, messages: List[Dict]) -> str:
    if LLM.model_type == "openai":
        return chat_openai(messages)
    
    # ローカルモデルの場合
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    # MPSは half 未対応ケースがあるため to() は安全に
    device = model.device
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=LLM.max_new_tokens,
            do_sample=True,
            temperature=LLM.temperature,
            top_p=LLM.top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    
    # 生成されたトークンのみを取得（入力プロンプトを除外）
    generated_tokens = out[0][inputs['input_ids'].shape[1]:]
    answer = tok.decode(generated_tokens, skip_special_tokens=True)
    
    # assistantの回答部分のみを抽出（念のため）
    if "assistant" in answer.lower():
        # "assistant\n" 以降を取得
        parts = answer.split("assistant", 1)
        if len(parts) > 1:
            answer = parts[1].strip()
    
    return answer.strip()

# =========================================
# OpenAI API チャット生成
# =========================================
def chat_openai(messages: List[Dict]) -> str:
    try:
        import openai
        
        # APIキーを設定
        api_key = LLM.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or configure in config.py")
        
        client = openai.OpenAI(
            api_key=api_key,
            base_url=LLM.openai_base_url if LLM.openai_base_url else None
        )
        
        # OpenAI APIを呼び出し
        response = client.chat.completions.create(
            model=LLM.openai_model,
            messages=messages,
            max_tokens=LLM.max_new_tokens,
            temperature=LLM.temperature,
            top_p=LLM.top_p
        )
        
        return response.choices[0].message.content.strip()
        
    except ImportError:
        raise ImportError("OpenAI library not installed. Run: pip install openai")
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

# =========================================
# メイン処理
# =========================================
def main():
    try:
        query_text = input("質問を入力してください: ").strip()
        if not query_text:
            print("空の質問です。終了します。")
            return

        emb = load_embedder()
        client = QdrantClient(host=QDR.host, port=QDR.port)

        hits = search(client, emb, query_text, top_k=5)
        contexts = [p for _, p in hits]

        tok, model = load_llm()
        msgs = build_prompt(query_text, contexts, tok, ctx_token_budget=2300)

        answer = chat(model, tok, msgs)

        print("\n=== 回答 ===\n")
        print(answer)

    except Exception as e:
        print("\n[ERROR] 実行中に例外が発生しました:")
        print(str(e))

if __name__ == "__main__":
    main()
