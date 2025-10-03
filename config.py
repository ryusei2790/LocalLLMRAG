# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class EmbeddingCfg:
    # 日本語含むRAGなら e5-base or bge-m3 を推奨
    model_name: str = "intfloat/multilingual-e5-base"  # or "BAAI/bge-m3"
    batch_size: int = 64
    normalize: bool = True  # COSINE を使うので True
    dim: int = 768          # bge-m3 を使うなら 1024

@dataclass
class QdrantCfg:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "rag_docs"
    distance: str = "COSINE"   # ← 埋め込みの正規化と対応させる

@dataclass
class RerankerCfg:
    # どちらか。日本語多いなら bge-reranker-v2-m3 が強力
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16

@dataclass
class LLMCfg:
    # 0.5B → 少なくとも 7B へ
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "auto"            # "auto"/"bfloat16"/"float16"
    max_new_tokens: int = 512
    temperature: float = 0.3       # 事実寄せは低め
    top_p: float = 0.9
    context_window: int = 32000    # 32k 版を使う場合

@dataclass
class GateCfg:
    sim_threshold: float = 0.35    # e5-baseの出発点
    dynamic: bool = True           # 平均-α*std などで動的補正
    max_pass: int = 8

@dataclass
class ChunkCfg:
    target_tokens: int = 400       # ← LLMトークン基準で
    overlap_tokens: int = 60
    min_chars: int = 150
    # 可能なら「段落/見出しでのセマンティック分割」を優先
