# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class EmbeddingCfg:
    # マルチリンガルが強い埋め込み（どちらかを選択）
    model_name: str = "BAAI/bge-m3"          # 次元: 1024
    # model_name: str = "intfloat/multilingual-e5-large"  # 次元: 1024
    batch_size: int = 64
    normalize: bool = True   # コサイン類似を使う場合 True 推奨

@dataclass
class QdrantCfg:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "rag_docs"

@dataclass
class LLMCfg:
    # あなたのローカルLLMパス（例：Qwen を merge 済み）
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    #model_path: str = "/Users/ryusei/project/mr_seino/LocalLLMRAG/Qwen2.5-0.5B-Instruct"
    #model_path: str = "../LocalLLMLoRA/merged_qwen"  # 例：./merged_qwen or /path/to/Qwen2.5-7B-Instruct
    dtype: str = "auto"                # "auto" / "bfloat16" / "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class ChunkCfg:
    target_tokens: int = 400   # 1チャンクの目安（埋め込み用）
    overlap_tokens: int = 60   # 前後の重なり
    min_chars: int = 150       # 短すぎるチャンクを弾く
