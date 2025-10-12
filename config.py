# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class EmbeddingCfg:
    # safetensors 版を使用（GPU 対応）
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2" #Qdrant/multilingual-e5-large-onnx こっちを本番環境で使う
    batch_size: int = 64
    normalize: bool = True   # コサイン類似を使う場合 True 推奨

@dataclass
class QdrantCfg:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "rag_docs"

@dataclass
class LLMCfg:
    # HuggingFaceから直接ダウンロード（初回のみ時間がかかる）
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # ローカルパスの場合: "Qwen/Qwen2.5-0.5B-Instruct" (相対パス)
    # HuggingFace Hubの場合: "Qwen/Qwen2.5-0.5B-Instruct" (自動ダウンロード)
    # 本番推奨: "tokyotech-llm/Swallow-MS-7b-v0.1"
    dtype: str = "auto"                # "auto" / "bfloat16" / "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class ChunkCfg:
    target_tokens: int = 400
    overlap_tokens: int = 60
    min_chars: int = 150
