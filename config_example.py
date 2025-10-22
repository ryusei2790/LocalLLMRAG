# -*- coding: utf-8 -*-
"""
設定例ファイル

このファイルを参考にconfig.pyを設定してください。
"""

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
    # モデルタイプ: "local" (ローカル) または "openai" (OpenAI API)
    model_type: str = "local"  # "local" / "openai" / "openai_gpt4o_mini" / "openai_gpt4o"
    
    # ローカルモデル設定（model_type="local"の場合）
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # ローカルパスの場合: "Qwen/Qwen2.5-0.5B-Instruct" (相対パス)
    # HuggingFace Hubの場合: "Qwen/Qwen2.5-0.5B-Instruct" (自動ダウンロード)
    # 本番推奨: "tokyotech-llm/Swallow-MS-7b-v0.1"
    dtype: str = "auto"                # "auto" / "bfloat16" / "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # OpenAI API設定（model_type="openai"の場合）
    openai_model: str = "gpt-4o-mini"  # "gpt-4o-mini" / "gpt-4o" / "gpt-3.5-turbo"
    openai_api_key: str = ""  # 環境変数 OPENAI_API_KEY から取得
    openai_base_url: str = ""  # カスタムエンドポイント用（空の場合はデフォルト）

@dataclass
class ChunkCfg:
    target_tokens: int = 400
    overlap_tokens: int = 60
    min_chars: int = 150

# =========================================
# 設定例
# =========================================

# 例1: ローカルモデルを使用
LOCAL_CONFIG = {
    "model_type": "local",
    "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_new_tokens": 512,
    "temperature": 0.7
}

# 例2: OpenAI API (GPT-4o mini) を使用
OPENAI_CONFIG = {
    "model_type": "openai",
    "openai_model": "gpt-4o-mini",
    "openai_api_key": "",  # 環境変数から取得
    "max_new_tokens": 512,
    "temperature": 0.7
}

# 例3: OpenAI API (GPT-4o) を使用
OPENAI_GPT4_CONFIG = {
    "model_type": "openai",
    "openai_model": "gpt-4o",
    "openai_api_key": "",  # 環境変数から取得
    "max_new_tokens": 1024,
    "temperature": 0.7
}

# 例4: カスタムエンドポイントを使用
CUSTOM_ENDPOINT_CONFIG = {
    "model_type": "openai",
    "openai_model": "gpt-4o-mini",
    "openai_api_key": "your-api-key",
    "openai_base_url": "https://your-custom-endpoint.com/v1",
    "max_new_tokens": 512,
    "temperature": 0.7
}
