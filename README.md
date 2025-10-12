# LocalLLMRAG

ローカルLLMとベクトルデータベース（Qdrant）を使用したRAG（Retrieval-Augmented Generation）システムです。PDFやテキストファイルをアップロードし、その内容に基づいて質問に回答するFlask APIアプリケーションです。

## 📋 目次

- [機能](#機能)
- [システム構成](#システム構成)
- [セットアップ](#セットアップ)
- [使い方](#使い方)
- [API仕様](#api仕様)
- [設定のカスタマイズ](#設定のカスタマイズ)

## 🚀 機能

- **文書の埋め込み**: PDFやテキストファイルをベクトル化してQdrantに保存
- **質問応答**: 保存された文書から関連情報を検索し、LLMが回答を生成
- **複数ファイル対応**: 一度に複数のファイルをアップロード可能
- **ローカル実行**: すべての処理がローカル環境で完結（外部APIへの通信不要）
- **高速キャッシュ**: モデルを初回ロード後にメモリに保持して高速化

## 🏗️ システム構成

- **埋め込みモデル**: `sentence-transformers/all-MiniLM-L6-v2`（384次元）
- **LLM**: `Qwen2.5-0.5B-Instruct`（ローカル）
- **ベクトルDB**: Qdrant（ローカル）
- **Webフレームワーク**: Flask
- **チャンク分割**: トークンベースの贪欲分割（オーバーラップ付き）

## 📦 セットアップ

### 1. 必要な環境

- Python 3.10以上
- 十分なメモリ（LLMロードに最低4GB推奨）
- Qdrantサーバー

### 2. 依存パッケージのインストール

```bash
# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 3. Qdrantの起動

Dockerを使用する場合：

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

または、Qdrantをローカルにインストールして起動してください。

### 4. LLMモデルの準備

プロジェクトルートに`Qwen/`ディレクトリが必要です。モデルファイルがない場合は、HuggingFaceからダウンロードしてください：

```bash
# HuggingFace CLIでダウンロード（例）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir Qwen/Qwen2.5-0.5B-Instruct
```

または`config.py`で別のモデルパスを指定できます。

## 💡 使い方

### 1. Flaskサーバーの起動

```bash
python app.py
```

サーバーは`http://localhost:1234`で起動します。

### 2. 文書のアップロード（埋め込み）

PDFやテキストファイルをベクトル化してQdrantに保存します。

**curlの例:**

```bash
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/sample.pdf" \
  -F "files=@docs/document.txt"
```

**Pythonの例:**

```python
import requests

url = "http://localhost:1234/embedd"
files = [
    ('files', open('docs/sample.pdf', 'rb')),
    ('files', open('docs/document.txt', 'rb'))
]

response = requests.post(url, files=files)
print(response.json())
```

**レスポンス例:**

```json
{
  "success": true,
  "message": "ファイルの埋め込みが完了しました",
  "processed_files": 2,
  "file_names": ["sample.pdf", "document.txt"],
  "total_chunks": 45
}
```

### 3. 質問の送信

アップロードした文書に基づいて質問に回答します。

**curlの例:**

```bash
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "この文書の主なテーマは何ですか？",
    "top_k": 5
  }'
```

**Pythonの例:**

```python
import requests

url = "http://localhost:1234/question"
data = {
    "question": "この文書の主なテーマは何ですか？",
    "top_k": 5
}

response = requests.post(url, json=data)
result = response.json()

print(f"質問: {result['question']}")
print(f"回答: {result['answer']}")
print(f"参照文書数: {result['num_contexts']}")
```

**レスポンス例:**

```json
{
  "success": true,
  "question": "この文書の主なテーマは何ですか？",
  "answer": "この文書の主なテーマは、ローカルLLMを活用したRAGシステムの構築です。\n主なポイントは以下の通りです：\n- ベクトル検索による関連情報の取得\n- コンテキストを用いた正確な回答生成\n- ローカル環境での完結した処理\n\n参照: [1],[2],[3]",
  "num_contexts": 5,
  "contexts": [
    {
      "index": 1,
      "source": "sample.pdf",
      "title": "",
      "page": "",
      "chunk_id": 0,
      "text_preview": "RAGシステムは、大規模言語モデルと情報検索を組み合わせた技術です..."
    }
  ]
}
```

### 4. ヘルスチェック

サーバーの稼働状態を確認します。

```bash
curl http://localhost:1234/health
```

**レスポンス:**

```json
{
  "status": "ok",
  "message": "Flask RAG API is running"
}
```

## 📖 API仕様

### `POST /embedd`

文書をアップロードしてベクトル化します。

**リクエスト:**
- Content-Type: `multipart/form-data`
- Body: `files` フィールドに1つ以上のファイル

**対応ファイル形式:**
- `.txt`
- `.md`
- `.pdf`
- `.json`

**レスポンス:**
```json
{
  "success": boolean,
  "message": string,
  "processed_files": number,
  "file_names": string[],
  "total_chunks": number
}
```

### `POST /question`

質問を送信して回答を取得します。

**リクエスト:**
- Content-Type: `application/json`
- Body:
  ```json
  {
    "question": "質問文（必須）",
    "top_k": 5,  // 取得する関連文書数（任意、デフォルト5）
    "source_filter": "sample.pdf"  // 特定ファイルに限定（任意）
  }
  ```

**レスポンス:**
```json
{
  "success": boolean,
  "question": string,
  "answer": string,
  "num_contexts": number,
  "contexts": [
    {
      "index": number,
      "source": string,
      "title": string,
      "page": string,
      "chunk_id": number,
      "text_preview": string
    }
  ]
}
```

### `GET /health`

サーバーのヘルスチェックを行います。

**レスポンス:**
```json
{
  "status": "ok",
  "message": "Flask RAG API is running"
}
```

## ⚙️ 設定のカスタマイズ

`config.py`で各種設定を変更できます。

### 埋め込みモデルの設定

```python
@dataclass
class EmbeddingCfg:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True
```

### Qdrantの設定

```python
@dataclass
class QdrantCfg:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "rag_docs"
```

### LLMの設定

```python
@dataclass
class LLMCfg:
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"  # ローカルLLMパス
    # model_path: str = "Qwen/Qwen2.5-7B-Instruct"  # より高性能なモデル
    dtype: str = "auto"  # "auto" / "bfloat16" / "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
```

### チャンク分割の設定

```python
@dataclass
class ChunkCfg:
    target_tokens: int = 400      # チャンクの目標トークン数
    overlap_tokens: int = 60      # オーバーラップするトークン数
    min_chars: int = 150          # 最小文字数
```

## 📂 ファイル構成

```
LocalLLMRAG/
├── app.py              # Flaskアプリケーション本体
├── config.py           # 設定ファイル
├── ingest.py           # 文書の読み込みと埋め込み処理
├── query.py            # 検索と回答生成処理
├── utils_chunk.py      # チャンク分割ユーティリティ
├── requirements.txt    # 依存パッケージリスト
├── README.md           # このファイル
├── docs/               # アップロード対象の文書を格納
├── Qwen/               # LLMモデルファイル
└── qdrant_storage/     # Qdrantのデータ保存先
```

## 🔧 トラブルシューティング

### Qdrantに接続できない

- Qdrantサーバーが起動しているか確認してください
- `config.py`のホストとポート設定を確認してください

### メモリ不足エラー

- より小さいLLMモデルを使用してください（0.5B版など）
- `config.py`で`dtype`を`"float16"`に設定してメモリ使用量を削減できます

### モデルのロードに時間がかかる

- 初回のみ時間がかかります（数分程度）
- 2回目以降はキャッシュされるため高速です

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

プルリクエストや Issue の報告を歓迎します！


