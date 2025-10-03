# Local LLM RAG（ローカル知識ベース検索 + 生成）

このリポジトリは、ローカルの埋め込みモデル + Qdrant ベクタDB + ローカル LLM（例: Qwen 系）で、手元ドキュメントに対するRAG（Retrieval-Augmented Generation）をシンプルに実行するための最小構成です。

- 文書取り込み: `ingest.py` が `docs/` 配下の `.txt/.md/.pdf` を分割・埋め込みし、Qdrant に投入
- 検索 + 生成: `query.py` が埋め込み検索の上位文脈をもとにローカル LLM へプロンプト
- 設定: `config.py` で埋め込み/DB/LLM/チャンクの各種設定を一括管理

---

## 前提条件
- OS: macOS (他OSでもPython + Qdrantが動作すれば可)
- Python: 3.9 以上推奨（本環境は 3.13）
- GPU: 任意（CPUでも可、LLMサイズ/速度に影響）
- Qdrant: ローカルで起動（Docker 推奨）

### Qdrant の起動（Docker 例）
```bash
# 6333(HTTP) と 6334(gRPC) を公開
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```
起動後、`http://localhost:6333/dashboard` にアクセス可能です。

---

## セットアップ
### 1) 仮想環境
```bash
cd /Users/ryusei/project/mr_seino/LocalLLMRAG
python3 -m venv venv
source venv/bin/activate
```

### 2) 依存ライブラリのインストール
```bash
pip install -U pip
pip install -r requirements.txt
```

初回は埋め込みモデルやトークナイザ等を自動でダウンロードします（Hugging Faceのキャッシュ使用）。

---

## ディレクトリ構成（主なもの）
- `docs/`: 取り込み対象のドキュメント（`.txt/.md/.pdf`）を配置
- `ingest.py`: 文書分割・埋め込み生成・Qdrant への upsert
- `query.py`: 検索→プロンプト組み立て→ローカル LLM で回答生成
- `utils_chunk.py`: ざっくりトークン長ベースの貪欲チャンク分割
- `config.py`: 各種設定（埋め込み/Qdrant/LLM/チャンク）
- `qdrant_storage/`: Qdrant 永続化データ（Docker の -v 連携時）
- `Qwen/`: サンプルのトークナイザ等（実運用は任意のローカル LLM へ差し替え）

---

## 設定
すべて `config.py` で編集できます。

- EmbeddingCfg
  - `model_name`: 例 `"BAAI/bge-m3"` または `"intfloat/multilingual-e5-large"`
  - `batch_size`, `normalize`: エンコード設定
- QdrantCfg
  - `host`, `port`, `collection`: 例 `127.0.0.1:6333`, `rag_docs`
- LLMCfg
  - `model_path`: ローカル LLM のパスまたは Hugging Face リポ名
    - 例: `"Qwen/Qwen2.5-0.5B-Instruct"`（HFから取得）
    - 例: ローカルフォルダ（事前に重みを配置）
  - `dtype`: `auto`/`bfloat16`/`float16`
  - `max_new_tokens`, `temperature`, `top_p`: 生成パラメータ
- ChunkCfg
  - `target_tokens`, `overlap_tokens`, `min_chars`: チャンク分割の粒度

---

## 使い方
### 1) ドキュメントを配置
`docs/` フォルダに `.txt/.md/.pdf` を入れます。

### 2) 取り込み（埋め込み＋DB upsert）
```bash
python ingest.py
```
- `docs/` 以下を再帰的に探索し、文書を文単位で分割→チャンク化→埋め込み→Qdrant へ upsert します。
- コレクションが未作成の場合は自動作成（ベクトル次元は埋め込みモデルから自動判定）。

### 3) 検索 + 生成
```bash
python query.py
```
- 対話プロンプトが表示されるので質問を入力
- Top-K（デフォルト 5）の文脈をプロンプトに詰め、ローカル LLM が回答します

---

## 実行例
```bash
$ python ingest.py
[INGEST] docs/greenheardRAG.pdf -> 42 chunks
Done. Total chunks: 42

$ python query.py
質問を入力してください: この資料のRAGの流れを教えて

=== 回答 ===
（モデル出力が表示）
```

---

## LLM の準備
- デフォルトでは `LLMCfg.model_path = "Qwen/Qwen2.5-0.5B-Instruct"` を参照します。
- 完全ローカルで通信を遮断したい場合は、ローカルにダウンロード済みのモデルディレクトリを指定してください。
  - 例: `LLMCfg.model_path = "/path/to/Qwen2.5-0.5B-Instruct"`
- Qwen 以外の CausalLM 互換モデルでも動作します（`transformers` に準拠）。

---

## よくある質問（FAQ）/ トラブルシューティング
- Q: Qdrant に接続できない
  - A: Docker で Qdrant を起動し、`config.py` の `QdrantCfg.host/port` を確認してください。
- Q: 取り込み後にヒットしない
  - A: `docs/` が空でないか、PDF からテキスト抽出できているか、`ChunkCfg.min_chars` が厳しすぎないか確認してください。
- Q: モデルが見つからない/遅い
  - A: 初回はモデル等を自動取得します。完全ローカルにしたい場合は事前にダウンロードし `model_path` をローカルパスに設定。軽量モデルへ切替も検討。
- Q: メモリ不足
  - A: `dtype` を `float16` にする、`max_new_tokens` を下げる、より小さい LLM を使う、CPU 実行するなどを検討。

---

## 開発メモ
- チャンク分割は `utils_chunk.greedy_chunk_by_tokens` を使用。日本語の句点などで文分割し、目標トークン長・オーバーラップを考慮して結合します。
- 近傍探索は Qdrant のコサイン類似度。メタデータとして `source`, `chunk_id` を保存しています。
- プロンプトは根拠（出典番号）を最後に列挙する方針で組み立てています。

---

## ライセンス
このリポジトリ自体のライセンスは、含まれるファイルの記載に従います。各モデル/依存ライブラリはそれぞれのライセンスに従ってください。

---

## SSHでクラウドGPUを使う場合（環境設定）
クラウド上のGPUインスタンス（例: Ubuntu 22.04 + NVIDIA GPU）にSSH接続して実行する手順です。

### 0) 前提
- GPU対応ドライバ/NVIDIA Container Toolkit などは各クラウドの手順に従ってセットアップ
- セキュリティグループ/ファイアウォールで必要なポート（例: 22/6333）を開放

### 1) 接続
```bash
# 例: 固定IP  のGPU VMに接続
ssh ubuntu@<your IP> -i ~/.ssh/your_key.pem
```

### 2) 必要パッケージ
```bash
sudo apt-get update
sudo apt-get install -y git python3-venv build-essential
```

### 3) リポジトリ配置と仮想環境
```bash
# サーバにコードを配置（ローカルからアップロード例）
# scp -r /Users/ryusei/project/mr_seino/LocalLLMRAG ubuntu@yourCloutGCPserverIP:~/

git clone https://github.com/ryusei2790/LocalLLMRAG.git

#ssh ubuntu@IP
cd ~/LocalLLMRAG
python3 -m venv venv
source venv/bin/activate
```

### 4) 依存インストール（GPU版PyTorch）
`requirements.txt` には `torch==2.4.0` が含まれます。GPUを使う場合はCUDAに合ったPyTorchビルドを上書きインストールしてください。

```bash
# まず通常の依存を入れる
pip install -U pip
pip install -r requirements.txt

pip install qdrant_client
pip install sentence_transformers
pip install pypdf
pip install accelerate
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121





# CUDA 12.1 環境の例（環境に合わせて変更）
# 公式ホイール: https://pytorch.org/get-started/locally/
# pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision torchaudio

# 動作確認
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
PY
```

- `cuda: True` かつ GPU が1つ以上認識されればOK
- 複数GPUのうち使用するものを限定したい場合は `CUDA_VISIBLE_DEVICES=0` などを設定

### 5) モデル/キャッシュの永続化（任意）
サーバ再作成時の再ダウンロードを避けるため、Hugging Faceのキャッシュを永続ディレクトリに設定可能です。
```bash
mkdir -p ~/hf-cache
export HF_HOME=~/hf-cache
export HF_HUB_CACHE=~/hf-cache
export TRANSFORMERS_CACHE=~/hf-cache
# 必要に応じ .bashrc に追記
```

### 6) Qdrant の配置方針
- 推奨: GPUサーバ上でQdrantも同居させる
```bash
# サーバ上でQdrant起動（永続ボリュームをローカルディレクトリに割当）
sudo docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $HOME/qdrant_storage:/qdrant/storage qdrant/qdrant:latest

```
- 既存ローカルPCのQdrantを使う場合（逆トンネル）:
  - サーバ→ローカルへ 6333 を転送して、サーバ側から `127.0.0.1:6333` でローカルQdrantに届くようにします
```bash
# ローカルPC側で実行（203.0.113.10 はサーバ）

```
  - `config.py` の `QdrantCfg.host` を `127.0.0.1` のままでOK（サーバ側プロセス視点でローカル転送先を参照）

### 7) ドキュメント投入と実行
```bash
# docs をサーバへコピー（ローカル→サーバ）
scp -r /Users/ryusei/project/mr_seino/LocalLLMRAG/docs ubuntu@203.0.113.10:~/LocalLLMRAG/

# 取り込み（サーバ側）
cd ~/LocalLLMRAG
source venv/bin/activate
python ingest.py

# 推論（サーバ側）
python query.py
```

### 8) モデルパス/精度の調整
- `config.py` の `LLMCfg.model_path` をサーバ上のパス（またはHF名）に変更
- メモリに応じて `LLMCfg.dtype` を `float16`/`bfloat16` に変更
- 生成長が大きい場合は `max_new_tokens` を下げるとメモリ削減
