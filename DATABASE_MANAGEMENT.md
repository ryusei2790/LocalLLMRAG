# ベクトルデータベース管理ガイド

LocalLLMRAGのベクトルデータベースを管理するためのAPIエンドポイントの使い方を説明します。

## 📋 目次

- [ファイルのアップロード（埋め込み）](#ファイルのアップロード埋め込み)
- [質問の送信](#質問の送信)
- [登録済み文書の一覧取得](#登録済み文書の一覧取得)
- [特定ファイルの削除](#特定ファイルの削除)
- [データベース全体の初期化](#データベース全体の初期化)
- [ファイルの更新方法](#ファイルの更新方法)
- [実用的な使用例](#実用的な使用例)

---

## 📤 ファイルのアップロード（埋め込み）

### `POST /embedd`

PDFやテキストファイルをアップロードして、ベクトル化してデータベースに保存します。

**対応ファイル形式:**
- `.txt` - テキストファイル
- `.md` - Markdownファイル
- `.pdf` - PDFファイル
- `.json` - JSONファイル

⚠️ **ファイル名に関する重要な注意事項:**

curlで日本語ファイル名をアップロードする際は、以下のいずれかの方法を使用してください：

1. **ファイル名を明示的に指定（推奨）:**
   ```bash
   curl -X POST http://localhost:1234/embedd \
     -F "files=@docs/日本語ファイル名.pdf;filename=document.pdf"
   ```

2. **事前にファイル名を英数字にリネーム:**
   ```bash
   mv "docs/日本語ファイル名.pdf" "docs/document.pdf"
   curl -X POST http://localhost:1234/embedd \
     -F "files=@docs/document.pdf"
   ```

3. **Pythonスクリプトを使用（最も確実）:**
   ```python
   import requests
   
   with open('docs/日本語ファイル名.pdf', 'rb') as f:
       files = {'files': ('document.pdf', f, 'application/pdf')}
       response = requests.post('http://localhost:1234/embedd', files=files)
       print(response.json())
   ```

**ファイル名のベストプラクティス:**
- ✅ 推奨: `document_plan.pdf`, `faq-2024.pdf`, `user-manual-v1.pdf`
- ❌ 非推奨: 日本語、スペース、特殊文字を含むファイル名

**リクエスト例（curl）:**

```bash
# 単一ファイルのアップロード
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/sample.pdf"

# 複数ファイルの同時アップロード
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/document1.pdf" \
  -F "files=@docs/document2.txt" \
  -F "files=@docs/faq.json"
```

**リクエスト例（Python）:**

```python
import requests

# 単一ファイル
url = "http://localhost:1234/embedd"
files = {'files': open('docs/sample.pdf', 'rb')}
response = requests.post(url, files=files)
print(response.json())

# 複数ファイル
files = [
    ('files', open('docs/document1.pdf', 'rb')),
    ('files', open('docs/document2.txt', 'rb')),
    ('files', open('docs/faq.json', 'rb'))
]
response = requests.post(url, files=files)
result = response.json()

print(f"処理成功: {result['success']}")
print(f"処理ファイル数: {result['processed_files']}")
print(f"生成チャンク数: {result['total_chunks']}")
```

**レスポンス例:**

```json
{
  "success": true,
  "message": "ファイルの埋め込みが完了しました",
  "processed_files": 1,
  "file_names": ["sample.pdf"],
  "total_chunks": 45
}
```

**エラーレスポンス例:**

```json
{
  "success": false,
  "message": "処理可能なファイルがありませんでした"
}
```

---

## 💬 質問の送信

### `POST /question`

アップロードした文書に基づいて、質問に対する回答を生成します。

**リクエスト例（curl）:**

```bash
# 基本的な質問
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "このPDFの主な内容を要約してください。",
    "top_k": 5
  }'

# 特定のファイルに限定して検索
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "このドキュメントの主な内容は？",
    "top_k": 3,
    "source_filter": "sample.pdf"
  }'

# すべての文書から検索（source_filterを空にする）
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "内容を要約してください。",
    "top_k": 5,
    "source_filter": ""
  }'
```

**リクエスト例（Python）:**

```python
import requests

url = "http://localhost:1234/question"

# 基本的な質問
data = {
    "question": "このPDFの主な内容を要約してください。",
    "top_k": 5
}
response = requests.post(url, json=data)
result = response.json()

print(f"質問: {result['question']}")
print(f"回答: {result['answer']}")
print(f"参照文書数: {result['num_contexts']}")

# 特定ファイルに限定
data = {
    "question": "このドキュメントの要点を教えてください",
    "top_k": 3,
    "source_filter": "sample.pdf"
}
response = requests.post(url, json=data)
result = response.json()

if result['success']:
    print(f"\n回答:\n{result['answer']}")
    print(f"\n参照した文書:")
    for ctx in result['contexts']:
        print(f"  [{ctx['index']}] {ctx['source']} (chunk {ctx['chunk_id']})")
else:
    print(f"エラー: {result['message']}")
```

**リクエストパラメータ:**

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `question` | string | ✅ | - | 質問文 |
| `top_k` | integer | ❌ | 5 | 検索する関連文書数（1-20推奨） |
| `source_filter` | string | ❌ | null | 特定ファイルに限定（ファイル名を指定） |

**レスポンス例:**

```json
{
  "success": true,
  "question": "このPDFの主な内容を要約してください。",
  "answer": "このPDFの主な内容は以下の通りです：\n\n- 重要なポイント1\n- 重要なポイント2\n- 重要なポイント3\n\n参照: [1],[2],[3]",
  "num_contexts": 3,
  "contexts": [
    {
      "index": 1,
      "source": "sample.pdf",
      "title": "",
      "page": "",
      "chunk_id": 0,
      "text_preview": "ドキュメントの冒頭部分のテキストがここに表示されます..."
    }
  ]
}
```

**エラーレスポンス例（文書が見つからない）:**

```json
{
  "success": false,
  "message": "関連する文書が見つかりませんでした。先にファイルを/embeddでアップロードしてください。"
}
```

**質問のベストプラクティス:**

1. **具体的に質問する**
   - ❌ 悪い例: "これは何？"
   - ✅ 良い例: "予約のキャンセル手数料について教えてください"

2. **キーワードを含める**
   - ❌ 悪い例: "どうすればいい？"
   - ✅ 良い例: "航空券の予約変更の手順を教えてください"

3. **top_kの調整**
   - 詳しい回答が欲しい: `top_k: 7-10`
   - 簡潔な回答が欲しい: `top_k: 3-5`
   - 広範囲の情報が欲しい: `top_k: 10-15`

4. **source_filterの活用**
   - 特定の文書に限定したい場合に使用
   - 複数文書から横断的に検索したい場合は指定しない

---

## 📄 登録済み文書の一覧取得

### `GET /documents`

現在ベクトルデータベースに登録されているファイルの一覧を取得します。

**リクエスト例（curl）:**

```bash
curl http://localhost:1234/documents
```

**リクエスト例（Python）:**

```python
import requests

response = requests.get("http://localhost:1234/documents")
result = response.json()

print(f"登録文書数: {result['document_count']}")
print(f"総チャンク数: {result['total_chunks']}")

for doc in result['documents']:
    print(f"  - {doc['source']}: {doc['chunk_count']}チャンク")
```

**レスポンス例:**

```json
{
  "success": true,
  "documents": [
    {
      "source": "document_a.pdf",
      "chunk_count": 45,
      "chunk_ids": [0, 1, 2, 3, ...]
    },
    {
      "source": "document_b.txt",
      "chunk_count": 3,
      "chunk_ids": [0, 1, 2]
    }
  ],
  "total_chunks": 48,
  "document_count": 2
}
```

---

## 🗑️ 特定ファイルの削除

### `DELETE /documents/<filename>`

特定のファイルをベクトルデータベースから削除します。

**リクエスト例（curl）:**

```bash
# document_a.pdfを削除
curl -X DELETE http://localhost:1234/documents/document_a.pdf

# サブディレクトリのファイルも削除可能
curl -X DELETE http://localhost:1234/documents/docs/document_b.txt
```

**リクエスト例（Python）:**

```python
import requests

filename = "document_a.pdf"
response = requests.delete(f"http://localhost:1234/documents/{filename}")
result = response.json()

if result['success']:
    print(f"削除成功: {result['deleted_count']}チャンクを削除")
else:
    print(f"削除失敗: {result['message']}")
```

**レスポンス例:**

```json
{
  "success": true,
  "message": "ファイル \"document_a.pdf\" を削除しました",
  "deleted_count": 45
}
```

**エラーレスポンス例（ファイルが見つからない場合）:**

```json
{
  "success": false,
  "message": "ファイル \"nonexistent.pdf\" は見つかりませんでした"
}
```

---

## 🔄 データベース全体の初期化

### `POST /reset`

ベクトルデータベース全体を初期化（全データ削除）します。

⚠️ **警告**: このエンドポイントは全データを削除します。本番環境では注意して使用してください。

**リクエスト例（curl）:**

```bash
curl -X POST http://localhost:1234/reset
```

**リクエスト例（Python）:**

```python
import requests

response = requests.post("http://localhost:1234/reset")
result = response.json()

if result['success']:
    print("データベースを初期化しました")
else:
    print(f"エラー: {result['message']}")
```

**レスポンス例:**

```json
{
  "success": true,
  "message": "データベースを初期化しました"
}
```

---

## 🔄 ファイルの更新方法

特定のファイルを更新したい場合は、以下の手順で行います：

### 方法1: 削除 → 再アップロード（推奨）

```bash
# 1. 古いファイルを削除
curl -X DELETE http://localhost:1234/documents/greenheardRAG.pdf

# 2. 新しいファイルをアップロード
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/greenheardRAG.pdf"
```

### 方法2: Pythonスクリプトで自動化

```python
import requests

def update_document(filepath: str):
    """ファイルを更新（削除→再アップロード）"""
    base_url = "http://localhost:1234"
    filename = filepath.split('/')[-1]
    
    # 1. 既存ファイルを削除
    delete_response = requests.delete(f"{base_url}/documents/{filename}")
    if delete_response.json()['success']:
        print(f"削除成功: {filename}")
    
    # 2. 新しいファイルをアップロード
    with open(filepath, 'rb') as f:
        files = {'files': f}
        upload_response = requests.post(f"{base_url}/embedd", files=files)
    
    if upload_response.json()['success']:
        print(f"アップロード成功: {filename}")
        return True
    else:
        print(f"エラー: {upload_response.json()['message']}")
        return False

# 使用例
update_document("docs/sample_document.pdf")
```

---

## 🎯 実用的な使用例

### シナリオ1: 初回セットアップ（ファイルアップロード→質問）

新しいプロジェクトでRAGシステムをセットアップする場合：

```bash
# 1. サーバーを起動
python app.py

# 2. 複数のドキュメントをアップロード
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/user_manual.pdf" \
  -F "files=@docs/faq.txt" \
  -F "files=@docs/policy.md"

# 3. 登録状況を確認
curl http://localhost:1234/documents

# 4. 質問してみる
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ユーザーマニュアルの主な内容を教えてください",
    "top_k": 5
  }'
```

### シナリオ2: データベースのクリーンアップ

不要なファイルを削除してデータベースを整理：

```bash
# 1. 現在の文書一覧を確認
curl http://localhost:1234/documents

# 2. 不要なファイルを削除
curl -X DELETE http://localhost:1234/documents/old_document.pdf
curl -X DELETE http://localhost:1234/documents/test.txt

# 3. 削除後の状態を確認
curl http://localhost:1234/documents
```

### シナリオ2: 特定ファイルからの質問と更新

特定のPDFについて質問して、その後内容を更新する：

```bash
# 1. 特定ファイルに関する質問
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "このドキュメントの主な内容は？",
    "top_k": 3,
    "source_filter": "document_v1.pdf"
  }'

# 2. PDFの内容が古くなったので更新
curl -X DELETE http://localhost:1234/documents/document_v1.pdf
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/document_v2.pdf"

# 3. 同じ質問で最新情報を取得
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "このドキュメントの主な内容は？",
    "top_k": 3,
    "source_filter": "document_v2.pdf"
  }'
```

### シナリオ3: 複数文書からの横断検索

```bash
# 1. 複数の文書をアップロード
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/manual_ja.pdf" \
  -F "files=@docs/manual_en.pdf" \
  -F "files=@docs/faq.txt"

# 2. すべての文書から検索（source_filterなし）
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "サポートの連絡先を教えてください",
    "top_k": 7
  }'

# 3. 日本語マニュアルのみから検索
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "サポートの連絡先を教えてください",
    "top_k": 5,
    "source_filter": "manual_ja.pdf"
  }'
```

### シナリオ4: 完全リセット

すべてのデータを削除して最初からやり直す：

```bash
# 1. データベースを初期化
curl -X POST http://localhost:1234/reset

# 2. 新しいファイルをアップロード
curl -X POST http://localhost:1234/embedd \
  -F "files=@docs/document1.pdf" \
  -F "files=@docs/document2.txt"

# 3. 動作確認
curl -X POST http://localhost:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "document1の内容を要約してください",
    "top_k": 5
  }'
```

### シナリオ5: ファイルの差分更新

特定のファイルだけを更新：

```python
import requests

base_url = "http://localhost:1234"

# 1. 更新が必要なファイルのリスト
files_to_update = [
    "docs/updated_document.pdf",
    "docs/revised_policy.txt"
]

for filepath in files_to_update:
    filename = filepath.split('/')[-1]
    
    # 既存ファイルを削除
    requests.delete(f"{base_url}/documents/{filename}")
    
    # 新しいファイルをアップロード
    with open(filepath, 'rb') as f:
        files = {'files': f}
        response = requests.post(f"{base_url}/embedd", files=files)
        print(f"{filename}: {response.json()['message']}")
```

---

## 🔧 トラブルシューティング

### ファイル名に特殊文字が含まれる場合

URLエンコードが必要です：

```python
from urllib.parse import quote
import requests

filename = "ファイル 名.pdf"
encoded_filename = quote(filename)
response = requests.delete(f"http://localhost:1234/documents/{encoded_filename}")
```

### 大量のファイルを一括削除

```python
import requests

base_url = "http://localhost:1234"

# すべてのファイルを取得
response = requests.get(f"{base_url}/documents")
documents = response.json()['documents']

# 特定の条件でフィルタリング（例：PDFのみ）
pdf_files = [doc['source'] for doc in documents if doc['source'].endswith('.pdf')]

# 一括削除
for filename in pdf_files:
    response = requests.delete(f"{base_url}/documents/{filename}")
    print(f"Deleted: {filename}")
```

### データベースの状態確認

```python
import requests

response = requests.get("http://localhost:1234/documents")
data = response.json()

print(f"=== データベース状態 ===")
print(f"登録文書数: {data['document_count']}")
print(f"総チャンク数: {data['total_chunks']}")
print(f"\n文書一覧:")
for doc in data['documents']:
    print(f"  📄 {doc['source']}: {doc['chunk_count']}チャンク")
```

---

## 📊 API一覧表

| エンドポイント | メソッド | 説明 | 用途 |
|---------------|---------|------|------|
| `/documents` | GET | 登録済み文書一覧 | 現在の状態確認 |
| `/documents/<filename>` | DELETE | 特定ファイル削除 | ファイル更新・削除 |
| `/reset` | POST | DB全体初期化 | 完全リセット |
| `/embedd` | POST | ファイルアップロード | 新規登録・再登録 |
| `/question` | POST | 質問応答 | RAG検索 |
| `/health` | GET | ヘルスチェック | サーバー確認 |

---

## 🔐 セキュリティに関する注意

本番環境では以下の対策を検討してください：

1. **認証・認可**: `/reset`や`DELETE`エンドポイントにアクセス制限を設ける
2. **ログ記録**: すべての削除操作をログに記録
3. **バックアップ**: 定期的にQdrantのデータをバックアップ
4. **確認ダイアログ**: フロントエンドで削除前に確認を求める

---

これで、ベクトルデータベースの完全な管理が可能になりました！🎉

✅ 判断可能な質問（回答がPDF内に存在する）

この研究のテーマは何ですか？

使用予定のAIモデルにはどのようなものが挙げられていますか？

アンケートの形式としてどのような2種類を比較しようとしていますか？

研究で利用予定のプラットフォームとして挙げられているものは何ですか？

今後の研究計画の中で、アンケート作成を行う時期はいつとされていますか？

❌ 判断不可能な質問（PDF内からは答えられない）

研究で使用するAIモデルのトークン数や学習データ量はどれくらいですか？

アンケートの対象人数は具体的に何人を想定していますか？

クラスター分析の際に使用する統計手法の名称は何ですか？

研究で使用するAIの学習環境（GPUやメモリなど）はどのような構成ですか？

本研究で想定している「最も有効なAIモデル」はどれですか？