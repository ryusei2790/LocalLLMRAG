# Postmanでのテストガイド

このガイドでは、Postmanを使ってFlask RAG APIをテストする方法を説明します。

## 📋 事前準備

1. **Qdrantを起動**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage \
       qdrant/qdrant
   ```

2. **Flaskサーバーを起動**
   ```bash
   source venv/bin/activate
   python app.py
   ```
   
   サーバーは `http://localhost:1234` で起動します。

---

## 🧪 テスト手順

### 1️⃣ ヘルスチェック（GET /health）

最初にサーバーが正常に動作しているか確認します。

**設定:**
- **Method**: `GET`
- **URL**: `http://localhost:1234/health`

**実行手順:**
1. Postmanを開く
2. 新しいリクエストを作成
3. メソッドを `GET` に設定
4. URLに `http://localhost:1234/health` を入力
5. 「Send」をクリック

**期待されるレスポンス:**
```json
{
    "status": "ok",
    "message": "Flask RAG API is running"
}
```

**Status Code**: `200 OK`

---

### 2️⃣ 文書のアップロード（POST /embedd）

テキストファイルをアップロードしてベクトル化します。

**設定:**
- **Method**: `POST`
- **URL**: `http://localhost:1234/embedd`
- **Body**: `form-data`

**実行手順:**

1. **新しいリクエストを作成**
2. **メソッドを `POST` に設定**
3. **URLに `http://localhost:1234/embedd` を入力**
4. **Body タブをクリック**
5. **`form-data` を選択**
6. **ファイルを追加:**
   - Key: `files` （重要: 型を `File` に変更）
   - Value: `sample.txt` を選択（「Select Files」ボタンから）
7. **複数ファイルを追加する場合:**
   - 「+」ボタンで行を追加
   - 同じく Key: `files` で別のファイルを選択
8. **「Send」をクリック**

**Postman での設定画面イメージ:**
```
Body タブ
├─ form-data (選択)
   ├─ KEY: files (type: File) | VALUE: [Select Files] → sample.txt
   └─ KEY: files (type: File) | VALUE: [Select Files] → (別のファイル・任意)
```

**期待されるレスポンス:**
```json
{
    "success": true,
    "message": "ファイルの埋め込みが完了しました",
    "processed_files": 1,
    "file_names": [
        "sample.txt"
    ],
    "total_chunks": 3
}
```

**Status Code**: `200 OK`

**⚠️ トラブルシューティング:**
- キーは必ず `files` （複数形）
- 型を `Text` から `File` に変更するのを忘れずに
- 対応形式: `.txt`, `.md`, `.pdf`, `.json`

---

### 3️⃣ 質問の送信（POST /question）

アップロードした文書に基づいて質問します。

**設定:**
- **Method**: `POST`
- **URL**: `http://localhost:1234/question`
- **Headers**: `Content-Type: application/json`
- **Body**: `raw` (JSON)

**実行手順:**

1. **新しいリクエストを作成**
2. **メソッドを `POST` に設定**
3. **URLに `http://localhost:1234/question` を入力**
4. **Headers タブをクリック**
   - Key: `Content-Type`
   - Value: `application/json`
5. **Body タブをクリック**
6. **`raw` を選択し、右側のドロップダウンから `JSON` を選択**
7. **以下のJSONを入力:**

#### 📝 テストケース 1: 基本的な質問

```json
{
    "question": "このドキュメントの主な内容は何ですか？",
    "top_k": 5
}
```

#### 📝 テストケース 2: 詳細な質問

```json
{
    "question": "重要なポイントを教えてください",
    "top_k": 3
}
```

#### 📝 テストケース 3: 要約を求める質問

```json
{
    "question": "内容を簡潔に要約してください",
    "top_k": 5
}
```

#### 📝 テストケース 4: 特定ファイルからの検索

```json
{
    "question": "このファイルに含まれる情報を教えてください",
    "top_k": 5,
    "source_filter": "sample.txt"
}
```

8. **「Send」をクリック**

**期待されるレスポンス:**
```json
{
    "success": true,
    "question": "このドキュメントの主な内容は何ですか？",
    "answer": "このドキュメントの主な内容は以下の通りです：\n\n- 重要なポイント1\n- 重要なポイント2\n- 重要なポイント3\n\n参照: [1]",
    "num_contexts": 3,
    "contexts": [
        {
            "index": 1,
            "source": "sample.txt",
            "title": "",
            "page": "",
            "chunk_id": 0,
            "text_preview": "ドキュメントの冒頭部分のテキストがここに表示されます..."
        }
    ]
}
```

**Status Code**: `200 OK`

---

## 📊 Postman Collection（インポート用JSON）

以下のJSONをコピーして、Postmanの「Import」機能で読み込めます。

```json
{
    "info": {
        "name": "LocalLLMRAG API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": [
        {
            "name": "Health Check",
            "request": {
                "method": "GET",
                "header": [],
                "url": {
                    "raw": "http://localhost:1234/health",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "1234",
                    "path": ["health"]
                }
            }
        },
        {
            "name": "Upload Documents",
            "request": {
                "method": "POST",
                "header": [],
                "body": {
                    "mode": "formdata",
                    "formdata": [
                        {
                            "key": "files",
                            "type": "file",
                            "src": "/path/to/sample.txt"
                        }
                    ]
                },
                "url": {
                    "raw": "http://localhost:1234/embedd",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "1234",
                    "path": ["embedd"]
                }
            }
        },
        {
            "name": "Ask Question - Basic",
            "request": {
                "method": "POST",
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    }
                ],
                "body": {
                    "mode": "raw",
                    "raw": "{\n    \"question\": \"このドキュメントの主な内容は何ですか？\",\n    \"top_k\": 5\n}"
                },
                "url": {
                    "raw": "http://localhost:1234/question",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "1234",
                    "path": ["question"]
                }
            }
        },
        {
            "name": "Ask Question - Technical",
            "request": {
                "method": "POST",
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    }
                ],
                "body": {
                    "mode": "raw",
                    "raw": "{\n    \"question\": \"重要なポイントを教えてください\",\n    \"top_k\": 3\n}"
                },
                "url": {
                    "raw": "http://localhost:1234/question",
                    "protocol": "http",
                    "host": ["localhost"],
                    "port": "1234",
                    "path": ["question"]
                }
            }
        }
    ]
}
```

**インポート手順:**
1. Postmanを開く
2. 左上の「Import」ボタンをクリック
3. 「Raw text」タブを選択
4. 上記のJSONをペースト
5. 「Continue」→「Import」をクリック

---

## 🎯 テスト推奨フロー

1. **ヘルスチェック** → サーバー起動確認
2. **文書アップロード** → データ準備
3. **質問送信（簡単な質問）** → 基本動作確認
4. **質問送信（複雑な質問）** → 精度確認
5. **質問送信（存在しない情報）** → エラーハンドリング確認

---

## 🐛 エラー時のチェックポイント

### 502 Bad Gateway / Connection Refused
- Flaskサーバーが起動しているか確認
- `python app.py` を実行

### 404 Not Found (質問時)
- 先に `/embedd` で文書をアップロードしているか確認
- Qdrantが起動しているか確認

### 400 Bad Request
- リクエストボディの形式を確認
- JSONの構文エラーがないか確認
- `/embedd` の場合、キーが `files` になっているか確認

### 500 Internal Server Error
- Flaskのコンソールログを確認
- LLMモデルが正しく配置されているか確認（`Qwen/` ディレクトリ）

---

## 💡 便利な機能

### Environment変数の設定

複数の環境（開発、本番など）を管理する場合：

1. Postmanの右上の歯車アイコン → 「Manage Environments」
2. 「Add」をクリック
3. 変数を追加:
   - Variable: `base_url`
   - Initial Value: `http://localhost:1234`
4. URLを `{{base_url}}/health` のように変更

### Tests スクリプト

レスポンスの自動検証：

```javascript
// Testsタブに追加
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has success field", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('success');
});
```

これでPostmanでの完全なテストが可能です！

