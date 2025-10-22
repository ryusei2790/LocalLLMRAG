# Flask RAG API - curlコマンド集

## 📋 エンドポイント一覧

### 1. POST /question - 質問応答
質問を送信してRAGシステムで回答を生成します。

#### 基本的な質問
```bash
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "予約をする時に必要なものは何ですか？"
  }'
```

#### 特定のファイルから検索
```bash
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "片道航空券はありますか？",
    "top_k": 3,
    "source_filter": "greenheardRAG.pdf"
  }'
```

#### 検索数を指定した質問
```bash
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "国際線で3歳の子どもの運賃はいくらになりますか？",
    "top_k": 5
  }'
```

#### 複雑な質問
```bash
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "予約の変更やキャンセルに関する費用について教えてください。",
    "top_k": 5,
    "source_filter": ""
  }'
```

---

### 2. GET /documents - 文書一覧
登録されている文書の一覧を取得します。

```bash
curl -X GET http://127.0.0.1:1234/documents
```

---

### 3. DELETE /documents/<filename> - 文書削除
特定の文書を削除します。

#### 特定ファイルの削除
```bash
curl -X DELETE http://127.0.0.1:1234/documents/greenheardRAG.pdf
```

#### 別のファイルの削除
```bash
curl -X DELETE http://127.0.0.1:1234/documents/FAQ1.json
```

---

### 4. POST /reset - データベース初期化
すべての文書データを削除してデータベースを初期化します。

```bash
curl -X POST http://127.0.0.1:1234/reset
```

---

### 5. GET /health - ヘルスチェック
APIサーバーの状態を確認します。

```bash
curl -X GET http://127.0.0.1:1234/health
```

---

## 🧪 テスト用の質問例

### 航空予約に関する質問
```bash
# 予約に必要なもの
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "予約をする時に必要なものは何ですか？"}'

# 予約変更の費用
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "発券後に予約を変更した場合、費用はかかりますか？"}'

# 子どもの運賃
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "国際線で3歳の子どもの運賃はいくらになりますか？"}'

# 片道航空券
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "片道航空券はありますか？"}'

# キャンセル費用
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "予約のキャンセルは発券前でも費用がかかりますか？"}'

# 超過手荷物
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "超過手荷物がある場合、事前に支払いはできますか？"}'
```

### 知識にない質問（回答不可能な質問）
```bash
# アップグレード
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "マイルを使って予約した航空券でもアップグレードは可能ですか？"}'

# 台風欠航
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "台風で欠航になった場合、返金や振替はどうなりますか？"}'

# ホテル朝食
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "ホテルの朝食でベジタリアン対応はできますか？"}'

# 機内Wi-Fi
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "機内Wi-Fiの料金は含まれていますか？"}'
```

---

## 🔧 パラメータ説明

### /question エンドポイントのパラメータ

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|-----------|----|----|---------|-----|
| `question` | string | ✅ | - | 質問文 |
| `top_k` | integer | ❌ | 5 | 検索する関連文書数 |
| `source_filter` | string | ❌ | null | 特定のソースファイルでフィルタリング |

### レスポンス形式

#### 成功時
```json
{
  "success": true,
  "question": "質問文",
  "answer": "LLMが生成した回答",
  "num_contexts": 3,
  "contexts": [
    {
      "index": 1,
      "source": "greenheardRAG.pdf",
      "title": "",
      "page": "",
      "chunk_id": 0,
      "text_preview": "文書の一部..."
    }
  ]
}
```

#### エラー時
```json
{
  "success": false,
  "message": "エラーメッセージ"
}
```

---

## 🚀 使用例

### 1. サーバー起動
```bash
cd /Users/ryusei/project/mr_seino/LocalLLMRAG
source venv/bin/activate
python app.py
```

### 2. ヘルスチェック
```bash
curl -X GET http://127.0.0.1:1234/health
```

### 3. 文書一覧確認
```bash
curl -X GET http://127.0.0.1:1234/documents
```

### 4. 質問実行
```bash
curl -X POST http://127.0.0.1:1234/question \
  -H "Content-Type: application/json" \
  -d '{"question": "予約をする時に必要なものは何ですか？"}'
```

### 5. データベースリセット
```bash
curl -X POST http://127.0.0.1:1234/reset
```

---

## 📝 注意事項

1. **サーバー起動**: curlコマンドを実行する前に、Flaskアプリが起動していることを確認してください
2. **ファイルアップロード**: 質問する前に、`/embedd`エンドポイントでファイルをアップロードしてください
3. **エンコーディング**: 日本語の質問は適切にエンコードされます
4. **タイムアウト**: 大きなファイルや複雑な質問の場合、応答に時間がかかる場合があります
