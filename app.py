# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from config import EmbeddingCfg, QdrantCfg, ChunkCfg
from ingest import (
    load_text_from_file,
    ensure_collection,
    embedder,
    upsert_chunks
)
from utils_chunk import greedy_chunk_by_tokens
from query import (
    load_embedder,
    load_llm,
    search,
    build_prompt,
    chat
)

app = Flask(__name__)

# 設定
EMB = EmbeddingCfg()
QDR = QdrantCfg()
CH = ChunkCfg()

# アップロード許可する拡張子
ALLOWED_EXTENSIONS = {'txt', 'md', 'pdf', 'json'}

# LLMとEmbedderをグローバルで保持（初回ロード後は再利用）
_embedder_cache = None
_llm_cache = None
_tokenizer_cache = None

def get_cached_embedder():
    """埋め込みモデルをキャッシュして再利用"""
    global _embedder_cache
    if _embedder_cache is None:
        print("[INFO] Loading embedder model...")
        _embedder_cache = load_embedder()
    return _embedder_cache

def get_cached_llm():
    """LLMとトークナイザーをキャッシュして再利用"""
    global _llm_cache, _tokenizer_cache
    if _llm_cache is None or _tokenizer_cache is None:
        print("[INFO] Loading LLM model (this may take a while)...")
        _tokenizer_cache, _llm_cache = load_llm()
    return _tokenizer_cache, _llm_cache

def allowed_file(filename: str) -> bool:
    """ファイル拡張子が許可されているかチェック"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file_to_chunks(filepath: str, filename: str) -> List[Dict]:
    """ファイルをチャンクに分割してメタデータを付与"""
    text = load_text_from_file(filepath)
    if not text.strip():
        return []
    
    chunks = greedy_chunk_by_tokens(
        text,
        target_tokens=CH.target_tokens,
        overlap_tokens=CH.overlap_tokens,
        min_chars=CH.min_chars,
    )
    
    all_chunks = []
    for i, ch in enumerate(chunks):
        all_chunks.append({
            "text": ch,
            "source": filename,
            "chunk_id": i
        })
    
    return all_chunks

@app.route('/embedd', methods=['POST'])
def embedd_files():
    """
    PDFやテキストファイルを受け取り、ベクトル化してQdrantに保存

    受理するリクエスト:
      - files: アップロードするファイル（複数可）
      - file:  単一ファイル（後方互換）

    レスポンス:
      - success, message, processed_files, total_chunks など
    """
    # 受信ログ（デバッグ用）
    print("Content-Type:", request.content_type)
    print("Headers:", dict(request.headers))
    print("Form keys:", list(request.form.keys()))
    print("Files keys:", list(request.files.keys()))

    try:
        # --- フィールド名の互換吸収（'files' 優先、次に 'file'） ---
        incoming_files = []
        if 'files' in request.files:
            incoming_files = request.files.getlist('files')
        elif 'file' in request.files:
            single = request.files.get('file')
            if single:
                incoming_files = [single]

        if not incoming_files:
            return jsonify({
                'success': False,
                'message': "ファイルが送信されていません（期待するフィールド名: 'files' または 'file'）。",
                'debug': {
                    'files_keys': list(request.files.keys()),
                    'form_keys': list(request.form.keys()),
                    'content_type': request.content_type
                }
            }), 400

        # --- 空ファイルや無名ファイルを除外 ---
        files = [f for f in incoming_files if f and getattr(f, 'filename', '').strip() != '']
        if not files:
            return jsonify({
                'success': False,
                'message': 'ファイルが選択されていません（filenameが空）。'
            }), 400

        # Qdrantクライアントと埋め込みモデルの初期化
        client = QdrantClient(host=QDR.host, port=QDR.port)
        model = embedder()  # 既存の関数名に合わせています
        dim = model.encode(["dim_check"], normalize_embeddings=EMB.normalize).shape[-1]
        ensure_collection(client, dim, QDR.collection)

        all_chunks = []
        processed_files = []

        # 各ファイルを処理
        for file in files:
            filename = secure_filename(file.filename)
            if not allowed_file(filename):
                print(f"[SKIP] 非対応拡張子: {filename}")
                continue

            # 一時ファイルとして保存
            suffix = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            try:
                # チャンクに分割
                chunks = process_file_to_chunks(tmp_path, filename)
                all_chunks.extend(chunks)
                processed_files.append(filename)
                print(f"[EMBEDD] {filename} -> {len(chunks)} chunks")
            finally:
                # 一時ファイルを削除
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception as _:
                    # 失敗しても致命的ではないのでログのみ
                    print(f"[WARN] 一時ファイル削除に失敗: {tmp_path}")

        if not all_chunks:
            return jsonify({
                'success': False,
                'message': '処理可能なファイルがありませんでした（拡張子/内容を確認してください）。',
                'debug': {
                    'received_filenames': [secure_filename(f.filename) for f in files]
                }
            }), 400

        # Qdrantにアップサート
        upsert_chunks(client, QDR.collection, model, all_chunks)

        return jsonify({
            'success': True,
            'message': 'ファイルの埋め込みが完了しました',
            'processed_files': len(processed_files),
            'file_names': processed_files,
            'total_chunks': len(all_chunks)
        }), 200

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/question', methods=['POST'])
def answer_question():
    """
    質問を受け取り、RAGを使って回答を生成
    
    リクエスト:
        - question: 質問文（必須）
        - top_k: 検索する関連文書数（任意、デフォルト5）
        - source_filter: 特定のソースファイルでフィルタリング（任意）
    
    レスポンス:
        - success: 成功フラグ
        - question: 元の質問
        - answer: LLMが生成した回答
        - contexts: 参照したコンテキスト情報
    """
    try:
        # リクエストボディから質問を取得
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'リクエストボディが必要です'
            }), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({
                'success': False,
                'message': '質問（question）が空です'
            }), 400
        
        # オプションパラメータ
        top_k = data.get('top_k', 5)
        source_filter = data.get('source_filter', None)
        
        print(f"[QUESTION] {question}")
        
        # Qdrantクライアントの初期化
        client = QdrantClient(host=QDR.host, port=QDR.port)
        
        # 埋め込みモデルをロード（キャッシュ利用）
        emb_model = get_cached_embedder()
        
        # ベクトル検索でコンテキストを取得
        print("[INFO] Searching for relevant contexts...")
        hits = search(
            client=client,
            emb_model=emb_model,
            query=question,
            top_k=top_k,
            source_filter=source_filter
        )
        
        if not hits:
            return jsonify({
                'success': False,
                'message': '関連する文書が見つかりませんでした。先にファイルを/embeddでアップロードしてください。'
            }), 404
        
        contexts = [payload for _, payload in hits]
        print(f"[INFO] Found {len(contexts)} relevant contexts")
        
        # LLMをロード（キャッシュ利用）
        tokenizer, llm_model = get_cached_llm()
        
        # プロンプトを構築
        print("[INFO] Building prompt...")
        messages = build_prompt(question, contexts, tokenizer, ctx_token_budget=2300)
        
        # LLMで回答生成
        print("[INFO] Generating answer...")
        answer = chat(llm_model, tokenizer, messages)
        
        # レスポンスを整形
        context_info = []
        for i, ctx in enumerate(contexts, 1):
            context_info.append({
                'index': i,
                'source': ctx.get('source', ''),
                'title': ctx.get('title', ''),
                'page': ctx.get('page', ''),
                'chunk_id': ctx.get('chunk_id', ''),
                'text_preview': ctx.get('text', '')[:200] + '...' if len(ctx.get('text', '')) > 200 else ctx.get('text', '')
            })
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'num_contexts': len(contexts),
            'contexts': context_info
        }), 200
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """
    登録されている文書の一覧を取得
    
    レスポンス:
        - success: 成功フラグ
        - documents: 文書情報のリスト
        - total: 総チャンク数
    """
    try:
        client = QdrantClient(host=QDR.host, port=QDR.port)
        
        # コレクションの存在確認
        collections = [c.name for c in client.get_collections().collections]
        if QDR.collection not in collections:
            return jsonify({
                'success': True,
                'documents': [],
                'total_chunks': 0,
                'message': 'コレクションが存在しません。先に/embeddでファイルをアップロードしてください。'
            }), 200
        
        # 全データを取得（sourceでグループ化）
        scroll_result = client.scroll(
            collection_name=QDR.collection,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]
        
        # sourceごとに集計
        doc_dict = {}
        for point in points:
            source = point.payload.get('source', 'unknown')
            if source not in doc_dict:
                doc_dict[source] = {
                    'source': source,
                    'chunk_count': 0,
                    'chunk_ids': []
                }
            doc_dict[source]['chunk_count'] += 1
            doc_dict[source]['chunk_ids'].append(point.payload.get('chunk_id', -1))
        
        documents = list(doc_dict.values())
        
        return jsonify({
            'success': True,
            'documents': documents,
            'total_chunks': len(points),
            'document_count': len(documents)
        }), 200
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/documents/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    """
    特定の文書を削除
    
    パラメータ:
        - filename: 削除する文書のファイル名
    
    レスポンス:
        - success: 成功フラグ
        - deleted_count: 削除したチャンク数
    """
    try:
        client = QdrantClient(host=QDR.host, port=QDR.port)
        
        # コレクションの存在確認
        collections = [c.name for c in client.get_collections().collections]
        if QDR.collection not in collections:
            return jsonify({
                'success': False,
                'message': 'コレクションが存在しません'
            }), 404
        
        # 対象ファイルのポイントIDを取得
        scroll_result = client.scroll(
            collection_name=QDR.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filename))]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False
        )
        
        points = scroll_result[0]
        point_ids = [point.id for point in points]
        
        if not point_ids:
            return jsonify({
                'success': False,
                'message': f'ファイル "{filename}" は見つかりませんでした'
            }), 404
        
        # ポイントを削除
        client.delete(
            collection_name=QDR.collection,
            points_selector=point_ids
        )
        
        print(f"[DELETE] Deleted {len(point_ids)} chunks from '{filename}'")
        
        return jsonify({
            'success': True,
            'message': f'ファイル "{filename}" を削除しました',
            'deleted_count': len(point_ids)
        }), 200
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/reset', methods=['POST'])
def reset_database():
    """
    データベース全体を初期化（全データ削除）
    
    レスポンス:
        - success: 成功フラグ
        - message: メッセージ
    """
    try:
        client = QdrantClient(host=QDR.host, port=QDR.port)
        
        # コレクションの存在確認
        collections = [c.name for c in client.get_collections().collections]
        if QDR.collection in collections:
            # コレクションを削除
            client.delete_collection(collection_name=QDR.collection)
            print(f"[RESET] Collection '{QDR.collection}' deleted")
        
        return jsonify({
            'success': True,
            'message': 'データベースを初期化しました'
        }), 200
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック用エンドポイント"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask RAG API is running'
    }), 200

if __name__ == '__main__':
    # # アプリ起動時にモデルを事前ロード
    # print("[STARTUP] Pre-loading embedder model...")
    # get_cached_embedder()
    # print("[STARTUP] Pre-loading LLM model...")
    # get_cached_llm()
    # print("[STARTUP] All models loaded. Starting server...")
    
    app.run(host='0.0.0.0', port=1234, debug=True)

