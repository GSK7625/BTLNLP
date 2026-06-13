# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import re
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template

GLOBAL_CORPUS_RAW = []
GLOBAL_CORPUS_TOKENIZED = []
GLOBAL_BM25 = None

# Setup paths so we can import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import metrics and components from src or demo
from src.utils.metrics import normalize_answer, compute_f1, compute_exact
from demo import ExtractiveReader

app = Flask(__name__)

# Global cache for heavy models
MODEL_CACHE = {}

def get_model_reader(model_key: str):
    """Lazy load the reader models to save memory and startup time."""
    if model_key not in MODEL_CACHE:
        if model_key == 'pretrained':
            # Pretrained XLM-RoBERTa on SQuADv2
            model_path = 'deepset/xlm-roberta-base-squad2'
            MODEL_CACHE[model_key] = ExtractiveReader(model_path)
        elif model_key == 'finetuned':
            # Fine-tuned on ViSpanExtractQA
            model_path = 'models/xlmroberta_finetuned'
            if not os.path.exists(model_path) or not os.listdir(model_path):
                raise ValueError("Fine-tuned model folder not found or empty. Please run training first.")
            MODEL_CACHE[model_key] = ExtractiveReader(model_path)
    return MODEL_CACHE[model_key]


def run_bm25_selector(question: str, context: str) -> dict:
    """Simple BM25 sentence overlap baseline reader."""
    t0 = time.time()
    
    # Simple word tokenization
    def tokenize(text: str):
        return text.lower().split()
        
    # Split sentences by common boundaries
    sentences = re.split(r'(?<=[.?!，。])\s+|(?<=,)\s+(?=[A-ZÁẮẶẤẦ])', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        latency = (time.time() - t0) * 1000
        return {
            'answer': context.strip(),
            'confidence': 1.0,
            'char_start': 0,
            'char_end': len(context),
            'latency_ms': round(latency, 1)
        }
        
    q_tokens = set(tokenize(question))
    best_sentence = sentences[0]
    best_overlap = -1
    
    for sent in sentences:
        s_tokens = set(tokenize(sent))
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent
            
    # Find the character boundary in context
    char_start = context.find(best_sentence)
    if char_start == -1:
        char_start = 0
        char_end = len(context)
    else:
        char_end = char_start + len(best_sentence)
        
    latency = (time.time() - t0) * 1000
    
    return {
        'answer': best_sentence,
        'confidence': 1.0, # BM25 overlap score has no standardized confidence, set to 1.0
        'char_start': char_start,
        'char_end': char_end,
        'latency_ms': round(latency, 1)
    }


def load_preloaded_examples():
    """Load demo examples plus a few short samples from test_clean.json."""
    examples = [
        {
            'id': 'ex_1',
            'question': 'Ai là chủ tịch tập đoàn Alibaba?',
            'context': 'Dư luận đang hết sức ngóng chờ sự kiện Chủ tịch tập đoàn thương mại điện tử Alibaba - Jack Ma (Mã Vân) đến Việt Nam, vậy Jack Ma là ai, Alibaba là tập đoàn thế nào mà lại có sức ảnh hưởng đến như vậy?',
            'gold': 'Jack Ma'
        },
        {
            'id': 'ex_2',
            'question': 'Thuật ngữ Big Bang do ai đề xuất?',
            'context': '1949, Fred Hoyle, một nhà toán học và thiên văn học nổi tiếng người Anh, trong một lần trả lời phỏng vấn của Đài BBC London vào Tháng 3, lần đầu tiên đã gieo thuật ngữ "Big Bang" để mô tả lý thuyết của Lemaître.',
            'gold': 'Fred Hoyle'
        },
        {
            'id': 'ex_3',
            'question': 'Người Châu Á đầu tiên nhận giải Fields là ai?',
            'context': 'Sau hơn 60 năm tồn tại, Fields Medal đã được trao cho 48 nhà toán học trên toàn thế giới. Nhà toán học Kunihiko Kodaira là người Nhật Bản và cũng là người châu Á đầu tiên giành Fields Medal.',
            'gold': 'Kunihiko Kodaira'
        },
        {
            'id': 'ex_4',
            'question': 'Ai sáng lập Uber?',
            'context': 'Người đồng sáng lập Uber, ông Travis Kalanick, chính thức trở thành tỷ phú, sau khi thu được 1,4 tỷ USD từ bán cổ phiếu.',
            'gold': 'Travis Kalanick'
        },
        {
            'id': 'ex_5',
            'question': 'Quang Hải được mệnh danh là gì?',
            'context': 'Quang Hải được mệnh danh là "Messi của Olympic Việt Nam". Tổng thống Hàn Quốc Moon Jae-in cũng bày tỏ sự ngưỡng mộ tài năng cầu thủ mang áo số 19.',
            'gold': 'Messi của Olympic Việt Nam'
        }
    ]
    
    # Read extra examples from test_clean.json to populate choices
    test_path = 'data/processed/test_clean.json'
    if os.path.exists(test_path):
        try:
            with open(test_path, encoding='utf-8') as f:
                test_data = json.load(f)
                added = 0
                for item in test_data:
                    q = item.get('question_raw', '')
                    ctx = item.get('context_raw', '')
                    ans = item.get('answer_text', '')
                    # Pick short ones for visualization friendliness
                    if q and ctx and ans and len(ctx) < 250 and len(q) < 60 and len(ans) < 30:
                        examples.append({
                            'id': f"test_{item.get('id', added)}",
                            'question': q,
                            'context': ctx,
                            'gold': ans
                        })
                        added += 1
                        if added >= 5: # Load 5 more samples
                            break
        except Exception as e:
            print(f"[WARN] Lỗi load test_clean.json: {e}")
            
    return examples


# Serve index.html directly from template or static files
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/examples', methods=['GET'])
def get_examples():
    try:
        examples = load_preloaded_examples()
        return jsonify(examples)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    question = data.get('question', '').strip()
    context = data.get('context', '').strip()
    selected_models = data.get('models', [])
    gold = data.get('gold', '').strip()
    
    if not question or not context:
        return jsonify({'error': 'Câu hỏi hoặc đoạn văn không được để trống.'}), 400
        
    results = {}
    
    for model_key in selected_models:
        try:
            if model_key == 'bm25':
                res = run_bm25_selector(question, context)
            elif model_key in ('pretrained', 'finetuned'):
                reader = get_model_reader(model_key)
                res = reader.predict(question, context)
            else:
                continue
                
            # Compute evaluation metrics if gold standard answer is available
            if gold:
                pred_norm = normalize_answer(res['answer'])
                gold_norm = normalize_answer(gold)
                res['em'] = int(pred_norm == gold_norm)
                res['f1'] = round(compute_f1(gold, res['answer']), 4)
            else:
                res['em'] = None
                res['f1'] = None
                
            results[model_key] = res
            
        except Exception as e:
            results[model_key] = {
                'error': str(e),
                'answer': 'Lỗi chạy model',
                'confidence': 0.0,
                'latency_ms': 0.0,
                'char_start': 0,
                'char_end': 0,
                'em': 0,
                'f1': 0.0
            }
            print(f"[ERROR] Model {model_key} failed: {e}")
            
    return jsonify({'results': results})


def init_global_corpus():
    global GLOBAL_CORPUS_RAW, GLOBAL_CORPUS_TOKENIZED, GLOBAL_BM25
    test_path = 'data/processed/test_clean.json'
    if os.path.exists(test_path):
        try:
            with open(test_path, encoding='utf-8') as f:
                test_data = json.load(f)
            unique_contexts = {}
            for item in test_data:
                ctx_raw = item.get('context_raw', '')
                ctx_bm25 = item.get('context_bm25', '')
                if ctx_raw and ctx_bm25 and ctx_raw not in unique_contexts:
                    unique_contexts[ctx_raw] = ctx_bm25
            
            GLOBAL_CORPUS_RAW = list(unique_contexts.keys())
            GLOBAL_CORPUS_TOKENIZED = [unique_contexts[c].split() for c in GLOBAL_CORPUS_RAW]
            
            from rank_bm25 import BM25Okapi
            GLOBAL_BM25 = BM25Okapi(GLOBAL_CORPUS_TOKENIZED)
            print(f"  [Corpus QA] Loaded global corpus with {len(GLOBAL_CORPUS_RAW)} unique contexts.")
        except Exception as e:
            print(f"  [Corpus QA] Failed to load global corpus: {e}")
    else:
        print(f"  [Corpus QA] Test clean file not found at: {test_path}")


@app.route('/api/predict_pipeline', methods=['POST'])
def predict_pipeline():
    data = request.json or {}
    question = data.get('question', '').strip()
    selected_models = data.get('models', [])
    gold = data.get('gold', '').strip()
    
    if not question:
        return jsonify({'error': 'Câu hỏi không được để trống.'}), 400
        
    if not GLOBAL_BM25:
        return jsonify({'error': 'Kho dữ liệu chưa được tải hoặc trống.'}), 500
        
    t0 = time.time()
    
    # Tiền xử lý câu hỏi cho BM25
    from src.data.preprocess import preprocess_for_retriever
    q_bm25 = preprocess_for_retriever(question)
    q_tokens = q_bm25.split()
    
    # Truy hồi top-3 contexts
    scores = GLOBAL_BM25.get_scores(q_tokens)
    top_3_indices = np.argsort(scores)[-3:][::-1]
    retrieved_contexts = [GLOBAL_CORPUS_RAW[idx] for idx in top_3_indices]
    
    retrieval_latency = (time.time() - t0) * 1000
    
    results = {}
    for model_key in selected_models:
        try:
            if model_key == 'bm25':
                # Với BM25, chọn câu tốt nhất trong đoạn văn Top-1
                best_context = retrieved_contexts[0]
                res = run_bm25_selector(question, best_context)
                res['selected_context'] = best_context
            elif model_key in ('pretrained', 'finetuned'):
                reader = get_model_reader(model_key)
                
                # Chạy Reader trên cả 3 đoạn văn, tìm kết quả có score cao nhất
                best_score = -1.0
                best_res = None
                best_context = retrieved_contexts[0]
                
                for ctx in retrieved_contexts:
                    pred = reader.predict(question, ctx)
                    score = pred.get('confidence', 0.0)
                    if score > best_score:
                        best_score = score
                        best_res = pred
                        best_context = ctx
                        
                res = best_res
                res['selected_context'] = best_context
            else:
                continue
                
            # Đánh giá nếu có đáp án đúng đi kèm
            if gold:
                pred_norm = normalize_answer(res['answer'])
                gold_norm = normalize_answer(gold)
                res['em'] = int(pred_norm == gold_norm)
                res['f1'] = round(compute_f1(gold, res['answer']), 4)
            else:
                res['em'] = None
                res['f1'] = None
                
            results[model_key] = res
            
        except Exception as e:
            results[model_key] = {
                'error': str(e),
                'answer': 'Lỗi chạy model',
                'confidence': 0.0,
                'latency_ms': 0.0,
                'char_start': 0,
                'char_end': 0,
                'em': 0,
                'f1': 0.0,
                'selected_context': retrieved_contexts[0]
            }
            print(f"[ERROR] Model {model_key} failed in pipeline: {e}")
            
    return jsonify({
        'retrieved_context': retrieved_contexts[0], # Default Top-1 context
        'retrieved_contexts': retrieved_contexts,   # List of all top-3 contexts
        'retrieval_latency_ms': round(retrieval_latency, 1),
        'results': results
    })


if __name__ == '__main__':
    # Initialize global corpus
    init_global_corpus()
    # Add template folders inside src/web
    app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')
    app.static_folder = os.path.join(os.path.dirname(__file__), 'static')
    
    print("==========================================================")
    print(" Khởi động Flask Server cho demo Hỏi Đáp Tiếng Việt")
    print(" Truy cập tại: http://127.0.0.1:5000")
    print("==========================================================")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
