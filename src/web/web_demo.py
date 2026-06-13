# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import re
import torch
from flask import Flask, request, jsonify, send_from_directory, render_template

# Setup paths so we can import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import metrics and components from src or demo
from src.utils.metrics import normalize_answer, compute_f1, compute_exact
from demo import ExtractiveReader

app = Flask(__name__)

# Global cache for heavy models
MODEL_CACHE = {}

class QwenReader:
    def __init__(self, model_path: str):
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU (CUDA)" if self.device == 0 else "CPU"
        print(f"  [QwenReader] Loading model: {model_path} | Device: {device_name}")
        from transformers import pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_path,
            device=self.device,
            torch_dtype=torch.float32
        )
        print("  [QwenReader] Model loaded successfully.")

    def predict(self, question: str, context: str) -> dict:
        t0 = time.time()
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Bạn là một trợ lý AI hữu ích. Nhiệm vụ của bạn là trích xuất chính xác câu trả lời cho câu hỏi từ đoạn văn ngữ cảnh được cung cấp. Chỉ trả về duy nhất cụm từ câu trả lời chính xác lấy từ ngữ cảnh, không viết lại câu, không thêm lời dẫn giải."
                },
                {
                    "role": "user",
                    "content": f"Ngữ cảnh: {context}\nCâu hỏi: {question}\nCâu trả lời ngắn:"
                }
            ]
            prompt = self.generator.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            outputs = self.generator(
                prompt,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                return_full_text=False
            )
            pred_answer = outputs[0]['generated_text'].strip()
            # Clean up potential LLM response wrapping
            pred_answer = pred_answer.replace("Câu trả lời:", "").strip()
            pred_answer = pred_answer.replace("câu trả lời:", "").strip()
            
            # Align char index in context
            char_start = context.find(pred_answer)
            if char_start != -1:
                char_end = char_start + len(pred_answer)
            else:
                char_start = 0
                char_end = 0
                
            latency = (time.time() - t0) * 1000
            return {
                'answer': pred_answer,
                'confidence': 1.0,
                'char_start': char_start,
                'char_end': char_end,
                'latency_ms': round(latency, 1)
            }
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return {
                'answer': '',
                'confidence': 0.0,
                'char_start': 0,
                'char_end': 0,
                'latency_ms': round(latency, 1),
                'error': str(e)
            }

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
        elif model_key == 'qwen':
            # Generative LLM reader (Qwen2.5-0.5B-Instruct)
            model_path = 'Qwen/Qwen2.5-0.5B-Instruct'
            MODEL_CACHE[model_key] = QwenReader(model_path)
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
            elif model_key in ('pretrained', 'finetuned', 'qwen'):
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


if __name__ == '__main__':
    # Add template folders inside src/web
    app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')
    app.static_folder = os.path.join(os.path.dirname(__file__), 'static')
    
    print("==========================================================")
    print(" Khởi động Flask Server cho demo Hỏi Đáp Tiếng Việt")
    print(" Truy cập tại: http://127.0.0.1:5000")
    print("==========================================================")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
