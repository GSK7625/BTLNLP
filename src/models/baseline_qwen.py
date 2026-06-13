# -*- coding: utf-8 -*-
import sys
import os
import io

# Setup path so we can import src even if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
=============================================================
BASELINE GENERATIVE LLM: Qwen2.5-Instruct (Generative Reader)
=============================================================
"""

import json
import argparse
import time
from tqdm import tqdm
import torch
from transformers import pipeline

from src.utils.metrics import normalize_answer, compute_exact, compute_f1, get_tokens
from src.models.baseline_pretrained import classify_error

def evaluate(data_path: str, model_name: str, num_samples: int = None):
    print(f"\n{'='*60}")
    print(f"BASELINE GENERATIVE LLM: Qwen2.5-Instruct")
    print(f"{'='*60}")
    print(f"Model     : {model_name}")
    print(f"Dữ liệu   : {data_path}")

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    print(f"Device    : {device_name}")
    print(f"\nĐang load model {model_name}...")

    t_load_start = time.time()
    try:
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return 0.0, 0.0
    print(f"Model loaded thành công trong {time.time() - t_load_start:.1f}s!\n")

    # Load data
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]
        print(f"Chạy thử trên {num_samples} mẫu đầu tiên.")

    total = len(data)
    em_scores = []
    f1_scores = []
    error_cases = []
    results_list = []

    print(f"Đang đánh giá {total} mẫu...")

    for i, item in enumerate(tqdm(data, desc="Evaluating with Qwen")):
        question = item['question_raw']
        context = item['context_raw']
        gold = item['answer_text']
        sid = item.get('id', i)

        # Construct Chat messages
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

        t_start = time.time()
        try:
            prompt = generator.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # Generate response
            outputs = generator(
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
        except Exception as e:
            print(f"Error at index {i}: {e}")
            pred_answer = ""
            
        latency = (time.time() - t_start) * 1000

        # Calculate metrics
        em = compute_exact(gold, pred_answer)
        f1 = compute_f1(gold, pred_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        # Log details
        char_start = context.find(pred_answer)
        if char_start != -1:
            char_end = char_start + len(pred_answer)
        else:
            char_start = 0
            char_end = 0

        res_item = {
            'id': sid,
            'question': question,
            'context': context,
            'gold': gold,
            'predicted': pred_answer,
            'em': em,
            'f1': round(f1, 4),
            'confidence': 1.0,
            'latency_ms': round(latency, 1),
            'char_start': char_start,
            'char_end': char_end
        }
        results_list.append(res_item)

        if em == 0 and len(error_cases) < 20:
            error_cases.append({
                'id': sid,
                'question': question,
                'context': context[:200] + '...' if len(context) > 200 else context,
                'gold': gold,
                'predicted': pred_answer,
                'confidence': 1.0,
                'f1': round(f1, 4),
                'error_type': classify_error(gold, pred_answer, context),
            })

    avg_em = sum(em_scores) / total * 100
    avg_f1 = sum(f1_scores) / total * 100

    print(f"\n{'─'*40}")
    print(f"  KẾT QUẢ BASELINE QWEN:")
    print(f"{'─'*40}")
    print(f"  Model        : {model_name}")
    print(f"  Số mẫu       : {total:,}")
    print(f"  Exact Match  : {avg_em:.2f}%")
    print(f"  Token F1     : {avg_f1:.2f}%")
    print(f"{'─'*40}")

    # Save details
    model_tag = model_name.replace('/', '_')
    results = {
        'model': model_name,
        'description': f'Qwen2.5 generative reader model: {model_name}',
        'data_path': data_path,
        'num_samples': total,
        'exact_match': round(avg_em, 4),
        'token_f1': round(avg_f1, 4),
        'error_analysis': error_cases,
        'predictions': results_list
    }

    out_path = data_path.replace('.json', f'_qwen_{model_tag}_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Kết quả đã lưu: {out_path}")

    # Standard clean name results for evaluate.py
    standard_out_path = data_path.replace('.json', f'_qwen_results.json')
    with open(standard_out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Kết quả chuẩn đã lưu: {standard_out_path}")

    return avg_em, avg_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Qwen: Qwen2.5 generative QA reader')
    parser.add_argument(
        '--data', type=str,
        default='data/processed/test_clean.json'
    )
    parser.add_argument(
        '--model', type=str,
        default='Qwen/Qwen2.5-0.5B-Instruct'
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Số lượng mẫu chạy đánh giá (mặc định: toàn bộ)'
    )
    args = parser.parse_args()

    evaluate(args.data, args.model, args.num_samples)
