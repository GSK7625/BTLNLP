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
BASELINE B2: XLM-RoBERTa Pretrained (Không fine-tune)
=============================================================
"""

import json
import argparse
from tqdm import tqdm
import torch

from src.utils.metrics import normalize_answer, compute_exact, compute_f1, get_tokens

# ------------------------------------------------------------------ #
#  Classify error types
# ------------------------------------------------------------------ #

def classify_error(gold: str, pred: str, context: str) -> str:
    gold_n = normalize_answer(gold)
    pred_n = normalize_answer(pred)
    ctx_n = normalize_answer(context)
    if not pred_n:
        return "Empty prediction"
    if gold_n == pred_n:
        return "Correct"
    if gold_n in ctx_n:
        toks_g = set(get_tokens(gold))
        toks_p = set(get_tokens(pred))
        if toks_g & toks_p:
            return "Partial match (boundary error)"
        return "Gold in context, wrong span selected"
    return "Gold not directly in context"


# ------------------------------------------------------------------ #
#  Main evaluation
# ------------------------------------------------------------------ #

def evaluate(data_path: str, model_name: str, batch_size: int, num_samples: int = None):
    print(f"\n{'='*60}")
    print(f"BASELINE B2: XLM-RoBERTa Pretrained (không fine-tune)")
    print(f"{'='*60}")
    print(f"Model     : {model_name}")
    print(f"Dữ liệu   : {data_path}")

    # ------ Load model & tokenizer ------
    import transformers
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    print(f"Device    : {device_name}")
    print(f"\nĐang load model {model_name}...")

    # Transformers 4.x dùng "question-answering", 5.x dùng "extractive-question-answering"
    _tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    _task = "question-answering" if _tv < (5, 0) else "extractive-question-answering"

    try:
        qa_pipeline = pipeline(
            _task,
            model=model_name,
            tokenizer=model_name,
            device=device,
        )
    except Exception as e:
        print(f"  [WARN] pipeline('{_task}') thất bại: {e}")
        print("  Thử dùng AutoModel trực tiếp...")
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import transformers as _tf
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = _tf.pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    print("Model loaded thành công!\n")

    # ------ Load data ------
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]
        print(f"Chạy thử trên {num_samples} mẫu đầu tiên.")

    total = len(data)
    em_scores = []
    f1_scores = []
    error_cases = []

    print(f"Đang đánh giá {total} mẫu (batch_size={batch_size})...")

    # Chuẩn bị batch inputs
    inputs = [
        {'question': s['question_raw'], 'context': s['context_raw']}
        for s in data
    ]
    gold_answers = [s['answer_text'] for s in data]
    sample_ids = [s.get('id', i) for i, s in enumerate(data)]

    # Chạy inference theo batch
    predictions = []
    for i in tqdm(range(0, total, batch_size), desc="Inference"):
        batch_inputs = inputs[i:i+batch_size]
        try:
            batch_results = qa_pipeline(
                question=[inp['question'] for inp in batch_inputs],
                context=[inp['context'] for inp in batch_inputs],
                max_answer_len=50
            )
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            predictions.extend(batch_results)
        except Exception as e:
            print(f"\n  [WARN] Lỗi batch {i}: {e}. Thử từng mẫu...")
            for inp in batch_inputs:
                try:
                    res = qa_pipeline(
                        question=inp['question'],
                        context=inp['context'],
                        max_answer_len=50
                    )
                    predictions.append(res)
                except:
                    predictions.append({'answer': ''})

    # Tính metric
    for idx, (pred_result, gold, sid) in enumerate(zip(predictions, gold_answers, sample_ids)):
        pred_answer = pred_result.get('answer', '')
        score_conf = pred_result.get('score', 0.0)

        em = compute_exact(gold, pred_answer)
        f1 = compute_f1(gold, pred_answer)
        em_scores.append(em)
        f1_scores.append(f1)

        if em == 0 and len(error_cases) < 20:
            ctx = inputs[idx]['context']
            error_cases.append({
                'id': sid,
                'question': inputs[idx]['question'],
                'context': ctx[:200] + '...' if len(ctx) > 200 else ctx,
                'gold': gold,
                'predicted': pred_answer,
                'confidence': round(score_conf, 4),
                'f1': round(f1, 4),
                'error_type': classify_error(gold, pred_answer, ctx),
            })

    avg_em = sum(em_scores) / total * 100
    avg_f1 = sum(f1_scores) / total * 100

    print(f"\n{'─'*40}")
    print(f"  KẾT QUẢ BASELINE B2 (XLM-RoBERTa pretrained):")
    print(f"{'─'*40}")
    print(f"  Model        : {model_name}")
    print(f"  Số mẫu       : {total:,}")
    print(f"  Exact Match  : {avg_em:.2f}%")
    print(f"  Token F1     : {avg_f1:.2f}%")
    print(f"{'─'*40}")

    # Lưu kết quả
    model_tag = model_name.replace('/', '_')
    results = {
        'model': model_name,
        'description': 'XLM-RoBERTa pretrained on SQuAD2 — no fine-tune on ViSpanExtractQA',
        'data_path': data_path,
        'num_samples': total,
        'exact_match': round(avg_em, 4),
        'token_f1': round(avg_f1, 4),
        'error_analysis': error_cases,
    }
    suffix = f"_pretrained_{model_tag}_{num_samples}samples_results.json" if num_samples else f"_pretrained_{model_tag}_results.json"
    out_path = data_path.replace('.json', suffix)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Kết quả đã lưu: {out_path}")

    # In ví dụ sai
    print(f"\n{'─'*40}")
    print(f"  VÍ DỤ LỖI (5 mẫu đầu):")
    print(f"{'─'*40}")
    for i, err in enumerate(error_cases[:5], 1):
        print(f"\n  [{i}] ID={err['id']} | F1={err['f1']} | Conf={err['confidence']}")
        print(f"  Câu hỏi  : {err['question']}")
        print(f"  Đúng     : {err['gold']}")
        print(f"  Dự đoán  : {err['predicted']}")
        print(f"  Loại lỗi : {err['error_type']}")

    return avg_em, avg_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline B2: XLM-RoBERTa Pretrained QA')
    parser.add_argument(
        '--data', type=str,
        default='data/processed/test_clean.json',
        help='Đường dẫn đến file JSON đã làm sạch'
    )
    parser.add_argument(
        '--model', type=str,
        default='deepset/xlm-roberta-base-squad2',
        help='Tên model trên HuggingFace (mặc định: deepset/xlm-roberta-base-squad2)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size khi inference (mặc định: 16)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Số mẫu để chạy thử (mặc định: toàn bộ)'
    )
    args = parser.parse_args()

    evaluate(args.data, args.model, args.batch_size, args.num_samples)
