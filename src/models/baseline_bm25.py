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
BASELINE B1: BM25-Only Reader
=============================================================
"""

import json
import re
import argparse
from tqdm import tqdm

from src.utils.metrics import normalize_answer, compute_exact, compute_f1

# ------------------------------------------------------------------ #
#  Rule-based span selector (BM25-only reader)
# ------------------------------------------------------------------ #

def tokenize_vi(text: str):
    """Tách token đơn giản cho tiếng Việt (split theo khoảng trắng)."""
    return text.lower().split()


def select_span_by_overlap(question: str, context: str) -> str:
    """
    Chiến lược BM25-only:
      1. Tách context thành các câu.
      2. Với mỗi câu, đếm số token overlap với question.
      3. Trả về câu có overlap cao nhất làm "answer span".
    Nếu không tách được câu → trả về toàn bộ context.
    """
    # Tách câu theo dấu chấm, chấm hỏi, chấm than, dấu phẩy + mệnh đề
    sentences = re.split(r'(?<=[.?!،。])\s+|(?<=,)\s+(?=[A-ZÁẮẶẤẦ])', context)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return context.strip()

    q_tokens = set(tokenize_vi(question))

    best_sentence = sentences[0]
    best_overlap = -1

    for sent in sentences:
        s_tokens = set(tokenize_vi(sent))
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent

    return best_sentence


# ------------------------------------------------------------------ #
#  Classify error types
# ------------------------------------------------------------------ #

def classify_error(gold: str, pred: str, context: str) -> str:
    """Phân loại lỗi đơn giản để phân tích."""
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    context_norm = normalize_answer(context)

    if gold_norm in context_norm:
        if gold_norm in pred_norm:
            return "Partial match (F1 > 0)"
        else:
            return "Answer in context but wrong sentence selected"
    else:
        return "Answer not directly in context (paraphrase/extraction issue)"


# ------------------------------------------------------------------ #
#  Main evaluation
# ------------------------------------------------------------------ #

def evaluate(data_path: str, num_samples: int = None):
    print(f"\n{'='*60}")
    print(f"BASELINE B1: BM25-Only Reader")
    print(f"{'='*60}")
    print(f"Dữ liệu: {data_path}")

    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]
        print(f"Chạy thử trên {num_samples} mẫu đầu tiên.")

    total = len(data)
    em_scores = []
    f1_scores = []
    error_cases = []  # Lưu ví dụ sai để phân tích lỗi

    print(f"\nĐang đánh giá {total} mẫu...")

    for sample in tqdm(data, desc="Evaluating BM25-only baseline"):
        question = sample['question_raw']
        context = sample['context_raw']
        gold_answer = sample['answer_text']

        # Dự đoán: chọn câu overlap cao nhất
        pred_answer = select_span_by_overlap(question, context)

        em = compute_exact(gold_answer, pred_answer)
        f1 = compute_f1(gold_answer, pred_answer)

        em_scores.append(em)
        f1_scores.append(f1)

        # Ghi lại mẫu sai để phân tích lỗi
        if em == 0 and len(error_cases) < 20:
            error_cases.append({
                'id': sample.get('id', '?'),
                'question': question,
                'context': context[:200] + '...' if len(context) > 200 else context,
                'gold': gold_answer,
                'predicted': pred_answer,
                'f1': round(f1, 4),
                'error_type': classify_error(gold_answer, pred_answer, context),
            })

    avg_em = sum(em_scores) / total * 100
    avg_f1 = sum(f1_scores) / total * 100

    print(f"\n{'─'*40}")
    print(f"  KẾT QUẢ BASELINE B1 (BM25-Only):")
    print(f"{'─'*40}")
    print(f"  Số mẫu       : {total:,}")
    print(f"  Exact Match  : {avg_em:.2f}%")
    print(f"  Token F1     : {avg_f1:.2f}%")
    print(f"{'─'*40}")

    # Lưu kết quả
    results = {
        'model': 'BM25-Only (Rule-based span selection)',
        'data_path': data_path,
        'num_samples': total,
        'exact_match': round(avg_em, 4),
        'token_f1': round(avg_f1, 4),
        'error_analysis': error_cases,
    }

    suffix = f"_bm25only_{num_samples}samples_results.json" if num_samples else "_bm25only_results.json"
    out_path = data_path.replace('.json', suffix)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Kết quả đã lưu: {out_path}")

    # In 5 ví dụ sai đầu tiên
    print(f"\n{'─'*40}")
    print(f"  VÍ DỤ LỖI (5 mẫu đầu):")
    print(f"{'─'*40}")
    for i, err in enumerate(error_cases[:5], 1):
        print(f"\n  [{i}] ID={err['id']} | F1={err['f1']}")
        print(f"  Câu hỏi  : {err['question']}")
        print(f"  Đúng     : {err['gold']}")
        print(f"  Dự đoán  : {err['predicted'][:100]}")
        print(f"  Loại lỗi : {err['error_type']}")

    return avg_em, avg_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline B1: BM25-Only Reader')
    parser.add_argument('--data', type=str,
                        default='data/processed/test_clean.json',
                        help='Đường dẫn đến file JSON đã làm sạch')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Số mẫu để chạy thử (mặc định: toàn bộ)')
    args = parser.parse_args()

    evaluate(args.data, args.num_samples)
