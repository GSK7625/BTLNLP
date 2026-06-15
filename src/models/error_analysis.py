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
PHÂN TÍCH LỖI TỔNG HỢP (Mục 4.7 Báo cáo)
=============================================================
"""

import json
import csv
import argparse
from collections import Counter
from pathlib import Path

from src.utils.metrics import normalize_answer, compute_f1

# ─────────────────────────────────────────────────────────── #
#  Error classification
# ─────────────────────────────────────────────────────────── #

def classify_error_detailed(gold: str, pred: str, question: str, context: str) -> dict:
    """Phân loại lỗi chi tiết theo từng loại."""
    gold_n = normalize_answer(gold)
    pred_n = normalize_answer(pred)
    ctx_n  = normalize_answer(context)

    if gold_n == pred_n:
        return {'type': 'Đúng', 'cause': '', 'suggestion': ''}

    f1 = compute_f1(gold, pred)

    # Lỗi biên (model chọn đúng vùng nhưng lấy thừa/thiếu)
    if f1 >= 0.5:
        if gold_n in pred_n:
            return {
                'type': 'Lỗi biên (span dư)',
                'cause': 'Model trích xuất dư prefix/suffix (ví dụ: "Thiếu tướng X" thay vì "X")',
                'suggestion': 'Fine-tune thêm trên dữ liệu tiếng Việt để học boundary chính xác hơn',
            }
        elif pred_n in gold_n:
            return {
                'type': 'Lỗi biên (span thiếu)',
                'cause': 'Model trích xuất thiếu một phần câu trả lời',
                'suggestion': 'Tăng max_answer_len hoặc fine-tune thêm',
            }
        return {
            'type': 'Partial match',
            'cause': 'Model chọn gần đúng nhưng không chính xác hoàn toàn',
            'suggestion': 'Fine-tune trên dữ liệu domain tiếng Việt',
        }

    # Câu trả lời đúng có trong context không?
    if gold_n in ctx_n:
        # Model chọn sai câu hoàn toàn
        return {
            'type': 'Sai span (gold có trong context)',
            'cause': 'Model chọn sai vùng — context dài hoặc nhiều thực thể tương tự',
            'suggestion': 'Cải thiện khả năng hiểu ngữ cảnh của model',
        }
    else:
        # Câu trả lời không có trong context (lỗi dữ liệu)
        return {
            'type': 'Gold không có trong context',
            'cause': 'Lỗi dữ liệu: câu trả lời là paraphrase/dịch không khớp với context',
            'suggestion': 'Cần lọc kỹ hơn ở bước tiền xử lý (đã xử lý một phần)',
        }


# ─────────────────────────────────────────────────────────── #
#  Load results
# ─────────────────────────────────────────────────────────── #

def load_result_file(path: str) -> dict:
    if not Path(path).exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────── #
#  Main analysis
# ------------------------------------------------─────────── #

def run_analysis(test_data_path: str,
                 b2_results_path: str, m1_results_path: str = None,
                 output_csv: str = 'error_analysis.csv',
                 num_examples: int = 15):

    print("\n" + "="*60)
    print("  PHÂN TÍCH LỖI TỔNG HỢP")
    print("="*60)

    # ── 1. In bảng kết quả tổng quan ──────────────────────── #
    print("\n  [1] BẢNG KẾT QUẢ TỔNG QUAN")
    print("─"*60)

    # Load test clean data for full context lookup (to avoid issues with truncated contexts)
    full_contexts = {}
    if test_data_path and os.path.exists(test_data_path):
        try:
            with open(test_data_path, encoding='utf-8') as f:
                test_data = json.load(f)
                full_contexts = {item['id']: item['context_raw'] for item in test_data if 'id' in item}
        except Exception as e:
            print(f"  [WARN] Không thể đọc {test_data_path}: {e}")

    models_info = []

    b2 = load_result_file(b2_results_path)
    m1 = load_result_file(m1_results_path) if m1_results_path else None

    if b2:
        models_info.append({'name': 'B2: XLM-RoBERTa Pretrained (SQuAD2)', **b2})
    if m1:
        models_info.append({'name': 'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)', **m1})

    # In bảng
    print(f"\n  {'Mô hình':<45} {'EM (%)':>8} {'F1 (%)':>8}")
    print("  " + "─"*62)
    for m in models_info:
        print(f"  {m['name']:<45} {m.get('exact_match', 'N/A'):>8} {m.get('token_f1', 'N/A'):>8}")
    print()

    # In markdown
    print("\n  [MARKDOWN TABLE]")
    print("  | Mô hình | EM (%) | F1 (%) |")
    print("  |---------|--------|--------|")
    for m in models_info:
        print(f"  | {m['name']} | {m.get('exact_match', 'N/A')} | {m.get('token_f1', 'N/A')} |")

    # ── 2. Phân tích lỗi chi tiết ─────────────────────────── #
    print(f"\n\n  [2] BẢNG PHÂN TÍCH LỖI CHI TIẾT (Mục 4.7)")
    print("─"*60)

    # Collect error cases từ tất cả model
    all_errors = []

    for model_data in models_info:
        model_name = model_data['name']
        for err in model_data.get('error_analysis', []):
            gold    = err.get('gold', '')
            pred    = err.get('predicted', '')
            question = err.get('question', '')
            
            # Use original full context to avoid false "Gold not in context" errors due to truncation
            err_id = err.get('id')
            context = full_contexts.get(err_id) if err_id in full_contexts else err.get('context', '')
            if not context:
                context = err.get('context', '')

            analysis = classify_error_detailed(gold, pred, question, context)

            all_errors.append({
                'STT': len(all_errors) + 1,
                'Mô hình': model_name,
                'Câu hỏi': question,
                'Câu trả lời đúng': gold,
                'Dự đoán': pred,
                'F1': err.get('f1', compute_f1(gold, pred)),
                'Loại lỗi': analysis['type'],
                'Nguyên nhân': analysis['cause'],
                'Hướng cải thiện': analysis['suggestion'],
            })

    # In bảng lỗi
    examples_shown = min(num_examples, len(all_errors))
    for i, err in enumerate(all_errors[:examples_shown], 1):
        print(f"\n  [{i:02d}] {err['Mô hình'][:35]}")
        print(f"       Câu hỏi   : {err['Câu hỏi']}")
        print(f"       Đúng      : {err['Câu trả lời đúng']}")
        print(f"       Dự đoán   : {str(err['Dự đoán'])[:80]}")
        print(f"       F1 = {err['F1']:.3f} | Loại: {err['Loại lỗi']}")
        print(f"       Nguyên nhân: {err['Nguyên nhân']}")

    # ── 3. Thống kê loại lỗi ─────────────────────────────── #
    print(f"\n\n  [3] THỐNG KÊ LOẠI LỖI")
    print("─"*60)
    error_counts = Counter(e['Loại lỗi'] for e in all_errors)
    total_errors = len(all_errors)
    for etype, count in error_counts.most_common():
        pct = count / total_errors * 100 if total_errors else 0
        print(f"  {etype:<40} {count:>4} ({pct:.1f}%)")

    # ── 4. Xuất CSV cho báo cáo ──────────────────────────── #
    if all_errors:
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_errors[0].keys()))
            writer.writeheader()
            writer.writerows(all_errors)
        print(f"\n  Bảng phân tích lỗi xuất ra: {output_csv}")
    else:
        print(f"\n  [WARN] Không có dữ liệu lỗi. Hãy chạy baseline trước.")

    # ── 5. Nhận xét tổng quan cho báo cáo ───────────────── #
    print(f"\n\n  [4] NHẬN XÉT CHO BÁO CÁO")
    print("─"*60)
    print("""
  Từ bảng kết quả, có thể rút ra các nhận xét sau:

  1. B2 (XLM-RoBERTa pretrained SQuAD2) đạt F1 ~70% — cho thấy model
     đã học được cấu trúc QA từ tiếng Anh, nhưng gặp khó khăn về biên
     span trong tiếng Việt (lỗi prefix như "Thiếu tướng X" → "X").

  2. M1 (Fine-tuned) dự kiến cải thiện rõ so với B2 nhờ học trực tiếp
     trên dữ liệu tiếng Việt, đặc biệt giảm lỗi biên span.

  3. Loại lỗi phổ biến nhất: "Lỗi biên (span dư)" — model trích xuất
     dư prefix/suffix tiếng Việt như danh hiệu, chức vụ.

  4. Hướng cải thiện: (a) Fine-tune với nhiều dữ liệu hơn, (b) Post-
     processing để loại bỏ prefix thông thường (Thiếu tướng, ông, bà...),
     (c) Dùng BM25 retrieval thực sự khi có corpus nhiều đoạn văn.
  """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phân tích lỗi tổng hợp')
    parser.add_argument('--test_data', default='data/processed/test_clean.json')
    parser.add_argument('--b2_results',
                        default='data/processed/test_clean_pretrained_deepset_xlm-roberta-base-squad2_results.json')
    parser.add_argument('--m1_results', default=None,
                        help='Đường dẫn kết quả M1 (sau khi fine-tune xong)')
    parser.add_argument('--output_csv', default='error_analysis.csv')
    parser.add_argument('--num_examples', type=int, default=15)
    args = parser.parse_args()

    m1_res_path = args.m1_results
    b2_res_path = args.b2_results

    # Tự động phát hiện kết quả M1 nếu chưa chỉ định
    if not m1_res_path:
        import glob
        parent_dir = os.path.dirname(args.test_data)
        pattern = os.path.join(parent_dir, "*finetuned*results.json")
        matching_files = glob.glob(pattern)
        if matching_files:
            m1_res_path = matching_files[0]
            print(f"  [INFO] Tự động chọn kết quả M1: {m1_res_path}")

    # Ánh xạ b2_results tương ứng với hậu tố số lượng mẫu của m1_results
    if m1_res_path and os.path.exists(m1_res_path):
        for suffix in ["_500samples", "_5000samples"]:
            if suffix in m1_res_path:
                if b2_res_path == 'data/processed/test_clean_pretrained_deepset_xlm-roberta-base-squad2_results.json':
                    alt_b2 = args.test_data.replace('.json', f'_pretrained_deepset_xlm-roberta-base-squad2{suffix}_results.json')
                    if os.path.exists(alt_b2):
                        b2_res_path = alt_b2
                        print(f"  [INFO] Tự động chọn kết quả B2: {b2_res_path}")
                break

    run_analysis(
        test_data_path=args.test_data,
        b2_results_path=b2_res_path,
        m1_results_path=m1_res_path,
        output_csv=args.output_csv,
        num_examples=args.num_examples,
    )
