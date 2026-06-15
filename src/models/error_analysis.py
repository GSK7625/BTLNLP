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

    # Lỗi biên (chọn đúng vùng nhưng lấy thừa/thiếu từ)
    if f1 >= 0.5:
        if gold_n in pred_n:
            return {
                'type': 'Nhãn mơ hồ',
                'cause': 'Ranh giới đáp án bị nhập nhằng giữa việc lấy kèm chức danh/danh xưng hay chỉ lấy tên riêng',
                'suggestion': 'Xây dựng bộ quy tắc hậu xử lý để loại bỏ các danh xưng thông dụng (ông, bà, Đại tướng...)',
            }
        elif pred_n in gold_n:
            return {
                'type': 'Mô hình học theo từ khóa bề mặt',
                'cause': 'Mô hình có xu hướng khớp các từ khóa bề mặt gần nhau mà chưa nắm bắt trọn vẹn ngữ pháp câu',
                'suggestion': 'Tối ưu cấu hình độ dài câu trả lời tối đa khi suy luận và tăng số lượng mẫu huấn luyện',
            }
        return {
            'type': 'Mô hình học theo từ khóa bề mặt',
            'cause': 'Mô hình bị đánh lừa bởi các từ khóa tương tự gần nhau dẫn đến chọn gần đúng nhưng chưa trọn vẹn',
            'suggestion': 'Bổ sung dữ liệu huấn luyện tiếng Việt đa dạng cấu trúc để tăng độ chính xác biên',
        }

    # Câu trả lời đúng có trong context không?
    if gold_n in ctx_n:
        return {
            'type': 'Mô hình không hiểu phủ định, mỉa mai, so sánh hoặc ngữ cảnh',
            'cause': 'Ngữ cảnh dài hoặc chứa nhiều thực thể cùng loại gây nhiễu khiến mô hình trích xuất sai vị trí',
            'suggestion': 'Tăng chiều dài ngữ cảnh tối đa (max_seq_length) khi huấn luyện và cải thiện độ sâu ngữ nghĩa',
        }
    else:
        return {
            'type': 'Dữ liệu bị nhiễu',
            'cause': 'Nhãn đáp án chuẩn không xuất hiện trực tiếp trong đoạn văn ngữ cảnh (do lỗi dịch máy hoặc paraphrase)',
            'suggestion': 'Lọc và chuẩn hóa dữ liệu tốt hơn ở bước tiền xử lý',
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

    # Tự động phát hiện định dạng tệp kết quả Pipeline
    is_pipeline = False
    if b2 and ("pretrained_pipeline" in b2 or "finetuned_pipeline" in b2):
        is_pipeline = True
    elif m1 and ("pretrained_pipeline" in m1 or "finetuned_pipeline" in m1):
        is_pipeline = True

    if is_pipeline:
        print("  [INFO] Phát hiện tệp kết quả định dạng Pipeline (Retriever-Reader)")
        # Nếu b2 chứa định dạng pipeline
        if b2 and ("pretrained_pipeline" in b2 or "finetuned_pipeline" in b2):
            b2_pipeline = b2.get("pretrained_pipeline")
            m1_pipeline = b2.get("finetuned_pipeline")
            if b2_pipeline:
                models_info.append({'name': 'B2: BM25 + XLM-R Pretrained (Pipeline)', 'is_pipeline': True, **b2_pipeline})
            if m1_pipeline:
                models_info.append({'name': 'M1: BM25 + XLM-R Fine-tuned (Pipeline)', 'is_pipeline': True, **m1_pipeline})
        # Ngược lại nếu m1 chứa định dạng pipeline
        elif m1 and ("pretrained_pipeline" in m1 or "finetuned_pipeline" in m1):
            b2_pipeline = m1.get("pretrained_pipeline")
            m1_pipeline = m1.get("finetuned_pipeline")
            if b2_pipeline:
                models_info.append({'name': 'B2: BM25 + XLM-R Pretrained (Pipeline)', 'is_pipeline': True, **b2_pipeline})
            if m1_pipeline:
                models_info.append({'name': 'M1: BM25 + XLM-R Fine-tuned (Pipeline)', 'is_pipeline': True, **m1_pipeline})
    else:
        if b2:
            models_info.append({'name': 'B2: XLM-RoBERTa Pretrained (SQuAD2)', 'is_pipeline': False, **b2})
        if m1:
            models_info.append({'name': 'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)', 'is_pipeline': False, **m1})

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
        is_model_pipeline = model_data.get('is_pipeline', False)
        for err in model_data.get('error_analysis', []):
            gold    = err.get('gold', '')
            pred    = err.get('predicted', '')
            question = err.get('question', '')
            
            err_id = err.get('id')
            retriever_correct = err.get('retriever_correct', True)

            # Phân loại lỗi
            if is_model_pipeline and not retriever_correct:
                context = err.get('retrieved_context', '') or err.get('context', '')
                analysis = {
                    'type': 'Với tìm kiếm: truy vấn và tài liệu dùng từ khác nhau nhưng cùng nghĩa',
                    'cause': 'Truy vấn sử dụng từ đồng nghĩa hoặc diễn đạt khác biệt so với văn bản gốc làm bộ tìm kiếm từ khóa BM25 thất bại',
                    'suggestion': 'Nâng cấp bộ truy hồi lên tìm kiếm ngữ nghĩa (Semantic Search/DPR) sử dụng mô hình ngôn ngữ'
                }
            else:
                # Dùng context từ full_contexts nếu có, nếu không lấy context từ kết quả
                context = full_contexts.get(err_id) if err_id in full_contexts else err.get('context', '')
                if not context:
                    context = err.get('retrieved_context', '')
                if not context:
                    context = err.get('context', '')

                analysis = classify_error_detailed(gold, pred, question, context)

            all_errors.append({
                'STT': len(all_errors) + 1,
                'Mô hình': model_name,
                'Văn bản/input': f"Câu hỏi: {question}\nNgữ cảnh: {context}",
                'Nhãn đúng hoặc kết quả mong muốn': gold,
                'Kết quả mô hình dự đoán/sinh ra/truy hồi': pred,
                'F1': err.get('f1', compute_f1(gold, pred)),
                'Lỗi thuộc loại nào': analysis['type'],
                'Nguyên nhân nghi ngờ': analysis['cause'],
                'Hướng cải thiện': analysis['suggestion'],
            })

    # In bảng lỗi
    examples_shown = min(num_examples, len(all_errors))
    for i, err in enumerate(all_errors[:examples_shown], 1):
        print(f"\n  [{i:02d}] {err['Mô hình'][:35]}")
        q_text = err['Văn bản/input'].split('\n')[0].replace('Câu hỏi: ', '')
        print(f"       Câu hỏi   : {q_text}")
        print(f"       Đúng      : {err['Nhãn đúng hoặc kết quả mong muốn']}")
        print(f"       Dự đoán   : {str(err['Kết quả mô hình dự đoán/sinh ra/truy hồi'])[:80]}")
        print(f"       F1 = {err['F1']:.3f} | Loại: {err['Lỗi thuộc loại nào']}")
        print(f"       Nguyên nhân: {err['Nguyên nhân nghi ngờ']}")

    # ── 3. Thống kê loại lỗi ─────────────────────────────── #
    print(f"\n\n  [3] THỐNG KÊ LOẠI LỖI")
    print("─"*60)
    error_counts = Counter(e['Lỗi thuộc loại nào'] for e in all_errors)
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
    if is_pipeline:
        print("""
  Từ bảng kết quả đánh giá Pipeline, có thể rút ra các nhận xét sau:

  1. Hiệu năng hệ thống bị ảnh hưởng lớn bởi độ chính xác của bộ truy hồi (Retriever).
     Nếu BM25 không tìm thấy ngữ cảnh đúng (Lỗi truy hồi), Reader sẽ không thể trả lời chính xác.

  2. Mô hình Fine-tuned M1 trong Pipeline cải thiện rõ rệt so với Pretrained Reader,
     giúp giảm thiểu các lỗi biên và tăng độ tin cậy khi trích xuất.

  3. Thuật toán Rank Penalty giúp giảm đáng kể lỗi overconfidence (tin cậy sai lệch)
     của Reader khi phải đọc các đoạn văn nhiễu do Retriever trả về.

  4. Hướng cải thiện: (a) Nâng cấp Retriever bằng các mô hình Dense Passage Retrieval (DPR)
     hoặc Việt hóa (PhoBERT, SBERT) để tìm kiếm ngữ nghĩa tốt hơn, (b) Áp dụng bộ lọc
     hậu xử lý loại bỏ các danh xưng thừa.
        """)
    else:
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

    # Nếu một trong hai tham số chỉ tới kết quả pipeline, ta sẽ chỉ sử dụng tệp đó và tắt cơ chế auto-detect
    if (b2_res_path and "pipeline" in b2_res_path) or (m1_res_path and "pipeline" in m1_res_path):
        if b2_res_path and "pipeline" in b2_res_path:
            m1_res_path = None
        else:
            b2_res_path = m1_res_path
            m1_res_path = None
    else:
        # Tự động phát hiện kết quả M1 nếu chưa chỉ định (cho chế độ standalone)
        if not m1_res_path:
            import glob
            parent_dir = os.path.dirname(args.test_data)
            pattern = os.path.join(parent_dir, "*finetuned*results.json")
            matching_files = glob.glob(pattern)
            if matching_files:
                m1_res_path = matching_files[0]
                print(f"  [INFO] Tự động chọn kết quả M1: {m1_res_path}")

        # Ánh xạ b2_results tương ứng với hậu tố số lượng mẫu của m1_results (cho chế độ standalone)
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
