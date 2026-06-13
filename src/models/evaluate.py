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
TỔNG HỢP KẾT QUẢ: So sánh tất cả mô hình
=============================================================
"""

import json
import argparse
from pathlib import Path

def load_result(path: str):
    """Load kết quả từ file JSON."""
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def print_comparison_table(results: list[dict]):
    """In bảng so sánh dạng markdown cho báo cáo."""
    print("\n" + "="*80)
    print("  BẢNG SO SÁNH KẾT QUẢ — Extractive QA trên ViSpanExtractQA (Test set)")
    print("="*80)

    header = f"{'Mô hình':<45} {'EM (%)':>8} {'F1 (%)':>8} {'Ghi chú'}"
    print(f"\n{header}")
    print("─"*80)

    for r in results:
        name = r['name']
        em = r.get('em', 'N/A')
        f1 = r.get('f1', 'N/A')
        note = r.get('note', '')
        em_str = f"{em:.2f}" if isinstance(em, float) else em
        f1_str = f"{f1:.2f}" if isinstance(f1, float) else f1
        print(f"  {name:<43} {em_str:>8} {f1_str:>8}   {note}")

    print("─"*80)
    print("\n  * EM = Exact Match (%), F1 = Token F1 (%)")
    print("  * [MR] = Generative reader (LLM-based comparison, khác cơ chế với extractive QA)")

    # In markdown table cho báo cáo
    print("\n\n  [MARKDOWN TABLE cho báo cáo]")
    print("─"*80)
    print("| Mô hình | EM (%) | F1 (%) | Ghi chú |")
    print("|---------|--------|--------|---------|")
    for r in results:
        name = r['name']
        em = r.get('em', 'N/A')
        f1 = r.get('f1', 'N/A')
        note = r.get('note', '')
        em_str = f"{em:.2f}" if isinstance(em, float) else em
        f1_str = f"{f1:.2f}" if isinstance(f1, float) else f1
        print(f"| {name} | {em_str} | {f1_str} | {note} |")
    print()


def run_comparison(data_path: str, num_samples: int, batch_size: int,
                   skip_b2: bool, model_b2: str, model_finetuned: str):

    results = []

    # ---- B1: BM25-Only ----
    print("\n" + "="*60)
    print("  [1/3] Chạy Baseline B1: BM25-Only...")
    print("="*60)
    from src.models.baseline_bm25 import evaluate as b1_eval
    em_b1, f1_b1 = b1_eval(data_path, num_samples)
    results.append({
        'name': 'B1: BM25-Only (Rule-based)',
        'em': em_b1, 'f1': f1_b1,
        'note': 'Baseline tối thiểu — không dùng model'
    })

    # ---- B2: XLM-RoBERTa Pretrained ----
    if not skip_b2:
        print("\n" + "="*60)
        print("  [2/3] Chạy Baseline B2: XLM-RoBERTa Pretrained...")
        print("="*60)
        from src.models.baseline_pretrained import evaluate as b2_eval
        em_b2, f1_b2 = b2_eval(data_path, model_b2, batch_size, num_samples)
        results.append({
            'name': f'B2: XLM-RoBERTa (pretrained, no FT)',
            'em': em_b2, 'f1': f1_b2,
            'note': 'Off-the-shelf — chưa fine-tune trên ViSpanExtractQA'
        })
    else:
        print("\n  [2/3] Bỏ qua B2 (--skip_b2).")
        results.append({'name': 'B2: XLM-RoBERTa (pretrained, no FT)', 'em': 'N/A', 'f1': 'N/A',
                        'note': 'Chưa chạy'})

    # ---- M1: Fine-tuned XLM-RoBERTa (phương pháp chính) ----
    if model_finetuned:
        print("\n" + "="*60)
        print("  [3/3] Chạy Phương pháp chính: XLM-RoBERTa Fine-tuned...")
        print("="*60)
        from src.models.baseline_pretrained import evaluate as ft_eval
        em_ft, f1_ft = ft_eval(data_path, model_finetuned, batch_size, num_samples)
        results.append({
            'name': 'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)',
            'em': em_ft, 'f1': f1_ft,
            'note': 'Phương pháp chính — fine-tuned trên dữ liệu sạch'
        })
    else:
        print("\n  [3/3] Chưa có model fine-tuned. Để thêm: --model_finetuned <path>")
        results.append({
            'name': 'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)',
            'em': 'N/A', 'f1': 'N/A',
            'note': '← Sẽ điền sau khi fine-tune xong'
        })

    # ---- Qwen2.5 Generative Reader ----
    qwen_json = data_path.replace('.json', '_qwen_results.json')
    if os.path.exists(qwen_json):
        try:
            with open(qwen_json, encoding='utf-8') as f:
                qwen_data = json.load(f)
            results.append({
                'name': '[MR] Qwen2.5 + BM25 (Generative reader)',
                'em': qwen_data.get('exact_match', 'N/A'),
                'f1': qwen_data.get('token_f1', 'N/A'),
                'note': 'Generative LLM — khác cơ chế, nhánh so sánh mở rộng'
            })
        except Exception as e:
            results.append({
                'name': '[MR] Qwen2.5 + BM25 (Generative reader)',
                'em': 'Error', 'f1': 'Error',
                'note': f'Lỗi đọc file kết quả: {e}'
            })
    else:
        results.append({
            'name': '[MR] Qwen2.5 + BM25 (Generative reader)',
            'em': 'N/A', 'f1': 'N/A',
            'note': 'Generative LLM — chưa có kết quả đánh giá (chạy baseline_qwen.py)'
        })

    # In bảng so sánh
    print_comparison_table(results)

    # Lưu tổng hợp
    summary_path = str(Path(data_path).parent / 'comparison_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Tổng hợp đã lưu: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='So sánh tất cả mô hình QA')
    parser.add_argument('--data', type=str,
                        default='data/processed/test_clean.json')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Số mẫu test (mặc định: 500 để chạy nhanh)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--skip_b2', action='store_true',
                        help='Bỏ qua B2 nếu chưa có GPU/internet')
    parser.add_argument('--model_b2', type=str,
                        default='deepset/xlm-roberta-base-squad2',
                        help='Model B2 trên HuggingFace')
    parser.add_argument('--model_finetuned', type=str, default=None,
                        help='Đường dẫn đến model fine-tuned (để trống nếu chưa có)')
    # Chế độ đọc kết quả có sẵn (không chạy lại)
    parser.add_argument('--from_results', action='store_true',
                        help='Đọc kết quả JSON có sẵn thay vì chạy lại inference')
    parser.add_argument('--b1_json', type=str,
                        default='data/processed/test_clean_bm25only_results.json')
    parser.add_argument('--b2_json', type=str,
                        default='data/processed/test_clean_pretrained_deepset_xlm-roberta-base-squad2_results.json')
    parser.add_argument('--m1_json', type=str, default=None,
                        help='File kết quả M1 (sau khi fine-tune trên GPU xong)')
    parser.add_argument('--qwen_json', type=str,
                        default='data/processed/test_clean_qwen_results.json',
                        help='File kết quả Qwen2.5')
    args = parser.parse_args()

    if args.from_results:
        # Đọc kết quả có sẵn từ file JSON
        results = []
        for path, name, note in [
            (args.b1_json,  'B1: BM25-Only (Rule-based)',
             'Baseline tối thiểu — không dùng model'),
            (args.b2_json,  'B2: XLM-RoBERTa Pretrained (SQuAD2)',
             'Off-the-shelf, chưa fine-tune trên ViSpanExtractQA'),
            (args.m1_json,  'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)',
             'Phương pháp chính — fine-tuned trên dữ liệu sạch'),
            (args.qwen_json, '[MR] Qwen2.5 + BM25 (Generative reader)',
             'Generative LLM — khác cơ chế, nhánh so sánh mở rộng'),
        ]:
            if path and os.path.exists(path):
                data = load_result(path)
                results.append({
                    'name': name,
                    'em': data.get('exact_match', 'N/A'),
                    'f1': data.get('token_f1', 'N/A'),
                    'note': note,
                })
                print(f"  ✓ Đọc: {path}")
            else:
                results.append({'name': name, 'em': 'N/A', 'f1': 'N/A', 'note': '← Chưa có kết quả'})
                print(f"  - Chưa có: {path}")

        print_comparison_table(results)
    else:
        run_comparison(
            data_path=args.data,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            skip_b2=args.skip_b2,
            model_b2=args.model_b2,
            model_finetuned=args.model_finetuned,
        )
