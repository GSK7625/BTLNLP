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



    # ---- B2: XLM-RoBERTa Pretrained ----
    if not skip_b2:
        print("\n" + "="*60)
        print("  [1/2] Chạy Baseline B2: XLM-RoBERTa Pretrained...")
        print("="*60)
        from src.models.baseline_pretrained import evaluate as b2_eval
        em_b2, f1_b2 = b2_eval(data_path, model_b2, batch_size, num_samples)
        results.append({
            'name': f'B2: XLM-RoBERTa (pretrained, no FT)',
            'em': em_b2, 'f1': f1_b2,
            'note': 'Off-the-shelf — chưa fine-tune trên ViSpanExtractQA'
        })
    else:
        print("\n  [1/2] Bỏ qua B2 (--skip_b2).")
        results.append({'name': 'B2: XLM-RoBERTa (pretrained, no FT)', 'em': 'N/A', 'f1': 'N/A',
                        'note': 'Chưa chạy'})

    # ---- M1: Fine-tuned XLM-RoBERTa (phương pháp chính) ----
    if model_finetuned:
        print("\n" + "="*60)
        print("  [2/2] Chạy Phương pháp chính: XLM-RoBERTa Fine-tuned...")
        print("="*60)
        from src.models.baseline_pretrained import evaluate as ft_eval
        em_ft, f1_ft = ft_eval(data_path, model_finetuned, batch_size, num_samples)
        results.append({
            'name': 'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)',
            'em': em_ft, 'f1': f1_ft,
            'note': 'Phương pháp chính — fine-tuned trên dữ liệu sạch'
        })
    else:
        print("\n  [2/2] Chưa có model fine-tuned. Để thêm: --model_finetuned <path>")
        results.append({
            'name': 'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)',
            'em': 'N/A', 'f1': 'N/A',
            'note': '← Sẽ điền sau khi fine-tune xong'
        })

    # ---- Pipeline Retriever-Reader ----
    print("\n" + "="*60)
    print("  Chạy hệ thống kết hợp Retriever-Reader...")
    print("="*60)
    try:
        from src.models.pipeline_retriever_reader import evaluate_pipeline
        # Run for Pretrained Pipeline
        res_pre_p = evaluate_pipeline(data_path, model_b2, num_samples, batch_size)
        results.append({
            'name': 'BM25 + XLM-R Pretrained (Pipeline)',
            'em': res_pre_p['exact_match'],
            'f1': res_pre_p['token_f1'],
            'note': f"BM25 Retriever + Pretrained Reader - BM25 Acc: {res_pre_p['retriever_accuracy']:.2f}%"
        })
        
        # Run for Fine-tuned Pipeline if model_finetuned is provided
        if model_finetuned:
            res_ft_p = evaluate_pipeline(data_path, model_finetuned, num_samples, batch_size)
            results.append({
                'name': 'BM25 + XLM-R Fine-tuned (Pipeline M1)',
                'em': res_ft_p['exact_match'],
                'f1': res_ft_p['token_f1'],
                'note': f"BM25 Retriever + M1 Reader - BM25 Acc: {res_ft_p['retriever_accuracy']:.2f}%"
            })
    except Exception as e:
        print(f"  [WARN] Lỗi khi chạy đánh giá Pipeline: {e}")

    # In bảng so sánh
    print_comparison_table(results)

    # Lưu tổng hợp
    suffix = f"_comparison_{num_samples}samples_results.json" if num_samples else "_comparison_results.json"
    summary_path = str(Path(data_path).parent / suffix)
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
    parser.add_argument('--m1_json', type=str,
                        default='data/processed/test_clean_finetuned_results.json',
                        help='File kết quả M1 (sau khi fine-tune trên GPU xong)')
    parser.add_argument('--pipeline_json', type=str,
                        default='data/processed/test_clean_pipeline_results.json',
                        help='File kết quả Pipeline Retriever-Reader')
    args = parser.parse_args()

    if args.from_results:
        # Ánh xạ động đường dẫn nếu để mặc định và có num_samples
        b1_path = args.b1_json
        if b1_path == 'data/processed/test_clean_bm25only_results.json' and args.num_samples:
            b1_path = args.data.replace('.json', f'_bm25only_{args.num_samples}samples_results.json')

        b2_path = args.b2_json
        if b2_path == 'data/processed/test_clean_pretrained_deepset_xlm-roberta-base-squad2_results.json' and args.num_samples:
            model_tag = args.model_b2.replace('/', '_')
            b2_path = args.data.replace('.json', f'_pretrained_{model_tag}_{args.num_samples}samples_results.json')

        m1_path = args.m1_json
        if m1_path == 'data/processed/test_clean_finetuned_results.json' and args.num_samples:
            m1_path = args.data.replace('.json', f'_finetuned_{args.num_samples}samples_results.json')

        # Fallback to search for fine-tuned model results if the exact file does not exist
        if not os.path.exists(m1_path):
            import glob
            parent_dir = os.path.dirname(m1_path)
            suffix = f"_{args.num_samples}samples_results.json" if args.num_samples else "_results.json"
            pattern = os.path.join(parent_dir, f"*finetuned*{suffix}")
            matching_files = glob.glob(pattern)
            if matching_files:
                m1_path = matching_files[0]

        pipeline_path = args.pipeline_json
        if pipeline_path == 'data/processed/test_clean_pipeline_results.json' and args.num_samples:
            pipeline_path = args.data.replace('.json', f'_pipeline_{args.num_samples}samples_results.json')

        # Đọc kết quả có sẵn từ file JSON
        results = []
        for path, name, note in [
            (b2_path,  'B2: XLM-RoBERTa Pretrained (SQuAD2)',
             'Off-the-shelf, chưa fine-tune trên ViSpanExtractQA'),
            (m1_path,  'M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)',
             'Phương pháp chính — fine-tuned trên dữ liệu sạch'),
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

        # Đọc kết quả Pipeline nếu có
        if pipeline_path and os.path.exists(pipeline_path):
            try:
                pipeline_data = load_result(pipeline_path)
                print(f"  ✓ Đọc kết quả Pipeline: {pipeline_path}")
                
                # Pretrained Pipeline
                if pipeline_data.get('pretrained_pipeline'):
                    pre_p = pipeline_data['pretrained_pipeline']
                    results.append({
                        'name': 'BM25 + XLM-R Pretrained (Pipeline)',
                        'em': pre_p.get('exact_match', 'N/A'),
                        'f1': pre_p.get('token_f1', 'N/A'),
                        'note': f"BM25 Retriever + Pretrained Reader - BM25 Acc: {pre_p.get('retriever_accuracy', 0):.2f}%"
                    })
                
                # Finetuned Pipeline
                if pipeline_data.get('finetuned_pipeline'):
                    ft_p = pipeline_data['finetuned_pipeline']
                    results.append({
                        'name': 'BM25 + XLM-R Fine-tuned (Pipeline M1)',
                        'em': ft_p.get('exact_match', 'N/A'),
                        'f1': ft_p.get('token_f1', 'N/A'),
                        'note': f"BM25 Retriever + M1 Reader - BM25 Acc: {ft_p.get('retriever_accuracy', 0):.2f}%"
                    })
            except Exception as e:
                print(f"  - Lỗi khi đọc kết quả Pipeline: {e}")
        else:
            print(f"  - Chưa có hoặc không tìm thấy kết quả Pipeline: {pipeline_path}")

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
