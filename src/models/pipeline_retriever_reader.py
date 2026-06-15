# -*- coding: utf-8 -*-
import sys
import os
import io
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
from rank_bm25 import BM25Okapi
import transformers
from transformers import pipeline

# Setup path so we can import src even if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.utils.metrics import normalize_answer, compute_exact, compute_f1

def evaluate_pipeline(data_path: str, model_name: str, num_samples: int = None, batch_size: int = 16, top_k: int = 5, rank_penalty: float = None):
    """
    Chạy đánh giá pipeline tích hợp với cải tiến Top-K Retrieval:
      1. BM25 truy hồi Top-K đoạn văn liên quan nhất.
      2. Transformer Reader chạy trên cả K đoạn văn, chọn câu trả lời có độ tin cậy (confidence score) cao nhất.
    """
    # Tự động phát hiện rank_penalty phù hợp nếu để None
    if rank_penalty is None:
        rank_penalty = 0.5 if "finetuned" in model_name.lower() else 0.0
        
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    
    print(f"\nĐang nạp dữ liệu kiểm thử: {data_path}...")
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]
        print(f"Chạy thử trên {num_samples} mẫu đầu tiên.")

    total = len(data)
    
    # ── 1. TẠO CORPUS NGỮ CẢNH ĐỘC NHẤT ─────────────────────── #
    unique_contexts = {}
    for item in data:
        ctx_raw = item['context_raw']
        ctx_bm25 = item['context_bm25']
        if ctx_raw not in unique_contexts:
            unique_contexts[ctx_raw] = ctx_bm25
            
    corpus_raw = list(unique_contexts.keys())
    corpus_bm25 = [unique_contexts[c] for c in corpus_raw]
    
    print(f"Tổng hợp được {len(corpus_raw):,} ngữ cảnh độc nhất làm Kho tài liệu (Corpus).")
    
    # Lập chỉ mục BM25
    tokenized_corpus = [c.split() for c in corpus_bm25]
    bm25 = BM25Okapi(tokenized_corpus)
    print("✓ Đã lập chỉ mục BM25 Okapi thành công.")

    # ── 2. CHẠY BỘ TRUY HỒI (BM25 RETRIEVER TOP-K) ───────────── #
    print(f"\n[Bước 1] Đang chạy BM25 Retriever truy hồi Top-{top_k} đoạn văn...")
    retrieved_contexts_batch = [] # List of list of strings
    retriever_correct = 0
    rr_scores = []
    
    # Khởi tạo đếm chính xác thực tế cho các mốc K
    k_targets = [1, 3, 5]
    retriever_correct_k = {k: 0 for k in k_targets}
    
    for item in tqdm(data, desc="BM25 Retrieval"):
        q_tokens = item['question_bm25'].split()
        gold_ctx = item['context_raw']
        
        # Lấy điểm số BM25
        scores = bm25.get_scores(q_tokens)
        # Lấy chỉ số của Top-K đoạn có score cao nhất
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        top_k_ctxs = [corpus_raw[idx] for idx in top_k_indices]
        
        retrieved_contexts_batch.append(top_k_ctxs)
        
        # Tính Top-K accuracy và MRR
        if gold_ctx in top_k_ctxs:
            retriever_correct += 1
            rank = top_k_ctxs.index(gold_ctx) + 1
            rr_scores.append(1.0 / rank)
            
            # Thống kê thực tế cho từng mốc K
            for k in k_targets:
                if rank <= k:
                    retriever_correct_k[k] += 1
        else:
            rr_scores.append(0.0)
            
    retriever_acc = retriever_correct / total * 100
    retriever_mrr = sum(rr_scores) / total * 100
    
    # Tính phần trăm chính xác cho từng mốc
    retriever_accs_k = {k: (retriever_correct_k[k] / total * 100) for k in k_targets}
    
    print(f"-> Độ chính xác của Retriever (Top-{top_k} Accuracy): {retriever_acc:.2f}% ({retriever_correct}/{total})")
    for k in k_targets:
        if k <= top_k:
            print(f"   - Top-{k} Accuracy (Thực tế): {retriever_accs_k[k]:.2f}% ({retriever_correct_k[k]}/{total})")
    print(f"-> Mean Reciprocal Rank (MRR@{top_k}): {retriever_mrr:.2f}%")

    # ── 3. CHẠY BỘ ĐỌC (TRANSFORMER READER TRÊN TOP-K) ─────────── #
    print(f"\n[Bước 2] Đang nạp mô hình Reader: {model_name} | Device: {device_name}...")
    t_load = time.time()
    
    # Chuẩn bị pipeline hỏi đáp
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
        print(f"  [WARN] pipeline('{_task}') thất bại: {e}. Thử dùng AutoModel trực tiếp...")
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    print(f"✓ Mô hình loaded thành công trong {time.time() - t_load:.1f}s!\n")

    # Tạo danh sách phẳng các cặp (câu hỏi, ngữ cảnh) để chạy batch
    flat_inputs = []
    for item, top_k_ctxs in zip(data, retrieved_contexts_batch):
        for ctx in top_k_ctxs:
            flat_inputs.append({'question': item['question_raw'], 'context': ctx})
            
    flat_total = len(flat_inputs)
    print(f"Tiến hành trích xuất câu trả lời trên {flat_total} cặp đầu vào (batch_size={batch_size})...")
    
    flat_predictions = []
    for i in tqdm(range(0, flat_total, batch_size), desc="Reader Inference"):
        batch_inputs = flat_inputs[i:i+batch_size]
        try:
            batch_results = qa_pipeline(
                question=[inp['question'] for inp in batch_inputs],
                context=[inp['context'] for inp in batch_inputs],
                max_answer_len=50
            )
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            flat_predictions.extend(batch_results)
        except Exception as e:
            # Fallback chạy từng câu nếu lỗi batch
            for inp in batch_inputs:
                try:
                    res = qa_pipeline(
                        question=inp['question'],
                        context=inp['context'],
                        max_answer_len=50
                    )
                    flat_predictions.append(res)
                except:
                    flat_predictions.append({'answer': '', 'score': 0.0})

    # Tính metric EM và F1 bằng cách chọn phương án có score tốt nhất trong Top-K
    em_scores = []
    f1_scores = []
    error_cases = []
    
    for idx, item in enumerate(data):
        gold = item['answer_text']
        
        # Phân mảnh dự đoán tương ứng với câu hỏi thứ idx
        start_idx = idx * top_k
        end_idx = start_idx + top_k
        item_preds = flat_predictions[start_idx:end_idx]
        item_ctxs = retrieved_contexts_batch[idx]
        
        # Chọn ứng cử viên có score (confidence) cao nhất sau khi trừ đi penalty theo rank
        best_score = -9999.0
        best_pred = {'answer': '', 'score': 0.0}
        best_ctx = item_ctxs[0]
        
        for rank_idx, (pred, ctx) in enumerate(zip(item_preds, item_ctxs)):
            score = pred.get('score', 0.0) if pred else 0.0
            # Áp dụng rank penalty để tránh overconfidence của model fine-tuned trên context sai
            score = score - rank_idx * rank_penalty
            if score > best_score:
                best_score = score
                best_pred = pred
                best_ctx = ctx
                
        pred_answer = best_pred.get('answer', '')
        conf_score = best_pred.get('score', 0.0)
        
        em = compute_exact(gold, pred_answer)
        f1 = compute_f1(gold, pred_answer)
        
        em_scores.append(em)
        f1_scores.append(f1)
        
        # Ghi nhận trường hợp lỗi
        if em == 0 and len(error_cases) < 20:
            error_cases.append({
                'id': item.get('id', idx),
                'question': item['question_raw'],
                'gold_context': item['context_raw'],
                'retrieved_context': best_ctx[:200] + '...' if len(best_ctx) > 200 else best_ctx,
                'retriever_correct': bool(best_ctx == item['context_raw']),
                'gold': gold,
                'predicted': pred_answer,
                'confidence': round(conf_score, 4),
                'f1': round(f1, 4),
            })
            
    avg_em = sum(em_scores) / total * 100
    avg_f1 = sum(f1_scores) / total * 100
    
    # Giải phóng bộ nhớ GPU/RAM
    del qa_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        'exact_match': round(avg_em, 4),
        'token_f1': round(avg_f1, 4),
        'retriever_accuracy': round(retriever_acc, 4),
        'retriever_accuracy_k': {str(k): round(v, 4) for k, v in retriever_accs_k.items()},
        'retriever_mrr': round(retriever_mrr, 4),
        'error_analysis': error_cases
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Đánh giá Pipeline kết hợp BM25 + Transformer Reader')
    parser.add_argument('--data', type=str, default='data/processed/test_clean.json')
    parser.add_argument('--model_pretrained', type=str, default='deepset/xlm-roberta-base-squad2')
    parser.add_argument('--model_finetuned', type=str, default='models/xlmroberta_finetuned')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=5, help='Số lượng đoạn văn truy hồi tối đa')
    parser.add_argument('--num_samples', type=int, default=None, help='Số lượng mẫu test để chạy')
    parser.add_argument('--rank_penalty', type=float, default=0.5, help='Hình phạt tin cậy theo thứ tự rank BM25 để giảm overconfidence của model fine-tuned')
    args = parser.parse_args()
    
    results = {}
    
    # ── 1. Chạy Pretrained Pipeline ────────────────────────── #
    print("\n" + "="*80)
    print(f"  [1/2] ĐÁNH GIÁ PIPELINE: BM25 Retriever (Top-{args.top_k}) + XLM-RoBERTa Pretrained")
    print("="*80)
    res_pre = evaluate_pipeline(args.data, args.model_pretrained, args.num_samples, args.batch_size, args.top_k, rank_penalty=0.0)
    results['pretrained_pipeline'] = res_pre
    
    # ── 2. Chạy Fine-tuned M1 Pipeline ─────────────────────── #
    if os.path.exists(args.model_finetuned) and os.listdir(args.model_finetuned):
        print("\n" + "="*80)
        print(f"  [2/2] ĐÁNH GIÁ PIPELINE: BM25 Retriever (Top-{args.top_k}) + XLM-RoBERTa Fine-tuned (M1)")
        print("="*80)
        res_ft = evaluate_pipeline(args.data, args.model_finetuned, args.num_samples, args.batch_size, args.top_k, rank_penalty=args.rank_penalty)
        results['finetuned_pipeline'] = res_ft
    else:
        print(f"\n[WARN] Không tìm thấy thư mục model fine-tuned tại: {args.model_finetuned}")
        print("Bỏ qua đánh giá pipeline với M1.")
        results['finetuned_pipeline'] = None

    # ── 3. LƯU KẾT QUẢ VÀ IN BẢNG SO SÁNH ───────────────────── #
    suffix = f"_pipeline_{args.num_samples}samples_results.json" if args.num_samples else "_pipeline_results.json"
    out_path = args.data.replace('.json', suffix)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu kết quả đánh giá pipeline tại: {out_path}")

    # In bảng kết quả so sánh
    print("\n" + "="*80)
    print(f"  KẾT QUẢ ĐÁNH GIÁ PIPELINE RETRIEVER-READER (BM25 Top-{args.top_k} + Reader)")
    print("="*80)
    print(f"  Retriever Top-{args.top_k} Accuracy: {res_pre['retriever_accuracy']:.2f}%")
    print(f"  Retriever MRR@{args.top_k}         : {res_pre['retriever_mrr']:.2f}%")
    print("─"*80)
    print(f"  {'Mô hình Reader kết hợp':<45} {'EM (%)':>8} {'F1 (%)':>8}")
    print("─"*80)
    print(f"  XLM-RoBERTa Pretrained + BM25 Retriever       {res_pre['exact_match']:>8.2f} {res_pre['token_f1']:>8.2f}")
    if results['finetuned_pipeline']:
        res_ft = results['finetuned_pipeline']
        print(f"  XLM-RoBERTa Fine-tuned (M1) + BM25 Retriever  {res_ft['exact_match']:>8.2f} {res_ft['token_f1']:>8.2f}")
    print("─"*80)
    
    # In dạng Markdown Table
    print("\n  [MARKDOWN TABLE]")
    print(f"  | Mô hình Pipeline kết hợp | Retriever Top-{args.top_k} (%) | MRR@{args.top_k} (%) | EM (%) | F1 (%) |")
    print("  |---|---|---|---|---|")
    print(f"  | BM25 + XLM-R Pretrained | {res_pre['retriever_accuracy']:.2f}% | {res_pre['retriever_mrr']:.2f}% | {res_pre['exact_match']:.2f}% | {res_pre['token_f1']:.2f}% |")
    if results['finetuned_pipeline']:
        print(f"  | BM25 + XLM-R Fine-tuned (M1) | {res_ft['retriever_accuracy']:.2f}% | {res_ft['retriever_mrr']:.2f}% | {res_ft['exact_match']:.2f}% | {res_ft['token_f1']:.2f}% |")
    print()
