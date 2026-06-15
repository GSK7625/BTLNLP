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
PHƯƠNG PHÁP CHÍNH (M1): Fine-tune XLM-RoBERTa trên ViSpanExtractQA
=============================================================
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)

from src.utils.metrics import compute_exact, compute_f1

# ─────────────────────────────────────────────────────────── #
#  Dataset
# ─────────────────────────────────────────────────────────── #

class ViQADataset(Dataset):
    """
    Dataset cho Extractive QA.
    Chuyển đổi character-level start/end → token-level start/end.
    """
    def __init__(self, data: list, tokenizer, max_length: int = 384, stride: int = 128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        skipped = 0
        for item in tqdm(data, desc="Tokenizing dataset", leave=False):
            question = item['question_raw']
            context  = item['context_raw']
            answer   = item['answer_text']
            char_start = item['start_index']
            char_end   = item['end_index']   # exclusive end

            # Tokenize với offset mapping
            encoding = tokenizer(
                question,
                context,
                max_length=max_length,
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # Lấy sequence_ids để phân biệt question/context tokens
            for i in range(len(encoding['input_ids'])):
                offsets = encoding['offset_mapping'][i]
                seq_ids = encoding.sequence_ids(i)

                # Tìm vị trí bắt đầu/kết thúc context trong tokens
                ctx_start_idx = next(j for j, s in enumerate(seq_ids) if s == 1)
                ctx_end_idx   = len(seq_ids) - 1 - next(
                    j for j, s in enumerate(reversed(seq_ids)) if s == 1)

                # Map char positions → token positions
                token_start = token_end = 0
                found = False
                for j in range(ctx_start_idx, ctx_end_idx + 1):
                    tok_char_start, tok_char_end = offsets[j]
                    if tok_char_start <= char_start < tok_char_end:
                        token_start = j
                    if tok_char_start < char_end <= tok_char_end:
                        token_end = j
                        found = True
                        break

                if not found:
                    # Answer không nằm trong window này (do truncation) → skip window
                    skipped += 1
                    continue

                self.samples.append({
                    'input_ids':      torch.tensor(encoding['input_ids'][i]),
                    'attention_mask': torch.tensor(encoding['attention_mask'][i]),
                    'start_positions': torch.tensor(token_start),
                    'end_positions':   torch.tensor(token_end),
                    'answer_text':    answer,
                    'context':        context,
                    'offsets':        offsets,
                    'seq_ids':        seq_ids,
                })

        print(f"  Dataset: {len(self.samples)} windows hợp lệ (bỏ qua {skipped} windows truncated)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids':       s['input_ids'],
            'attention_mask':  s['attention_mask'],
            'start_positions': s['start_positions'],
            'end_positions':   s['end_positions'],
        }


# ─────────────────────────────────────────────────────────── #
#  Inference helper
# ─────────────────────────────────────────────────────────── #

def predict_answer(model, tokenizer, question: str, context: str,
                   max_length: int = 384, device='cpu') -> str:
    """Trích xuất câu trả lời từ (câu hỏi, context) cho trước."""
    encoding = tokenizer(
        question, context,
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop('offset_mapping')[0]
    seq_ids = encoding.sequence_ids(0)

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits   = outputs.end_logits[0].cpu().numpy()

    # Chỉ xét context tokens
    ctx_indices = [i for i, s in enumerate(seq_ids) if s == 1]
    if not ctx_indices:
        return ""

    # Tìm cặp (start, end) có tổng logit cao nhất và start <= end
    best_score = float('-inf')
    best_start = best_end = ctx_indices[0]

    for s_idx in ctx_indices:
        for e_idx in ctx_indices:
            if e_idx < s_idx or (e_idx - s_idx) > 50:
                continue
            score = start_logits[s_idx] + end_logits[e_idx]
            if score > best_score:
                best_score = score
                best_start = s_idx
                best_end   = e_idx

    # Map token positions → character offsets → text
    char_start = offset_mapping[best_start][0]
    char_end   = offset_mapping[best_end][1]
    return context[char_start:char_end]


# ─────────────────────────────────────────────────────────── #
#  Training loop
# ─────────────────────────────────────────────────────────── #

def train(args):
    print(f"\n{'='*60}")
    print(f"  FINE-TUNE: XLM-RoBERTa trên ViSpanExtractQA")
    print(f"{'='*60}")

    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load tokenizer & model
    print(f"\n  Load model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    model.to(device)

    # Load dữ liệu train
    print(f"\n  Load dữ liệu train: {args.train_data}")
    with open(args.train_data, encoding='utf-8') as f:
        train_raw = json.load(f)

    if args.max_train_samples > 0:
        train_raw = train_raw[:args.max_train_samples]
        print(f"  Dùng {len(train_raw):,} mẫu train (--max_train_samples {args.max_train_samples})")
    else:
        print(f"  Dùng toàn bộ {len(train_raw):,} mẫu train")

    # Load dữ liệu val (để monitor training)
    print(f"\n  Load dữ liệu validation: {args.val_data}")
    with open(args.val_data, encoding='utf-8') as f:
        val_raw = json.load(f)
    val_raw = val_raw[:args.max_val_samples]
    print(f"  Dùng {len(val_raw):,} mẫu validation")

    # Tạo dataset
    print(f"\n  Tokenize train set...")
    train_dataset = ViQADataset(train_raw, tokenizer,
                                 max_length=args.max_length, stride=args.stride)
    print(f"  Tokenize val set...")
    val_dataset = ViQADataset(val_raw, tokenizer,
                               max_length=args.max_length, stride=args.stride)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\n  Cấu hình training:")
    print(f"    - Epochs        : {args.num_epochs}")
    print(f"    - Batch size    : {args.batch_size}")
    print(f"    - Learning rate : {args.learning_rate}")
    print(f"    - Max length    : {args.max_length}")
    print(f"    - Steps/epoch   : {len(train_loader)}")
    print(f"    - Total steps   : {total_steps}")

    # ── TRAINING ──────────────────────────────────────────── #
    best_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        # ── Train phase ──
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'avg_loss': f'{total_loss/(step+1):.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"\n  Epoch {epoch+1} — avg train loss: {avg_loss:.4f}")

        # ── Eval phase ──
        print(f"  Đánh giá trên {len(val_raw)} mẫu validation...")
        model.eval()
        em_scores, f1_scores = [], []

        eval_subset = val_raw[:min(300, len(val_raw))]  # nhanh hơn
        for item in tqdm(eval_subset, desc="  Eval", leave=False):
            pred = predict_answer(model, tokenizer,
                                  item['question_raw'], item['context_raw'],
                                  max_length=args.max_length, device=device)
            em_scores.append(compute_exact(item['answer_text'], pred))
            f1_scores.append(compute_f1(item['answer_text'], pred))

        em  = sum(em_scores) / len(em_scores) * 100
        f1  = sum(f1_scores) / len(f1_scores) * 100
        print(f"  Validation — EM: {em:.2f}%  F1: {f1:.2f}%")

        # Lưu model tốt nhất
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Saved best model → {args.output_dir} (F1={f1:.2f}%)")

    print(f"\n  Training xong! Best val F1 = {best_f1:.2f}%")
    print(f"  Model lưu tại: {args.output_dir}")
    return args.output_dir


# ─────────────────────────────────────────────────────────── #
#  Evaluation (full test set)
# ─────────────────────────────────────────────────────────── #

def evaluate_finetuned(model_path: str, test_data_path: str,
                        max_length: int = 384, num_samples: int = None):
    print(f"\n{'='*60}")
    print(f"  ĐÁNH GIÁ MODEL FINE-TUNED: {model_path}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()

    with open(test_data_path, encoding='utf-8') as f:
        test_raw = json.load(f)
    if num_samples:
        test_raw = test_raw[:num_samples]

    em_scores, f1_scores, error_cases = [], [], []

    for item in tqdm(test_raw, desc="Evaluating fine-tuned model"):
        pred = predict_answer(model, tokenizer,
                              item['question_raw'], item['context_raw'],
                              max_length=max_length, device=device)
        em = compute_exact(item['answer_text'], pred)
        f1 = compute_f1(item['answer_text'], pred)
        em_scores.append(em)
        f1_scores.append(f1)

        if em == 0:
            ctx = item['context_raw']
            error_cases.append({
                'id': item.get('id', '?'),
                'question': item['question_raw'],
                'context': ctx[:200] + '...' if len(ctx) > 200 else ctx,
                'gold': item['answer_text'],
                'predicted': pred,
                'f1': round(f1, 4),
            })

    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100

    print(f"\n{'─'*40}")
    print(f"  KẾT QUẢ M1 (XLM-RoBERTa Fine-tuned):")
    print(f"{'─'*40}")
    print(f"  Số mẫu       : {len(test_raw):,}")
    print(f"  Exact Match  : {avg_em:.2f}%")
    print(f"  Token F1     : {avg_f1:.2f}%")
    print(f"{'─'*40}")

    results = {
        'model': model_path,
        'description': 'XLM-RoBERTa fine-tuned on ViSpanExtractQA (cleaned)',
        'data_path': test_data_path,
        'num_samples': len(test_raw),
        'exact_match': round(avg_em, 4),
        'token_f1': round(avg_f1, 4),
        'error_analysis': error_cases,
    }
    suffix = f"_finetuned_{num_samples}samples_results.json" if num_samples else "_finetuned_results.json"
    out = test_data_path.replace('.json', suffix)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Kết quả lưu: {out}")

    # In ví dụ lỗi
    print(f"\n  VÍ DỤ LỖI (5 mẫu):")
    for i, err in enumerate(error_cases[:5], 1):
        print(f"\n  [{i}] ID={err['id']} | F1={err['f1']}")
        print(f"  Câu hỏi : {err['question']}")
        print(f"  Đúng    : {err['gold']}")
        print(f"  Dự đoán : {err['predicted']}")

    return avg_em, avg_f1


# ─────────────────────────────────────────────────────────── #
#  Entry point
# ─────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune XLM-RoBERTa cho Extractive QA')
    parser.add_argument('--mode', choices=['train', 'eval', 'train_eval'],
                        default='train_eval',
                        help='train / eval / train_eval (mặc định: train_eval)')
    parser.add_argument('--model_name', type=str,
                        default='deepset/xlm-roberta-base-squad2',
                        help='Model khởi điểm (pretrained checkpoint)')
    parser.add_argument('--train_data', type=str,
                        default='data/processed/train_clean.json')
    parser.add_argument('--val_data', type=str,
                        default='data/processed/validation_clean.json')
    parser.add_argument('--test_data', type=str,
                        default='data/processed/test_clean.json')
    parser.add_argument('--output_dir', type=str,
                        default='models/xlmroberta_finetuned',
                        help='Thư mục lưu model fine-tuned')
    parser.add_argument('--max_train_samples', type=int, default=5000,
                        help='Số mẫu train tối đa (dùng -1 để train toàn bộ)')
    parser.add_argument('--max_val_samples', type=int, default=500,
                        help='Số mẫu validation tối đa')
    parser.add_argument('--max_test_samples', type=int, default=None,
                        help='Số mẫu test (mặc định: toàn bộ)')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--stride', type=int, default=64)
    args = parser.parse_args()

    if args.mode in ('train', 'train_eval'):
        model_path = train(args)
    else:
        model_path = args.output_dir

    if args.mode in ('eval', 'train_eval'):
        if os.path.exists(model_path):
            evaluate_finetuned(model_path, args.test_data,
                               max_length=args.max_length,
                               num_samples=args.max_test_samples)
        else:
            print(f"\n  [ERROR] Không tìm thấy model tại: {model_path}")
            print("  Chạy mode 'train' trước hoặc chỉ định --output_dir đúng.")
