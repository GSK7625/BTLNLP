# -*- coding: utf-8 -*-
import sys
import io

if getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
=============================================================
DEMO: Vietnamese Extractive QA System
=============================================================
Hệ thống hỏi đáp tiếng Việt dựa trên:
  - BM25 Retriever: tìm đoạn văn liên quan (nếu tích hợp)
  - XLM-RoBERTa Reader: trích xuất câu trả lời chính xác

Cách dùng:
  python demo.py                    # Demo tương tác
  python demo.py --model models/xlmroberta_finetuned  # Dùng model fine-tuned
  python demo.py --model deepset/xlm-roberta-base-squad2  # Dùng pretrained
=============================================================
"""

import json
import os
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

from src.utils.metrics import normalize_answer, compute_f1

class ExtractiveReader:
    def __init__(self, model_name_or_path: str):
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU (CUDA)" if self.device == 0 else "CPU"
        print(f"  [Reader] Loading model: {model_name_or_path} | Device: {device_name}")
        
        try:
            self.pipeline = pipeline(
                "question-answering",
                model=model_name_or_path,
                tokenizer=model_name_or_path,
                device=self.device
            )
        except Exception as e:
            print(f"  [WARN] pipeline initialization failed: {e}. Trying direct AutoModel loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
            self.pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )
        print("  [Reader] Model loaded successfully.")

    def predict(self, question: str, context: str) -> dict:
        t0 = time.time()
        try:
            res = self.pipeline(question=question, context=context, max_answer_len=50)
            latency = (time.time() - t0) * 1000
            return {
                'answer': res.get('answer', '').strip(),
                'confidence': float(res.get('score', 0.0)),
                'char_start': int(res.get('start', 0)),
                'char_end': int(res.get('end', 0)),
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

DEMO_EXAMPLES = [
    {
        'question': 'Ai là chủ tịch tập đoàn Alibaba?',
        'context': 'Dư luận đang hết sức ngóng chờ sự kiện Chủ tịch tập đoàn thương mại điện tử Alibaba - Jack Ma (Mã Vân) đến Việt Nam, vậy Jack Ma là ai, Alibaba là tập đoàn thế nào mà lại có sức ảnh hưởng đến như vậy?',
        'gold': 'Jack Ma'
    },
    {
        'question': 'Thuật ngữ Big Bang do ai đề xuất?',
        'context': '1949, Fred Hoyle, một nhà toán học và thiên văn học nổi tiếng người Anh, trong một lần trả lời phỏng vấn của Đài BBC London vào Tháng 3, lần đầu tiên đã gieo thuật ngữ "Big Bang" để mô tả lý thuyết của Lemaître.',
        'gold': 'Fred Hoyle'
    },
    {
        'question': 'Người Châu Á đầu tiên nhận giải Fields là ai?',
        'context': 'Sau hơn 60 năm tồn tại, Fields Medal đã được trao cho 48 nhà toán học trên toàn thế giới. Nhà toán học Kunihiko Kodaira là người Nhật Bản và cũng là người châu Á đầu tiên giành Fields Medal.',
        'gold': 'Kunihiko Kodaira'
    },
    {
        'question': 'Ai sáng lập Uber?',
        'context': 'Người đồng sáng lập Uber, ông Travis Kalanick, chính thức trở thành tỷ phú, sau khi thu được 1,4 tỷ USD từ bán cổ phiếu.',
        'gold': 'Travis Kalanick'
    },
    {
        'question': 'Quang Hải được mệnh danh là gì?',
        'context': 'Quang Hải được mệnh danh là "Messi của Olympic Việt Nam". Tổng thống Hàn Quốc Moon Jae-in cũng bày tỏ sự ngưỡng mộ tài năng cầu thủ mang áo số 19.',
        'gold': 'Messi của Olympic Việt Nam'
    }
]

def run_demo(reader: ExtractiveReader, examples: list):
    print("\n" + "="*60)
    print("  Chạy các ví dụ demo có sẵn...")
    print("="*60)
    
    total = len(examples)
    correct = 0
    total_f1 = 0.0
    
    for i, ex in enumerate(examples, 1):
        q, ctx, gold = ex['question'], ex['context'], ex['gold']
        print(f"\n  ┌─ Câu hỏi {i}/{total} ───────────────────────────────────")
        print(f"  │  Q: {q}")
        print(f"  │  Context: {ctx[:80]}...")
        
        res = reader.predict(q, ctx)
        pred = res['answer']
        f1 = compute_f1(gold, pred)
        em = int(normalize_answer(gold) == normalize_answer(pred))
        
        correct += em
        total_f1 += f1
        
        print(f"  │  ✓ Dự đoán   : {pred}")
        print(f"  │    Gold      : {gold}")
        print(f"  │    EM={em} | F1={f1:.2f} | Conf={res['confidence']:.3f} | {res['latency_ms']}ms")
        print("  └─────────────────────────────────────────────")
        
    print("\n  " + "─"*45)
    print(f"  Tổng kết demo ({total} ví dụ):")
    print(f"    Exact Match : {(correct/total)*100:.1f}%")
    print(f"    Token F1    : {(total_f1/total)*100:.1f}%")
    print("  " + "─"*45)

def run_interactive(reader: ExtractiveReader):
    print("\n" + "="*60)
    print("  Chế độ tương tác (nhập 'quit' hoặc 'q' để thoát)")
    print("="*60)
    
    while True:
        try:
            print("\n" + "─"*50)
            question = input("  Câu hỏi: ").strip()
            if question.lower() in ('quit', 'exit', 'q', ''):
                print("  Thoát demo.")
                break
                
            context = input("  Ngữ cảnh: ").strip()
            if not context:
                print("  [!] Ngữ cảnh không được để trống.")
                continue
                
            res = reader.predict(question, context)
            print(f"\n  → Trả lời  : {res['answer']}")
            print(f"     Độ tin cậy: {res['confidence']:.3f}")
            print(f"     Thời gian : {res['latency_ms']} ms")
        except KeyboardInterrupt:
            print("\n  Thoát demo.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo Vietnamese Extractive QA')
    parser.add_argument('--model', type=str,
                        default='deepset/xlm-roberta-base-squad2',
                        help='Model path hoặc HuggingFace ID')
    parser.add_argument('--mode', choices=['demo', 'interactive', 'both'],
                        default='both',
                        help='demo=chạy ví dụ, interactive=nhập tay, both=cả hai')
    args = parser.parse_args()

    # Ưu tiên model fine-tuned nếu có
    finetuned_path = 'models/xlmroberta_finetuned'
    if args.model == 'deepset/xlm-roberta-base-squad2' and os.path.exists(finetuned_path) and os.listdir(finetuned_path):
        print(f"  [INFO] Tìm thấy model fine-tuned tại: {finetuned_path}")
        print(f"  [INFO] Sử dụng model fine-tuned thay vì pretrained mặc định.")
        args.model = finetuned_path

    reader = ExtractiveReader(args.model)

    if args.mode in ('demo', 'both'):
        run_demo(reader, DEMO_EXAMPLES)

    if args.mode in ('interactive', 'both'):
        run_interactive(reader)
