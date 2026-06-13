# -*- coding: utf-8 -*-
import os
import re
import sys
import io
import unicodedata
import pandas as pd

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from datasets import load_dataset
from underthesea import word_tokenize
from functools import lru_cache
from tqdm import tqdm

# Force UTF-8 encoding for stdout to print Vietnamese text cleanly on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 1. TIỀN XỬ LÝ CHO RETRIEVER (BM25) ---
@lru_cache(maxsize=100000)
def preprocess_for_retriever(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    # Chỉ giữ lại ký tự chữ cái tiếng Việt, chữ số và khoảng trắng
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return word_tokenize(text.strip(), format="text")

# --- 2. HÀM CĂN CHỈNH VÀ KHÔI PHỤC ĐÁP ÁN (READER ALIGNMENT) ---
def clean_and_align_sample(sample, idx):
    context = sample.get('context', '')
    question = sample.get('question', '')
    
    # Lấy đáp án từ dictionary 'answers' hoặc trường phẳng 'answer_text'
    answers_dict = sample.get('answers', {})
    if answers_dict and len(answers_dict.get('text', [])) > 0:
        answer_text = answers_dict['text'][0]
    else:
        answer_text = sample.get('answer_text', '')
        
    if not answer_text or not context:
        return None, "empty"

    # Bước A: Chuẩn hóa Unicode NFC để tránh lỗi font chữ tiếng Việt lệch mã dựng sẵn/tổ hợp
    context_norm = unicodedata.normalize('NFC', context)
    answer_norm = unicodedata.normalize('NFC', answer_text)
    question_norm = unicodedata.normalize('NFC', question)
    
    # Loại bỏ khoảng trắng thừa đầu cuối của đáp án
    answer_stripped = answer_norm.strip()

    # Bước B: So khớp trực tiếp (Exact Match)
    if answer_stripped in context_norm:
        start_idx = context_norm.find(answer_stripped)
        end_idx = start_idx + len(answer_stripped)
        return {
            'id': sample.get('id', idx),
            'question_raw': question_norm,
            'context_raw': context_norm,
            'answer_text': answer_stripped,
            'start_index': start_idx,
            'end_index': end_idx
        }, "exact_match"
        
    # Bước C: So khớp không phân biệt chữ hoa/thường (Casing Match) để cứu dữ liệu
    start_idx = context_norm.lower().find(answer_stripped.lower())
    if start_idx != -1:
        # Trích xuất đoạn văn bản gốc tương ứng trong ngữ cảnh
        aligned_answer = context_norm[start_idx : start_idx + len(answer_stripped)]
        end_idx = start_idx + len(aligned_answer)
        return {
            'id': sample.get('id', idx),
            'question_raw': question_norm,
            'context_raw': context_norm,
            'answer_text': aligned_answer,  # Đáp án mới trùng khớp hoàn hảo với context
            'start_index': start_idx,
            'end_index': end_idx
        }, "recovered_casing"

    # Không thể khớp (lỗi dịch máy hoặc paraphrase hoàn toàn)
    return None, "invalid_translation"

# --- 3. PIPELINE CHẠY ĐỒNG BỘ TRÊN CÁC TẬP DỮ LIỆU ---
def preprocess_dataset():
    print("--- Đang tải dữ liệu ViSpanExtractQA từ Hugging Face... ---")
    raw_datasets = load_dataset("ntphuc149/ViSpanExtractQA")
    
    output_dir = os.path.abspath("data/processed")
    os.makedirs(output_dir, exist_ok=True)
    
    for split in raw_datasets.keys():
        print(f"\n[Xử lý phân vùng: {str(split).upper()}]")
        dataset = raw_datasets[split]
        
        clean_samples = []
        stats = {"exact_match": 0, "recovered_casing": 0, "invalid_translation": 0, "empty": 0}
        
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            processed, status = clean_and_align_sample(sample, idx)
            stats[status] += 1
            
            if processed:
                # Tạo thêm trường đã được xử lý cho BM25
                processed['question_bm25'] = preprocess_for_retriever(processed['question_raw'])
                processed['context_bm25'] = preprocess_for_retriever(processed['context_raw'])
                clean_samples.append(processed)
                
        # Thống kê kết quả
        print(f" Kết quả xử lý tập {split}:")
        print(f"  - Tổng số mẫu gốc: {len(dataset)}")
        print(f"  - Số mẫu giữ lại (Sạch): {len(clean_samples)}")
        print(f"  - Số mẫu khớp hoàn hảo: {stats['exact_match']}")
        print(f"  - Số mẫu khôi phục thành công (lệch chữ hoa/thường): {stats['recovered_casing']}")
        print(f"  - Số mẫu bị loại bỏ (lỗi dịch): {stats['invalid_translation']}")
        
        # Lưu ra file JSON để sẵn sàng sử dụng
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(clean_samples)
        output_file = os.path.join(output_dir, f"{str(split)}_clean.json")
        df.to_json(output_file, orient='records', force_ascii=False, indent=4)
        print(f" -> Đã lưu tập sạch vào: {output_file}")

if __name__ == "__main__":
    preprocess_dataset()
