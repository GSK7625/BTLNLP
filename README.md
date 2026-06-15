# Vietnamese Extractive Question Answering — BM25 & XLM-RoBERTa

> **BÀI TẬP LỚN CUỐI KỲ: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)**
> 
> * **Đơn vị**: Khoa Công nghệ Thông tin - Trường Đại học Xây dựng Hà Nội (HUCE)
> * **Bài toán**: Extractive Question Answering (Hỏi đáp trích xuất tiếng Việt)
> * **Dữ liệu**: [ntphuc149/ViSpanExtractQA](https://huggingface.co/datasets/ntphuc149/ViSpanExtractQA)

---

## 1. Mô tả bài toán & Bối cảnh sử dụng

### Định nghĩa bài toán
* **Đầu vào (Input)**: Một **Câu hỏi** ($Q$) bằng tiếng Việt tự nhiên và một **Ngữ cảnh** ($C$) chứa thông tin liên quan đến câu hỏi.
* **Đầu ra (Output)**: Một **Phân đoạn văn bản** (Span) trích xuất trực tiếp từ ngữ cảnh $C$ đại diện cho câu trả lời chính xác nhất.
* **Mục tiêu**: Xây dựng mô hình Reader học cách dự đoán chỉ số bắt đầu (Start Index) và kết thúc (End Index) của câu trả lời trong chuỗi token ngữ cảnh.

### Bối cảnh ứng dụng thực tế
Hệ thống có thể được ứng dụng và tích hợp trực tiếp vào:
* Cổng thông tin hỗ trợ học vụ, hỗ trợ sinh viên tra cứu nhanh các câu hỏi quy chế từ văn bản hướng dẫn hành chính.
* Hệ thống tra cứu văn bản pháp luật, tài liệu nội bộ và quy trình doanh nghiệp.
* Công cụ chatbot thông minh trả lời dựa trên kho tri thức kết hợp (RAG - Retrieval-Augmented Generation).

---

## 2. Kiến trúc hệ thống & Pipeline xử lý

Hệ thống được thiết kế theo kiến trúc **Retriever–Reader** tích hợp đầy đủ từ tiền xử lý đến giao diện trực quan.

---

## 3. Cấu trúc thư mục dự án

```
BTLNLP/
│
├── data/                       # Dữ liệu dự án
│   └── processed/              # Dữ liệu sạch sau tiền xử lý và kết quả thực nghiệm
│       ├── train_clean.json        # Dữ liệu huấn luyện đã làm sạch (~196 MB)
│       ├── validation_clean.json   # Dữ liệu kiểm định đã làm sạch (~24.5 MB)
│       ├── test_clean.json         # Dữ liệu kiểm thử đã làm sạch (~24.5 MB)
│       ├── _comparison_500samples_results.json # Tệp so sánh các mô hình trên mốc 500 mẫu
│       ├── test_clean_pipeline_500samples_results.json # Kết quả chạy pipeline (500 mẫu)
│       └── test_clean_finetuned_results.json           # Kết quả huấn luyện M1 trên toàn bộ test set
│
├── docs/                       # Tài liệu môn học, đề bài PDF
│   └── Project cuối kỳ.pdf
│
├── models/                     # Thư mục lưu checkpoint mô hình
│   └── xlmroberta_finetuned/   # Trọng số mô hình chính M1 sau khi fine-tune (~1.1 GB)
│
├── src/                        # Mã nguồn chính của dự án (Python Package)
│   ├── __init__.py
│   │
│   ├── data/                   # Tiền xử lý & Phân tích EDA dữ liệu
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Tiền xử lý dữ liệu (Unicode NFC, sửa casing, tách từ)
│   │   └── eda.py              # Phân tích phân phối và cấu trúc tập dữ liệu
│   │
│   ├── models/                 # Huấn luyện và đánh giá mô hình
│   │   ├── __init__.py
│   │   ├── baseline_bm25.py    # Baseline B1: BM25-Only câu chứa từ khóa
│   │   ├── baseline_pretrained.py # Baseline B2 & Reader M1: Trích xuất span
│   │   ├── train.py            # Huấn luyện mô hình chính M1 (XLM-RoBERTa)
│   │   ├── pipeline_retriever_reader.py # Đánh giá hệ thống kết hợp BM25 + Reader
│   │   ├── evaluate.py         # Tổng hợp, so sánh kết quả và xuất markdown table
│   │   └── error_analysis.py   # Thống kê phân tích lỗi chi tiết xuất ra CSV
│   │
│   ├── utils/                  # Thư mục tiện ích dùng chung
│   │   ├── __init__.py
│   │   └── metrics.py          # Đo lường F1, EM và chuẩn hóa đáp án tiếng Việt
│   │
│   └── web/                    # Giao diện demo Web tương tác (Flask)
│       ├── static/             # CSS, JS, các assets tĩnh
│       ├── templates/          # HTML templates (index.html)
│       └── web_demo.py         # Backend server Flask
│
├── requirements.txt            # Danh sách các thư viện phụ thuộc
└── README.md                   # Hướng dẫn dự án (Tệp này)
```

---

## 4. Phân tích EDA dữ liệu

Trước khi tiến hành tiền xử lý, dự án thực hiện **Phân tích Khám phá Dữ liệu (EDA)** qua script [eda.py](file:///c:/Users/Kien/BTLNLP/src/data/eda.py) nhằm hiểu rõ cấu trúc và chất lượng dữ liệu thô:

- **Thống kê phân phối**: Kích thước từng split (train/validation/test), độ dài trung bình của context, câu hỏi và đáp án.
- **Quét vấn đề chất lượng**: Phát hiện mẫu null, câu hỏi trùng lặp, và tổng số đáp án không tìm thấy trong ngữ cảnh.
- **Phân tích chi tiết lỗi gán nhãn**: Phân loại nguyên nhân đáp án không khớp ngữ cảnh thành 4 nhóm:
  1. Lệch chữ hoa/thường (Case Mismatch)
  2. Khoảng trắng thừa (Whitespace Mismatch)
  3. Lệch chuẩn hóa Unicode NFC vs NFD
  4. Lỗi dịch máy / Paraphrase (không khắc phục được)
- **Trực quan hóa mẫu lỗi**: Highlight cụ thể vị trí lỗi trong ngữ cảnh để xác nhận trực quan từng loại.

Kết quả EDA là cơ sở trực tiếp xác định các bước xử lý cần thiết ở phần tiếp theo.

---

## 5. Quy trình tiền xử lý dữ liệu

Dựa trên insight từ EDA, tập dữ liệu gốc được làm sạch qua [preprocess.py](file:///c:/Users/Kien/BTLNLP/src/data/preprocess.py) với các bước:
1. **Chuẩn hóa Unicode (NFC)**: Đưa toàn bộ ký tự về dạng NFC để tránh lỗi không khớp chuỗi do sự khác biệt giữa ký tự tổ hợp và dựng sẵn (ví dụ: `hòa` vs `hoà`).
2. **Khôi phục lệch Casing**: Sửa các lỗi viết hoa/viết thường giữa nhãn câu trả lời gốc và ngữ cảnh, khôi phục thành công khoảng **15%** số mẫu lỗi gán nhãn trong tập dữ liệu gốc.
3. **Phân đoạn & Ánh xạ Token**: Sử dụng thư viện `underthesea` để tách từ tiếng Việt chuẩn cho mô hình từ khóa (BM25), đồng thời cấu hình `offset_mapping` của tokenizer để ánh xạ chính xác vị trí ký tự sang token ID tương ứng trong XLM-RoBERTa.

---

## 6. Hướng dẫn cài đặt & Vận hành

### Thiết lập môi trường ảo
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\activate          # Trên Windows (PowerShell/CMD)
# source .venv/bin/activate     # Trên Linux/Mac

# Nâng cấp pip và cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### Quy trình chạy thực nghiệm tuần tự

#### Bước 1: Phân tích EDA dữ liệu
```bash
python src/data/eda.py
```
*Phân tích phân phối, phát hiện lỗi gán nhãn và xác định chiến lược xử lý dữ liệu.*

#### Bước 2: Tiền xử lý và làm sạch dữ liệu
```bash
python src/data/preprocess.py
```
*Dữ liệu sạch sau khi tiền xử lý sẽ được lưu tự động tại thư mục `data/processed/`.*

#### Bước 3: Huấn luyện mô hình XLM-RoBERTa (M1)
*(Nên thực hiện trên môi trường có GPU để tối ưu hóa thời gian huấn luyện)*
```bash
python src/models/train.py \
    --mode train_eval \
    --max_train_samples -1 \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --output_dir models/xlmroberta_finetuned
```

#### Bước 4: Đánh giá mô hình (So sánh mốc 500 và mốc 5000 mẫu)
Hệ thống hỗ trợ chạy đánh giá với số lượng mẫu tùy chọn qua tham số `--num_samples`:
*   **Mốc 500 mẫu**: Tốc độ đánh giá nhanh, phù hợp kiểm thử nhanh trên CPU tại máy cá nhân.
*   **Mốc 5000 mẫu**: Đánh giá trên tập mẫu lớn hơn để số liệu có độ tin cậy và tính đại diện cao (nên chạy trên GPU).

##### Mốc 1: Chạy đánh giá 500 mẫu (Kiểm thử nhanh)
```bash
# Đánh giá Baseline B2 (XLM-R Pretrained)
python src/models/baseline_pretrained.py --model deepset/xlm-roberta-base-squad2 --num_samples 500

# Đánh giá Mô hình Fine-tuned M1 (Tinh chỉnh)
python src/models/baseline_pretrained.py --model models/xlmroberta_finetuned --num_samples 500

# Đánh giá hệ thống Pipeline (Retriever + Reader)
python src/models/pipeline_retriever_reader.py --num_samples 500

# Tổng hợp bảng so sánh kết quả mốc 500
python src/models/evaluate.py --num_samples 500 --from_results --m1_json data/processed/test_clean_pretrained_models_xlmroberta_finetuned_500samples_results.json

# Vẽ biểu đồ kết quả mốc 500 (lưu tại results/figures_500/)
python src/models/visualize_results.py --num_samples 500 --out_dir results/figures_500
```

##### Mốc 2: Chạy đánh giá 5000 mẫu (Kiểm thử quy mô lớn)
```bash
# Đánh giá Baseline B2 (XLM-R Pretrained)
python src/models/baseline_pretrained.py --model deepset/xlm-roberta-base-squad2 --num_samples 5000

# Đánh giá Mô hình Fine-tuned M1 (Tinh chỉnh)
python src/models/baseline_pretrained.py --model models/xlmroberta_finetuned --num_samples 5000

# Đánh giá hệ thống Pipeline (Retriever + Reader)
python src/models/pipeline_retriever_reader.py --num_samples 5000

# Tổng hợp bảng so sánh kết quả mốc 5000
python src/models/evaluate.py --num_samples 5000 --from_results --m1_json data/processed/test_clean_pretrained_models_xlmroberta_finetuned_5000samples_results.json

# Vẽ biểu đồ kết quả mốc 5000 (lưu tại results/figures_5000/)
python src/models/visualize_results.py --num_samples 5000 --out_dir results/figures_5000
```

#### Bước 5: Phân tích lỗi định lượng (CSV)
Xuất tệp phân tích lỗi định lượng dạng CSV từ kết quả dự đoán của M1:
```bash
# Phân tích lỗi cho mốc 500 mẫu
python src/models/error_analysis.py \
    --m1_results data/processed/test_clean_pretrained_models_xlmroberta_finetuned_500samples_results.json \
    --output_csv error_analysis_500.csv

# Phân tích lỗi cho mốc 5000 mẫu
python src/models/error_analysis.py \
    --m1_results data/processed/test_clean_pretrained_models_xlmroberta_finetuned_5000samples_results.json \
    --output_csv error_analysis_5000.csv
```

#### Bước 6: Khởi động Web Demo trực quan
Khởi động máy chủ Flask tại local phục vụ giao diện hỏi đáp trực tuyến:
```bash
python src/web/web_demo.py
```
*Sau khi khởi động, truy cập ứng dụng thông qua trình duyệt tại địa chỉ: [http://127.0.0.1:5000](http://127.0.0.1:5000).*

---

## 7. Kết quả thực nghiệm

### 1. Đánh giá trên mốc 500 mẫu kiểm thử (Mốc kiểm thử nhanh)
Dưới đây là bảng so sánh hiệu năng của các phương pháp trên **500 mẫu** đầu tiên trích xuất từ tập dữ liệu sạch `test_clean.json`:

| Mô hình / Phương pháp | EM (%) | F1 (%) | Cơ chế xử lý | Ghi chú thực nghiệm |
| :--- | :---: | :---: | :---: | :--- |
| **B2: XLM-RoBERTa Pretrained (SQuAD2)** | 44.60 | 70.39 | Trích xuất (Deepset) | Mô hình cơ sở chưa qua huấn luyện thích nghi trên ViSpanExtractQA. |
| **M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)** | **60.60** | **81.05** | Trích xuất tối ưu | Mô hình Reader độc lập phục vụ so sánh .|
| **BM25 + XLM-R Pretrained (Pipeline)** | 37.80 | 61.35 | Retriever + Reader | Tích hợp Top-5 đoạn văn bản (Retriever Acc: 95.00%). |
| **BM25 + XLM-R Fine-tuned (Pipeline M1)** | **53.80** | **71.95** | **Retriever + M1 Reader** | **Phương pháp đề xuất chính của nhóm |

### 2. Đánh giá trên mốc 5000 mẫu kiểm thử (Mốc kiểm chứng quy mô lớn)
Dưới đây là bảng so sánh hiệu năng của các phương pháp trên **5000 mẫu** trích xuất từ tập dữ liệu sạch `test_clean.json`:

| Mô hình / Phương pháp | EM (%) | F1 (%) | Cơ chế xử lý | Ghi chú thực nghiệm |
| :--- | :---: | :---: | :---: | :--- |
| **B2: XLM-RoBERTa Pretrained (SQuAD2)** | 44.3 | 66.5 | Trích xuất (Deepset) | Mô hình cơ sở chưa qua huấn luyện thích nghi trên ViSpanExtractQA. |
| **M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)** | **56.5** | **76.1** | **Trích xuất tối ưu** | **Mô hình đề xuất chính, tinh chỉnh trên dữ liệu tiếng Việt sạch.** |
| **BM25 + XLM-R Pretrained (Pipeline)** | 34.9 | 52.4 | Retriever + Reader | Tích hợp Top-5 đoạn văn bản (Retriever Acc: 85.3%). |
| **BM25 + XLM-R Fine-tuned (Pipeline M1)** | **42.4** | **57.5** | Retriever + M1 Reader | Tích hợp Top-5 đoạn văn bản kết hợp Rank Penalty (Hình phạt thứ tự). |

> [!NOTE]
> * **Exact Match (EM)**: Tỷ lệ phần trăm câu trả lời dự đoán trùng khớp hoàn toàn từng ký tự với nhãn gốc (sau khi chuẩn hóa).
> * **Token F1**: Đo mức độ trùng khớp mức độ từ (token-level) giữa câu trả lời dự đoán và nhãn gốc.

### 3. Hiệu năng của mô hình chính M1 trên toàn bộ dữ liệu kiểm thử
Khi được kiểm chứng trên **toàn bộ 10,275 mẫu** của tập dữ liệu kiểm thử (`test_clean.json`), mô hình **M1: XLM-RoBERTa Fine-tuned** đạt kết quả vượt trội và ổn định:
* **Exact Match (EM)**: **59.53%**
* **Token F1**: **76.31%**

---

## 8. Phân tích lỗi định lượng

Dựa trên kết quả phân tích lỗi của mô hình M1 trên tập kiểm thử 5000 mẫu, các loại lỗi được phân bổ cụ thể như sau (số liệu thống kê chính thức):

*   **Lỗi biên (Span dư thừa/thiếu)**: **75.0%** (Phần lớn do lỗi trích xuất dư chức danh/mô tả).
*   **Sai span hoàn toàn (đáp án có trong ngữ cảnh)**: **25.0%** (Model chọn sai vùng dữ liệu do ngữ cảnh dài hoặc nhiều thực thể gây nhiễu).

### Các ví dụ lỗi tiêu biểu:
1. **Lỗi biên (dư thừa từ ngữ cảnh) - 75.0%**: 
   * *Câu hỏi*: "Ai là bộ trưởng bộ quốc phòng Việt Nam?"
   * *Nhãn gốc*: `Ngô Xuân Lịch`
   * *Mô hình dự đoán*: `Đại tướng Ngô Xuân Lịch` (Dư thừa chức danh quân hàm).
2. **Sai lệch span hoàn toàn - 25.0%**:
   * Thường xuất hiện trong các đoạn văn dài chứa nhiều thực thể có cùng kiểu (ví dụ: đoạn văn có nhiều tên người hoặc nhiều mốc thời gian khác nhau) dẫn đến mô hình bị phân tán xác suất trích xuất.

### Hướng cải thiện đề xuất:
* **Hậu xử lý (Post-processing)**: Xây dựng bộ lọc loại bỏ các chức danh, danh xưng tiếng Việt thông dụng (`ông`, `bà`, `Đại tướng`, `Giám đốc`...) khỏi câu trả lời được trích xuất.
* **Mở rộng Context Length**: Tăng độ dài chuỗi tối đa (`max_length`) lên 384 hoặc 512 token trong quá trình huấn luyện và inference để giữ đầy đủ ngữ cảnh của các đoạn văn dài.


