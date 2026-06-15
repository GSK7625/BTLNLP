# DÀN Ý & NỘI DUNG SLIDE THUYẾT TRÌNH BẢO VỆ DỰ ÁN NLP
**ĐỀ TÀI: VIETNAMESE EXTRACTIVE QUESTION ANSWERING VIA RETRIEVER-READER PIPELINE**

---

### Slide 1: Trang bìa (Giới thiệu chung)
* **Tiêu đề**: Xây dựng Hệ thống Hỏi đáp Trích xuất Tiếng Việt dựa trên mô hình Retriever-Reader.
* **Môn học**: Xử lý Ngôn ngữ Tự nhiên (NLP) - Trường Đại học Xây dựng Hà Nội.
* **Nhóm thực hiện**: Nhóm `[Điền tên nhóm]`
* **Thành viên**:
  1. `[Tên thành viên 1]` - Lớp `[Lớp]` (Trưởng nhóm)
  2. `[Tên thành viên 2]` - Lớp `[Lớp]`
  3. `[Tên thành viên 3]` - Lớp `[Lớp]`
  4. `[Tên thành viên 4]` - Lớp `[Lớp]`
  5. `[Tên thành viên 5]` - Lớp `[Lớp]`
* **Kịch bản thuyết trình (Speaker Notes)**: *"Kính chào thầy cô và các bạn, hôm nay nhóm chúng em xin trình bày về dự án cuối kỳ môn NLP với đề tài Hỏi đáp trích xuất tiếng Việt sử dụng kiến trúc Retriever-Reader kết hợp BM25 và XLM-RoBERTa..."*

---

### Slide 2: Bài toán & Bối cảnh sử dụng
* **Định nghĩa bài toán**: Extractive QA (Hỏi đáp trích xuất).
  * **Đầu vào**: Câu hỏi $Q$ và Ngữ cảnh $C$.
  * **Đầu ra**: Trích xuất chính xác một phân đoạn câu trả lời ngắn (Answer Span) trực tiếp từ $C$.
* **Bối cảnh sử dụng**:
  * Tích hợp vào hệ thống hỗ trợ học vụ, tự động tra cứu văn bản quy định của nhà trường để trả lời thắc mắc của sinh viên.
  * Ứng dụng trong RAG (Retrieval-Augmented Generation) để cung cấp tài liệu tham khảo đáng tin cậy, tránh hiện tượng LLM bịa đặt thông tin.
* **Kịch bản thuyết trình (Speaker Notes)**: *"Bài toán hỏi đáp trích xuất yêu cầu mô hình tìm đúng chỉ số bắt đầu và kết thúc của đáp án trong văn bản. Ứng dụng thực tế lớn nhất của nó là làm bộ đọc thông tin cho hệ thống RAG tra cứu văn bản nội bộ..."*

---

### Slide 3: Phân tích Dữ liệu thực nghiệm
* **Tập dữ liệu**: `ntphuc149/ViSpanExtractQA` (Quy mô: 121.488 mẫu).
  * **Train**: 97.189 mẫu (80%) | **Val**: 12.147 mẫu (10%) | **Test**: 12.152 mẫu (10%).
* **Thống kê độ dài**: Context trung bình **166,6 từ** | Question trung bình **13,4 từ** | Answer trung bình **5,3 từ**.
* **Các lỗi nghiêm trọng phát hiện trên dữ liệu gốc**:
  * **18% dữ liệu** bị lỗi lệch vị trí đáp án (không xuất hiện nguyên văn trong ngữ cảnh).
  * *Nguyên nhân*: Lệch chữ hoa/thường (13%), khoảng trắng thừa (1%), và lỗi dịch máy tự động từ SQuAD2 tiếng Anh (86%).
* **Kịch bản thuyết trình (Speaker Notes)**: *"Chúng em sử dụng tập dữ liệu ViSpanExtractQA gồm hơn 120.000 mẫu. Quá trình EDA tự động phát hiện tới 18% dữ liệu gốc bị lỗi không khớp ngữ cảnh, phần lớn do lỗi dịch máy từ SQuAD2 sang tiếng Việt..."*

---

### Slide 4: Quy trình Tiền xử lý Dữ liệu
* **Chuẩn hóa Unicode**: Đưa toàn bộ context/question/answer về chuẩn **NFC** để tránh lỗi gán nhãn do ký tự tổ hợp (ví dụ: `hòa` vs `hoà`).
* **Căn chỉnh & Khôi phục Casing**:
  * Tự động tìm kiếm chuỗi không phân biệt hoa thường và khôi phục nhãn gốc trùng khớp với context (cứu được 13% mẫu lỗi chữ hoa thường).
  * Loại bỏ triệt để các mẫu bị lỗi dịch máy sai từ vựng không thể khôi phục (~15.413 mẫu ở tập train).
* **Tách từ tiếng Việt**: Sử dụng `underthesea` cho mô hình BM25 để cải thiện khớp từ ghép.
* **Kịch bản thuyết trình (Speaker Notes)**: *"Để làm sạch dữ liệu, nhóm tiến hành chuẩn hóa Unicode dạng NFC, tự động khôi phục các mẫu lệch viết hoa viết thường bằng thuật toán tìm kiếm Casing Match, và lọc bỏ hoàn toàn các mẫu dịch máy sai trước khi huấn luyện..."*

---

### Slide 5: Hệ thống Pipeline Retriever-Reader tổng thể
* **Kiến trúc hai giai đoạn (Retriever-Reader)**:
  ```
  [Câu hỏi người dùng] 
         │
         ▼
  [BM25 Retriever] ── Truy hồi Top-3 ngữ cảnh từ 500 đoạn văn bản ──┐
         │                                                        │
         ▼                                                        ▼
  [Transformer Reader] ── Trích xuất đáp án kèm score tin cậy từ 3 đoạn văn ──┘
         │
         ▼
  [Hậu xử lý (Confidence Selection)] ── Chọn đáp án có score cao nhất
  ```
* **Ý nghĩa**: Giúp hệ thống hoạt động trên một kho tài liệu lớn thay vì bắt buộc người dùng cung cấp sẵn đoạn văn ngữ cảnh.
* **Kịch bản thuyết trình (Speaker Notes)**: *"Đây là sơ đồ pipeline tổng thể của hệ thống. Người dùng chỉ cần nhập câu hỏi, BM25 Retriever sẽ tìm Top-3 đoạn văn liên quan nhất từ kho dữ liệu, sau đó Reader sẽ đọc cả 3 đoạn văn này và chọn ra đáp án có điểm tin cậy cao nhất..."*

---

### Slide 6: Mô hình Baseline vs Phương pháp chính
* **Baseline B2 (XLM-RoBERTa Pretrained)**: Dùng checkpoint lớn `deepset/xlm-roberta-base-squad2` chạy trực tiếp trên tiếng Việt (zero-shot transfer).
* **Phương pháp chính M1 (XLM-RoBERTa Fine-tuned)**: Fine-tune mô hình trên dữ liệu tiếng Việt sạch đã xử lý lỗi.
  * *Tham số*: 3 epochs, learning rate = 2e-5, batch size = 32 , max_length 384
* **Kịch bản thuyết trình (Speaker Notes)**: *"Chúng em xây dựng baseline là XLM-RoBERTa gốc chưa fine-tune. Phương pháp chính M1 của nhóm là fine-tune XLM-RoBERTa trên tập dữ liệu tiếng Việt sạch của ViSpanExtractQA để tăng cường độ chính xác..."*

---

### Slide 7: Kết quả thực nghiệm tổng hợp
* **Bảng kết quả trên tập kiểm thử mốc 500 mẫu (Kiểm thử nhanh)**:

| Mô hình | EM (%) | F1 (%) | Cơ chế xử lý |
| :--- | :---: | :---: | :--- |
| **B2: XLM-RoBERTa Pretrained (SQuAD2)** | 44.60 | 70.39 | Trích xuất QA |
| **M1: XLM-RoBERTa Fine-tuned** | **60.60** | **81.05** | **Trích xuất QA** |
| **BM25 + XLM-R Pretrained (Pipeline)** | 37.80 | 61.35 | Tích hợp Retriever-Reader |
| **BM25 + XLM-R Fine-tuned (Pipeline M1)** | **53.80** | **71.95** | **Tích hợp Retriever-Reader (Tốt nhất)** |

* **Kịch bản thuyết trình (Speaker Notes)**: *"Kết quả thực nghiệm trên 500 mẫu cho thấy mô hình M1 độc lập tinh chỉnh đạt EM 60.60% (tăng 16% so với pretrained). Đối với hệ thống hỏi đáp thực tế, pipeline BM25 + M1 Reader đạt EM 53.80%, vượt trội hơn hẳn so với pretrained pipeline chỉ đạt 37.80%. Điều này chứng minh hiệu quả của việc fine-tuning trên dữ liệu sạch và thuật toán Rank Penalty..."*

---

### Slide 7B: Thực nghiệm kiểm chứng trên mốc 5000 mẫu
* **Bảng kết quả trên tập kiểm thử mốc 5000 mẫu (Kiểm chứng lớn)**:

| Mô hình | EM (%) | F1 (%) | Cơ chế xử lý |
| :--- | :---: | :---: | :--- |
| **B2: XLM-RoBERTa Pretrained (SQuAD2)** | 44.32 | 66.52 | Trích xuất QA |
| **M1: XLM-RoBERTa Fine-tuned** | **56.52** | **76.12** | **Trích xuất QA** |
| **BM25 + XLM-R Pretrained (Pipeline)** | 34.88 | 52.37 | Tích hợp Retriever-Reader |
| **BM25 + XLM-R Fine-tuned (Pipeline M1)** | **42.44** | **57.51** | **Tích hợp Retriever-Reader (Tốt nhất)** |

* **Kịch bản thuyết trình (Speaker Notes)**: *"Khi mở rộng kiểm chứng lên 5000 mẫu, xu thế kết quả vẫn nhất quán. Mô hình Fine-tuned M1 đạt EM 56.52% độc lập và Pipeline M1 đạt EM 42.44% (vượt xa mức 34.88% của pretrained pipeline). Dù hiệu năng có giảm nhẹ ở cả hai mốc do không gian tìm kiếm lớn hơn làm giảm độ chính xác của BM25, nhưng hệ thống đề xuất của nhóm vẫn luôn đạt hiệu năng tốt nhất..."*

---

### Slide 8: Phân tích & Nhận xét kết quả
* **Hiệu ứng của việc Fine-tuning**: Mô hình M1 học được cách căn chỉnh biên tốt hơn trên tập tiếng Việt sạch, cải thiện mạnh mẽ cả EM và F1 nhờ quá trình lọc nhiễu dữ liệu.
* **Sức mạnh của Pipeline kết hợp**:
  * Retriever (BM25) đạt độ chính xác **Top-5 Retrieval Accuracy đạt 95.00% ở mốc 500 mẫu và 85.34% ở mốc 5000 mẫu** (Top-3 đạt 93.40% và 82.00%).
  * Bộ lọc Retriever làm giảm chiều dài ngữ cảnh đầu vào của Reader, tránh hiện tượng mô hình Reader bị phân tán sự chú ý khi văn bản quá dài.
* **Kịch bản thuyết trình (Speaker Notes)**: *"Nhận xét chính là việc tích hợp Retriever giúp Reader hoạt động hiệu quả hơn hẳn. Do bộ Retriever lọc trúng văn bản liên quan tới 95.00% (ở mốc 500) và 85.34% (ở mốc 5000), giúp loại bỏ ngữ cảnh nhiễu, làm giảm chiều dài ngữ cảnh đầu vào của Reader, tránh hiện tượng mô hình Reader bị phân tán sự chú ý khi văn bản quá dài."*

---

### Slide 9: Phân tích lỗi chi tiết (Error Analysis)
* **Thống kê các nhóm lỗi của mô hình M1**:
  * **Lỗi biên câu trả lời (Span dư/thiếu)**: **46.6%** (Trích xuất dư chức danh chiếm 45.0% - Ví dụ: Dự đoán `'Đại tướng Ngô Xuân Lịch'` thay vì `'Ngô Xuân Lịch'`, trích xuất thiếu chiếm 1.6%).
  * **Sai lệch vùng chứa câu trả lời (Wrong span)**: **41.7%** (Thường xảy ra khi context chứa nhiều tên riêng hoặc mốc thời gian gây nhiễu).
  * **Nhãn nhiễu / Lỗi dữ liệu gốc (Label noise)**: **11.7%**
* **Hướng khắc phục**:
  * Hậu xử lý cắt tỉa các danh từ xưng hô, chức danh thông dụng bằng regex hoặc danh sách đen.
  * Sử dụng mô hình NER để phát hiện đúng loại thực thể tương ứng với câu hỏi (ví dụ hỏi "Ai" thì bắt buộc trả về thực thể `PER`).
* **Kịch bản thuyết trình (Speaker Notes)**: *"Phân tích lỗi định lượng chỉ ra lỗi biên câu trả lời chiếm tỷ lệ cao nhất với 46.6%. Mô hình thường trích xuất thừa chức danh như Thiếu tướng, Giáo sư. Giải pháp là nhóm đề xuất dùng luật hậu xử lý regex hoặc mô hình NER để cắt tỉa danh xưng..."*

---

### Slide 10: Công nghệ tự học & Khả năng ứng dụng
* **Công nghệ tự học**:
  * Ánh xạ Tokenizer Offset Mapping của Hugging Face để gán nhãn token cho bài toán trích xuất.
  * Tự thiết kế hệ thống web demo Flask hiển thị kết quả trực quan song song.
  * Nghiên cứu thuật toán BM25 Okapi để lập chỉ mục tìm kiếm tài liệu.
* **Khả năng ứng dụng thực tế**:
  * *Tốc độ phản hồi cực nhanh*: Trung bình khoảng ~50ms cho mỗi yêu cầu hỏi đáp trên CPU thông thường.
  * *Chi phí vận hành thấp*: Không yêu cầu phần cứng GPU đắt đỏ để chạy web demo, phù hợp tích hợp vào máy chủ trường học hoặc doanh nghiệp nhỏ.
* **Kịch bản thuyết trình (Speaker Notes)**: *"Qua dự án, chúng em tự học được cách ánh xạ tokenizer offset mapping và phát triển web demo Flask. Hệ thống có độ trễ rất thấp chỉ ~50ms, chi phí vận hành rẻ, hoàn toàn có thể chạy tốt trên CPU bình thường ở thực tế..."*

---

### Slide 11: Demo Giao diện Web UI tương tác
* **Các tính năng nổi bật của Web Demo**:
  * Hỗ trợ 2 chế độ: **Đọc hiểu (Reader-only)** và **Tìm kiếm (Retriever-Reader)**.
  * Cho phép chọn các ví dụ mẫu có sẵn hoặc người dùng tự nhập ngữ cảnh và câu hỏi tùy ý.
  * Hiển thị song song kết quả của các mô hình: XLM-R Pretrained và M1 Fine-tuned.
  * Trực quan hóa điểm tin cậy (Confidence Score) và thời gian xử lý (Latency).
* **Kịch bản thuyết trình (Speaker Notes)**: *"Đây là giao diện Web UI tương tác mà nhóm đã phát triển. Trang web cho phép hiển thị song song kết quả của các mô hình XLM-R Pretrained và Fine-tuned để so sánh trực quan, kèm thời gian phản hồi thực tế của hệ thống..."*

---

### Slide 12: Kết luận & Hướng phát triển
* **Kết luận**:
  * Dự án đã xây dựng hoàn chỉnh và chạy thực nghiệm thành công hệ thống hỏi đáp trích xuất tiếng Việt đạt điểm EM tốt nhất là 60.60% (độc lập) và 53.80% (truy hồi pipeline).
  * Thực hiện tiền xử lý triệt để, khắc phục lỗi căn chỉnh nhãn của tập dữ liệu gốc.
* **Hướng phát triển tương lai**:
  * Thay thế BM25 bằng bộ truy hồi ngữ nghĩa Dense Passage Retrieval (DPR) sử dụng mô hình embedding tiếng Việt (PhoBERT, SBERT).
  * Nghiên cứu tích hợp thêm cơ chế tự động cắt tỉa danh xưng bằng mô hình NER.
* **Kịch bản thuyết trình (Speaker Notes)**: *"Tóm lại, dự án đã đạt được các yêu cầu đề ra của bài tập lớn cuối kỳ. Hướng phát triển tiếp theo là thay thế BM25 bằng tìm kiếm ngữ nghĩa sâu hơn và giải quyết triệt để lỗi biên câu trả lời. Nhóm em xin chân thành cảm ơn thầy cô đã lắng nghe..."*
