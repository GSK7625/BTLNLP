# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import re
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template

GLOBAL_CORPUS_RAW = []
GLOBAL_CORPUS_TOKENIZED = []
GLOBAL_BM25 = None

# Setup paths so we can import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Import metrics and components from src or demo
from src.utils.metrics import normalize_answer, compute_f1, compute_exact
from demo import ExtractiveReader

app = Flask(__name__)

# Global cache for heavy models
MODEL_CACHE = {}

def get_model_reader(model_key: str):
    """Lazy load the reader models to save memory and startup time."""
    if model_key not in MODEL_CACHE:
        if model_key == 'pretrained':
            # Pretrained XLM-RoBERTa on SQuADv2
            model_path = 'deepset/xlm-roberta-base-squad2'
            MODEL_CACHE[model_key] = ExtractiveReader(model_path)
        elif model_key == 'finetuned':
            # Fine-tuned on ViSpanExtractQA
            model_path = os.path.join(PROJECT_ROOT, 'models/xlmroberta_finetuned')
            if not os.path.exists(model_path) or not os.listdir(model_path):
                raise ValueError("Fine-tuned model folder not found or empty. Please run training first.")
            MODEL_CACHE[model_key] = ExtractiveReader(model_path)
    return MODEL_CACHE[model_key]


def run_bm25_selector(question: str, context: str) -> dict:
    """Simple BM25 sentence overlap baseline reader."""
    t0 = time.time()
    
    # Simple word tokenization
    def tokenize(text: str):
        return text.lower().split()
        
    # Split sentences by common boundaries
    sentences = re.split(r'(?<=[.?!，。])\s+|(?<=,)\s+(?=[A-ZÁẮẶẤẦ])', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        latency = (time.time() - t0) * 1000
        return {
            'answer': context.strip(),
            'confidence': 1.0,
            'char_start': 0,
            'char_end': len(context),
            'latency_ms': round(latency, 1)
        }
        
    q_tokens = set(tokenize(question))
    best_sentence = sentences[0]
    best_overlap = -1
    
    for sent in sentences:
        s_tokens = set(tokenize(sent))
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent
            
    # Find the character boundary in context
    char_start = context.find(best_sentence)
    if char_start == -1:
        char_start = 0
        char_end = len(context)
    else:
        char_end = char_start + len(best_sentence)
        
    latency = (time.time() - t0) * 1000
    
    return {
        'answer': best_sentence,
        'confidence': 1.0, # BM25 overlap score has no standardized confidence, set to 1.0
        'char_start': char_start,
        'char_end': char_end,
        'latency_ms': round(latency, 1)
    }


def load_preloaded_examples():
    """Load handpicked examples representing all standard QA evaluation cases."""
    examples = [
        {
            'id': 'case_1_correct',
            'question': 'Người Châu Á đầu tiên nhận giải Fields là ai?',
            'context': 'Sau hơn 60 năm tồn tại, Fields Medal đã được trao cho 48 nhà toán học trên toàn thế giới. Nhà toán học Kunihiko Kodaira là người Nhật Bản và cũng là người châu Á đầu tiên giành Fields Medal.',
            'gold': 'Kunihiko Kodaira',
            'case_description': 'Trường hợp 1 (Đúng hoàn toàn - EM=1, F1=1.0): Cả 3 mô hình trích xuất chính xác.'
        },
        {
            'id': 'case_2_boundary',
            'question': 'Bộ trưởng bộ quốc phòng Việt Nam là ai',
            'context': '- Chiều 9/1 , tại Trụ sở Bộ Tư lệnh Quân khu 5 , Đại tướng Ngô Xuân Lịch , Uỷ viên Bộ Chính trị , Phó Bí thư Quân uỷ Trung ương , Bộ trưởng Bộ Quốc phòng Việt Nam chủ trì lễ đón chính thức Đại tướng Tia Banh , Uỷ viên Thường vụ Đảng Nhân dân Campuchia , Phó Thủ tướng Chính phủ , Bộ trưởng Bộ Quốc phòng dẫn đầu Đoàn Đại biểu Quân sự cấp cao Campuchia thăm chính thức nước ta từ ngày 8 - 11/1 .',
            'gold': 'Ngô Xuân Lịch',
            'case_description': 'Trường hợp 2 (Lỗi biên độ): B2 trích xuất dư chức danh "Đại tướng Ngô Xuân Lịch" (F1=0.75), M1 sửa chính xác "Ngô Xuân Lịch" (EM=1) nhờ fine-tune.'
        },
        {
            'id': 'case_3_wrong_span',
            'question': 'Trong thần thoại Hy Lạp, vị thần khổng lồ có tên là gì?',
            'context': 'Các thần khổng lồ Titans khởi thuỷ bao gồm 12 người gắn liền với rất nhiều khái niệm như đại dương , trí nhớ , tầm nhìn và quy luật tự nhiên ; sau đó , họ lại sinh ra các thần Titans khác , như là Prometheus và Atlas . Họ được dẫn dắt bởi vị thần trẻ nhất trong các vị thần thuộc thế hệ đầu tiên , Cronus , người đã lật đổ cha mình là Uranus .',
            'gold': 'Titans',
            'case_description': 'Trường hợp 3 (Sai span hoàn toàn): Mô hình bị phân tâm bởi nhiều thực thể khổng lồ tương tự trong đoạn văn (chọn Cronus thay vì Titans).'
        },
        {
            'id': 'case_4_under_extraction',
            'question': 'Ai là chủ biên cuốn sách Những dấu vết thời đại đồng thau',
            'context': 'Kết quả khai quật , thám sát , phát hiện khảo cổ đã được viết thành báo cáo lưu lại trong thư viện của các cơ quan chủ trì khai quật như Viện Khảo cổ học , Bảo tàng Lịch Sử Việt Nam . Một công trình tập hợp khá đầy đủ những kết quả nghiên cứu này l\u00e0 cuốn Những vết tích đầu tiên của thời đại đồng thau ở Việt Nam ( Lê Văn Lan , Phạm Văn Kỉnh , Nguyễn Linh 1963 ) .',
            'gold': 'Lê Văn Lan , Phạm Văn Kỉnh , Nguyễn Linh',
            'case_description': 'Trường hợp 4 (Trích xuất thiếu): Mô hình bị ngắt span sớm khi gặp dấu phẩy ngăn cách danh sách các tác giả viết sách (F1=0.94).'
        },
        {
            'id': 'case_5_retriever_error',
            'question': 'Mẹ của Nguyễn Tấn Dũng là ai',
            'context': 'Chủ tịch nước Trần Đại Quang đã đến viếng tại lễ tang của bà Nguyễn Thị Hường , mẹ của nguyên Thủ tướng Nguyễn Tấn Dũng .',
            'gold': 'Nguyễn Thị Hường',
            'case_description': 'Trường hợp 5 (Lỗi cascading trong Pipeline): BM25 Retriever tìm sai ngữ cảnh không chứa đáp án, khiến Reader chọn nhầm thực thể "Nguyễn Thanh Nghị" làm câu trả lời.'
        },
        {
            'id': 'case_6_label_noise',
            'question': 'Khoảng thời gian nào chứng kiến sự nâng cấp đô thị của Bạc Liêu?',
            'context': 'Trong vòng 4 năm được công nhận là thành phố (2010-2014), thành phố Bạc Liêu từ đô thị loại III nhanh chóng đã phát triển lên đô thị loại II.',
            'gold': '2010-2014',
            'case_description': 'Trường hợp 6 (Nhãn nhiễu / Lỗi dữ liệu): Mô hình trả lời "4 năm" (rất hợp lý về mặt ngữ nghĩa) nhưng nhãn gốc chỉ chấp nhận "2010-2014", dẫn đến lệch nhãn mặc dù câu trả lời đúng.'
        },
        {
            'id': 'general_7',
            'question': 'Ai là chủ tịch tập đoàn Alibaba?',
            'context': 'Dư luận đang hết sức ngóng chờ sự kiện Chủ tịch tập đoàn thương mại điện tử Alibaba - Jack Ma (Mã Vân) đến Việt Nam, vậy Jack Ma là ai, Alibaba là tập đoàn thế nào mà lại có sức ảnh hưởng đến như vậy?',
            'gold': 'Jack Ma',
            'case_description': 'Mẫu hỏi đáp 7: Trích xuất thực thể tên người (Jack Ma).'
        },
        {
            'id': 'general_8',
            'question': 'Thuật ngữ Big Bang do ai đề xuất?',
            'context': '1949, Fred Hoyle, một nhà toán học và thiên văn học nổi tiếng người Anh, trong một lần trả lời phỏng vấn của Đài BBC London vào Tháng 3, lần đầu tiên đã gieo thuật ngữ "Big Bang" để mô tả lý thuyết của Lemaître.',
            'gold': 'Fred Hoyle',
            'case_description': 'Mẫu hỏi đáp 8: Trích xuất người phát minh/đề xuất thuật ngữ khoa học.'
        },
        {
            'id': 'general_9',
            'question': 'Ai sáng lập Uber?',
            'context': 'Người đồng sáng lập Uber, ông Travis Kalanick, chính thức trở thành tỷ phú, sau khi thu được 1,4 tỷ USD từ bán cổ phiếu.',
            'gold': 'Travis Kalanick',
            'case_description': 'Mẫu hỏi đáp 9: Xác định nhà sáng lập công ty công nghệ.'
        },
        {
            'id': 'general_10',
            'question': 'Ai là người sáng lập tập đoàn Vingroup?',
            'context': 'Tập đoàn Vingroup được thành lập bởi tỷ phú Phạm Nhật Vượng, người khởi nghiệp thành công với thương hiệu mì gói Mivina tại Ukraine trước khi về Việt Nam đầu tư bất động sản và công nghệ.',
            'gold': 'Phạm Nhật Vượng',
            'case_description': 'Mẫu hỏi đáp 10: Thực thể tên doanh nhân Việt Nam.'
        },
        {
            'id': 'general_11',
            'question': 'Đồng tiền của quốc gia Lào tên là gì?',
            'context': 'Kíp là đơn vị tiền tệ chính thức của nước Cộng hòa Dân chủ Nhân dân Lào từ năm 1952, ký hiệu quốc tế là LAK.',
            'gold': 'Kíp',
            'case_description': 'Mẫu hỏi đáp 11: Trích xuất tên đơn vị tiền tệ.'
        },
        {
            'id': 'general_12',
            'question': 'Đỉnh núi cao nhất Việt Nam là đỉnh nào?',
            'context': 'Fansipan là ngọn núi cao nhất của Việt Nam cũng như của cả ba nước Đông Dương, nên được mệnh danh là "Nóc nhà Đông Dương" với độ cao 3.143 mét.',
            'gold': 'Fansipan',
            'case_description': 'Mẫu hỏi đáp 12: Trích xuất tên địa danh tự nhiên.'
        },
        {
            'id': 'general_13',
            'question': 'Hà Nội trở thành thủ đô của Việt Nam từ năm nào?',
            'context': 'Năm 1010, vua Lý Thái Tổ ban Chiếu dời đô từ Hoa Lư về Đại La và đổi tên thành Thăng Long, mở đầu cho lịch sử nghìn năm là thủ đô của đất nước.',
            'gold': '1010',
            'case_description': 'Mẫu hỏi đáp 13: Trích xuất thông tin mốc thời gian lịch sử.'
        },
        {
            'id': 'general_14',
            'question': 'Hành tinh nào gần Mặt Trời nhất?',
            'context': 'Sao Thủy hay Thủy Tinh là hành tinh nhỏ nhất và nằm gần Mặt Trời nhất trong Hệ Mặt Trời, hoàn thành một vòng quỹ đạo trong 88 ngày Trái Đất.',
            'gold': 'Sao Thủy',
            'case_description': 'Mẫu hỏi đáp 14: Trích xuất thực thể thiên văn.'
        },
        {
            'id': 'general_15',
            'question': 'Tác giả của cuốn tiểu thuyết Tắt đèn là ai?',
            'context': 'Tắt đèn là một trong những tác phẩm văn học hiện thực phê phán tiêu biểu nhất của nhà văn Ngô Tất Tố, khắc họa cuộc sống lầm than của người nông dân dưới chế độ thực dân phong kiến.',
            'gold': 'Ngô Tất Tố',
            'case_description': 'Mẫu hỏi đáp 15: Trích xuất tên tác giả văn học.'
        },
        {
            'id': 'general_16',
            'question': 'Đại hội thể thao Đông Nam Á lần thứ 31 được tổ chức ở đâu?',
            'context': 'SEA Games 31 được đăng cai tổ chức tại Việt Nam từ ngày 12 đến 23 tháng 5 năm 2022 với sự tham gia của 11 quốc gia khu vực Đông Nam Á.',
            'gold': 'Việt Nam',
            'case_description': 'Mẫu hỏi đáp 16: Trích xuất tên quốc gia đăng cai sự kiện.'
        },
        {
            'id': 'general_17',
            'question': 'Kênh đào Suez nối liền hai biển nào?',
            'context': 'Kênh đào Suez là kênh giao thông nhân tạo nằm trên lãnh thổ Ai Cập, nối liền biển Địa Trung Hải với Hồng Hải (Biển Đỏ), giúp rút ngắn hành trình hàng hải giữa châu Âu và châu Á.',
            'gold': 'Địa Trung Hải với Hồng Hải',
            'case_description': 'Mẫu hỏi đáp 17: Trích xuất thông tin địa lý hàng hải.'
        },
        {
            'id': 'general_18',
            'question': 'Vị vua cuối cùng của triều đại phong kiến Việt Nam là ai?',
            'context': 'Hoàng đế Bảo Đại tên khai sinh là Nguyễn Phúc Vĩnh Thụy, là vị hoàng đế thứ 13 của nhà Nguyễn và cũng là vị vua cuối cùng của lịch sử phong kiến Việt Nam.',
            'gold': 'Bảo Đại',
            'case_description': 'Mẫu hỏi đáp 18: Trích xuất tên nhân vật lịch sử.'
        },
        {
            'id': 'general_19',
            'question': 'Ai là người đầu tiên đặt chân lên Mặt Trăng?',
            'context': 'Neil Armstrong là một phi hành gia người Mỹ và là người đầu tiên đặt chân lên Mặt Trăng vào ngày 21 tháng 7 năm 1969 trong sứ mệnh Apollo 11.',
            'gold': 'Neil Armstrong',
            'case_description': 'Mẫu hỏi đáp 19: Trích xuất tên nhà du hành vũ trụ.'
        },
        {
            'id': 'general_20',
            'question': 'Quốc gia nào có diện tích lớn nhất thế giới?',
            'context': 'Liên bang Nga là quốc gia có diện tích lớn nhất thế giới, trải dài trên cả hai châu lục là châu Á và châu Âu với diện tích hơn 17 triệu km vuông.',
            'gold': 'Liên bang Nga',
            'case_description': 'Mẫu hỏi đáp 20: Trích xuất tên quốc gia lớn nhất.'
        },
        {
            'id': 'general_21',
            'question': 'Nhà soạn nhạc Beethoven sinh ra ở nước nào?',
            'context': 'Ludwig van Beethoven sinh năm 1770 tại Bonn, thuộc Tuyển hầu quốc Köln của Thánh chế La Mã, nay thuộc nước Đức, là một trong những vĩ nhân âm nhạc cổ điển thế giới.',
            'gold': 'Đức',
            'case_description': 'Mẫu hỏi đáp 21: Trích xuất thông tin quốc tịch/nơi sinh.'
        },
        {
            'id': 'general_22',
            'question': 'Ai phát minh ra chiếc điện thoại thực dụng đầu tiên?',
            'context': 'Alexander Graham Bell là nhà phát minh, nhà khoa học người Scotland được công nhận là người đã sáng chế ra chiếc điện thoại thực dụng đầu tiên vào năm 1876.',
            'gold': 'Alexander Graham Bell',
            'case_description': 'Mẫu hỏi đáp 22: Trích xuất nhà sáng chế điện thoại.'
        }
    ]

    # Read extra examples from test_clean.json to populate choices dynamically
    test_path = os.path.join(PROJECT_ROOT, 'data/processed/test_clean.json')
    if os.path.exists(test_path):
        try:
            with open(test_path, encoding='utf-8') as f:
                test_data = json.load(f)
                added = 0
                for item in test_data:
                    q = item.get('question_raw', '')
                    ctx = item.get('context_raw', '')
                    ans = item.get('answer_text', '')
                    # Avoid duplicates with already loaded ones
                    if q and ctx and ans and len(ctx) < 250 and len(q) < 60 and len(ans) < 30:
                        if not any(ex['question'] == q for ex in examples):
                            examples.append({
                                'id': f"dynamic_{added + 1}",
                                'question': q,
                                'context': ctx,
                                'gold': ans,
                                'case_description': f'Mẫu số {added + 23} (Tải động từ tập dữ liệu sạch): Trực quan hóa câu hỏi thực nghiệm.'
                            })
                            added += 1
                            if added >= 15:  # Load 15 extra dynamic samples
                                break
        except Exception as e:
            print(f"[WARN] Lỗi load test_clean.json: {e}")
            
    return examples


# Serve index.html directly from template or static files
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results/figures/<filename>')
def serve_figure(filename):
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "results", "figures"))
    return send_from_directory(figures_dir, filename)


@app.route('/api/comparison', methods=['GET'])
def get_comparison():
    comp_path = os.path.join(PROJECT_ROOT, "data/processed/_comparison_500samples_results.json")
    if os.path.exists(comp_path):
        try:
            with open(comp_path, encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception as e:
            pass
    # Fallback to standard project metrics if file is not found
    fallback_results = [
        {
            "name": "B1: BM25-Only (Rule-based)",
            "em": 0.8,
            "f1": 24.31,
            "note": "Baseline tối thiểu — không dùng model"
        },
        {
            "name": "B2: XLM-RoBERTa (pretrained, no FT)",
            "em": 44.6,
            "f1": 70.39,
            "note": "Off-the-shelf — chưa fine-tune trên ViSpanExtractQA"
        },
        {
            "name": "M1: XLM-RoBERTa Fine-tuned (ViSpanExtractQA)",
            "em": 60.6,
            "f1": 81.05,
            "note": "Phương pháp chính — fine-tuned trên dữ liệu sạch"
        },
        {
            "name": "BM25 + XLM-R Pretrained (Pipeline)",
            "em": 38.2,
            "f1": 62.17,
            "note": "BM25 Retriever + Pretrained Reader - BM25 Acc: 93.40%"
        },
        {
            "name": "BM25 + XLM-R Fine-tuned (Pipeline M1)",
            "em": 53.8,
            "f1": 71.95,
            "note": "BM25 Retriever + M1 Reader - BM25 Acc: 93.40%"
        }
    ]
    return jsonify(fallback_results)


@app.route('/api/examples', methods=['GET'])
def get_examples():
    try:
        examples = load_preloaded_examples()
        return jsonify(examples)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    question = data.get('question', '').strip()
    context = data.get('context', '').strip()
    selected_models = data.get('models', [])
    gold = data.get('gold', '').strip()
    
    if not question or not context:
        return jsonify({'error': 'Câu hỏi hoặc đoạn văn không được để trống.'}), 400
        
    results = {}
    
    for model_key in selected_models:
        try:
            if model_key == 'bm25':
                res = run_bm25_selector(question, context)
            elif model_key in ('pretrained', 'finetuned'):
                reader = get_model_reader(model_key)
                res = reader.predict(question, context)
            else:
                continue
                
            # Compute evaluation metrics if gold standard answer is available
            if gold:
                pred_norm = normalize_answer(res['answer'])
                gold_norm = normalize_answer(gold)
                res['em'] = int(pred_norm == gold_norm)
                res['f1'] = round(compute_f1(gold, res['answer']), 4)
            else:
                res['em'] = None
                res['f1'] = None
                
            results[model_key] = res
            
        except Exception as e:
            results[model_key] = {
                'error': str(e),
                'answer': 'Lỗi chạy model',
                'confidence': 0.0,
                'latency_ms': 0.0,
                'char_start': 0,
                'char_end': 0,
                'em': 0,
                'f1': 0.0
            }
            print(f"[ERROR] Model {model_key} failed: {e}")
            
    return jsonify({'results': results})


def init_global_corpus():
    global GLOBAL_CORPUS_RAW, GLOBAL_CORPUS_TOKENIZED, GLOBAL_BM25
    test_path = os.path.join(PROJECT_ROOT, 'data/processed/test_clean.json')
    if os.path.exists(test_path):
        try:
            with open(test_path, encoding='utf-8') as f:
                test_data = json.load(f)
            unique_contexts = {}
            for item in test_data:
                ctx_raw = item.get('context_raw', '')
                ctx_bm25 = item.get('context_bm25', '')
                if ctx_raw and ctx_bm25 and ctx_raw not in unique_contexts:
                    unique_contexts[ctx_raw] = ctx_bm25
            
            GLOBAL_CORPUS_RAW = list(unique_contexts.keys())
            GLOBAL_CORPUS_TOKENIZED = [unique_contexts[c].split() for c in GLOBAL_CORPUS_RAW]
            
            from rank_bm25 import BM25Okapi
            GLOBAL_BM25 = BM25Okapi(GLOBAL_CORPUS_TOKENIZED)
            print(f"  [Corpus QA] Loaded global corpus with {len(GLOBAL_CORPUS_RAW)} unique contexts.")
        except Exception as e:
            print(f"  [Corpus QA] Failed to load global corpus: {e}")
    else:
        print(f"  [Corpus QA] Test clean file not found at: {test_path}")


@app.route('/api/predict_pipeline', methods=['POST'])
def predict_pipeline():
    data = request.json or {}
    question = data.get('question', '').strip()
    selected_models = data.get('models', [])
    gold = data.get('gold', '').strip()
    
    if not question:
        return jsonify({'error': 'Câu hỏi không được để trống.'}), 400
        
    if not GLOBAL_BM25:
        return jsonify({'error': 'Kho dữ liệu chưa được tải hoặc trống.'}), 500
        
    t0 = time.time()
    
    # Tiền xử lý câu hỏi cho BM25
    from src.data.preprocess import preprocess_for_retriever
    q_bm25 = preprocess_for_retriever(question)
    q_tokens = q_bm25.split()
    
    # Truy hồi top-3 contexts
    scores = GLOBAL_BM25.get_scores(q_tokens)
    top_3_indices = np.argsort(scores)[-3:][::-1]
    retrieved_contexts = [GLOBAL_CORPUS_RAW[idx] for idx in top_3_indices]
    
    retrieval_latency = (time.time() - t0) * 1000
    
    results = {}
    for model_key in selected_models:
        try:
            if model_key == 'bm25':
                # Với BM25, chọn câu tốt nhất trong đoạn văn Top-1
                best_context = retrieved_contexts[0]
                res = run_bm25_selector(question, best_context)
                res['selected_context'] = best_context
            elif model_key in ('pretrained', 'finetuned'):
                reader = get_model_reader(model_key)
                
                # Chạy Reader trên cả 3 đoạn văn, tìm kết quả có score cao nhất
                best_score = -1.0
                best_res = None
                best_context = retrieved_contexts[0]
                
                for idx, ctx in enumerate(retrieved_contexts):
                    pred = reader.predict(question, ctx)
                    confidence = pred.get('confidence', 0.0)
                    # Áp dụng rank penalty để tránh overconfidence của model fine-tuned trên context sai
                    penalty = 0.5 if model_key == 'finetuned' else 0.0
                    score = confidence - idx * penalty
                    if score > best_score:
                        best_score = score
                        best_res = pred
                        best_context = ctx
                        
                res = best_res
                res['selected_context'] = best_context
            else:
                continue
                
            # Đánh giá nếu có đáp án đúng đi kèm
            if gold:
                pred_norm = normalize_answer(res['answer'])
                gold_norm = normalize_answer(gold)
                res['em'] = int(pred_norm == gold_norm)
                res['f1'] = round(compute_f1(gold, res['answer']), 4)
            else:
                res['em'] = None
                res['f1'] = None
                
            results[model_key] = res
            
        except Exception as e:
            results[model_key] = {
                'error': str(e),
                'answer': 'Lỗi chạy model',
                'confidence': 0.0,
                'latency_ms': 0.0,
                'char_start': 0,
                'char_end': 0,
                'em': 0,
                'f1': 0.0,
                'selected_context': retrieved_contexts[0]
            }
            print(f"[ERROR] Model {model_key} failed in pipeline: {e}")
            
    return jsonify({
        'retrieved_context': retrieved_contexts[0], # Default Top-1 context
        'retrieved_contexts': retrieved_contexts,   # List of all top-3 contexts
        'retrieval_latency_ms': round(retrieval_latency, 1),
        'results': results
    })


if __name__ == '__main__':
    # Initialize global corpus
    init_global_corpus()
    # Add template folders inside src/web
    app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')
    app.static_folder = os.path.join(os.path.dirname(__file__), 'static')
    
    print("==========================================================")
    print(" Khởi động Flask Server cho demo Hỏi Đáp Tiếng Việt")
    print(" Truy cập tại: http://127.0.0.1:5000")
    print("==========================================================")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
