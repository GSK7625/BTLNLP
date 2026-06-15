# -*- coding: utf-8 -*-
import os
import sys
import io
import unicodedata

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from datasets import load_dataset
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Force UTF-8 encoding for stdout to print Vietnamese text cleanly on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Suppress Hugging Face Hub warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

console = Console()

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import gaussian_kde
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

def plot_token_distributions(ctx_len, q_len, a_len):
    console.print("\n[bold cyan]--- Đang vẽ biểu đồ phân phối độ dài token bằng Seaborn & Matplotlib ---[/bold cyan]")
    
    # Thiết lập font tiếng Việt cho Matplotlib
    for _font in ["Arial", "Tahoma", "DejaVu Sans"]:
        try:
            matplotlib.rcParams["font.family"] = _font
            break
        except Exception:
            continue
            
    # Áp dụng giao diện Seaborn
    sns.set_style("white")
            
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Lấy xlim_max theo phân vị 99.5 của độ dài ngữ cảnh
    xlim_max = int(np.percentile(ctx_len, 99.5))
    x_grid = np.linspace(0, xlim_max, 500)
    
    # Tính toán KDE bằng scipy.stats.gaussian_kde chuẩn
    kde_ctx = gaussian_kde(ctx_len)
    kde_q = gaussian_kde(q_len)
    kde_a = gaussian_kde(a_len)
    
    # Đánh giá và chuẩn hóa đỉnh về 1.0 để hiển thị rõ trên cùng một trục
    y_ctx = kde_ctx(x_grid)
    y_q = kde_q(x_grid)
    y_a = kde_a(x_grid)
    
    norm_ctx = y_ctx / (y_ctx.max() if y_ctx.max() > 0 else 1)
    norm_q = y_q / (y_q.max() if y_q.max() > 0 else 1)
    norm_a = y_a / (y_a.max() if y_a.max() > 0 else 1)
    
    # Vẽ các đường mật độ và tô vùng dưới đường bằng màu được chỉ định
    # Màu xanh dương (Context): #1E88E5
    ax.plot(x_grid, norm_ctx, label="Ngữ cảnh (Context)", color="#1E88E5", linewidth=2.5)
    ax.fill_between(x_grid, norm_ctx, alpha=0.15, color="#1E88E5")
    
    # Màu cam (Question): #FF8F00
    ax.plot(x_grid, norm_q, label="Câu hỏi (Question)", color="#FF8F00", linewidth=2.2)
    ax.fill_between(x_grid, norm_q, alpha=0.12, color="#FF8F00")
    
    # Màu xanh lá (Answer Span): #4CAF50
    ax.plot(x_grid, norm_a, label="Đáp án (Answer / Span)", color="#4CAF50", linewidth=2.2)
    ax.fill_between(x_grid, norm_a, alpha=0.10, color="#4CAF50")
    
    # Định dạng trục hoành và trục tung
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, 1.1)
    
    ax.set_title("Phân phối Độ dài Token của các Trường Dữ liệu (Tập Train)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Độ dài văn bản (Số lượng Token) →", fontsize=10, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Tỉ lệ mẫu (Mật độ chuẩn hóa) ↑", fontsize=10, fontweight="semibold", labelpad=10)
    
    # Hiển thị vạch tỷ lệ phần trăm tương đối trên trục tung
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    
    # Thiết lập lưới và đường bao bằng despine của Seaborn
    ax.grid(True, linestyle="--", linewidth=0.5, color="#E2E8F0", alpha=0.6)
    sns.despine(ax=ax, left=False, bottom=False)
    
    # Tùy chỉnh màu sắc đường viền trục tọa độ
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    
    ax.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=9)
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_path = os.path.join("results", "figures", "token_length_distribution.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    console.print(f"[bold green]✓ Đã vẽ và lưu biểu đồ phân phối độ dài token tại: {out_path}[/bold green]\n")

def run_eda():
    console.print("\n[bold cyan]--- Đang tải dữ liệu để phân tích chi tiết ---[/bold cyan]\n")
    raw_datasets = load_dataset("ntphuc149/ViSpanExtractQA")

    # 1. Thống kê số lượng mẫu và cách chia tập dữ liệu
    total_samples = sum(len(raw_datasets[split]) for split in raw_datasets.keys())

    table_splits = Table(title="[bold green]THỐNG KÊ SỐ LƯỢNG MẪU[/bold green]", box=box.ROUNDED)
    table_splits.add_column("Phân vùng (Split)", style="cyan", no_wrap=True)
    table_splits.add_column("Số lượng mẫu (Size)", style="magenta", justify="right")
    table_splits.add_column("Tỷ lệ (%)", style="yellow", justify="right")

    for split in raw_datasets.keys():
        split_len = len(raw_datasets[split])
        pct = (split_len / total_samples) * 100
        table_splits.add_row(str(split), f"{split_len:,}", f"{pct:.2f}%")
    table_splits.add_row("[bold]Tổng cộng[/bold]", f"[bold]{total_samples:,}[/bold]", "100.00%")

    console.print(table_splits)
    console.print()

    # Khởi tạo các danh sách gom ví dụ lỗi để trực quan hóa
    examples_case = []
    examples_whitespace = []
    examples_nfc = []
    examples_mismatch = []
    train_records = []

    # Duyệt qua từng tập dữ liệu (train, validation, test) để thống kê và phân tích
    for split_name in raw_datasets.keys():
        console.print(Panel(f"[bold cyan]PHÂN TÍCH TẬP DỮ LIỆU: {str(split_name).upper()}[/bold cyan]", border_style="cyan"))
        
        df_split = raw_datasets[split_name].to_pandas()
        assert isinstance(df_split, pd.DataFrame)
        
        # 2. Thống kê độ dài từ (tính sơ bộ bằng khoảng trắng)
        df_split['ctx_len'] = df_split['context'].apply(lambda x: len(str(x).split()))
        df_split['q_len'] = df_split['question'].apply(lambda x: len(str(x).split()))
        df_split['a_len'] = df_split['answer_text'].apply(lambda x: len(str(x).split()))
        
        table_lengths = Table(title=f"[bold green]THỐNG KÊ ĐỘ DÀI TỪ TRÊN TẬP {str(split_name).upper()}[/bold green]", box=box.ROUNDED)
        table_lengths.add_column("Trường dữ liệu", style="cyan")
        table_lengths.add_column("Min", style="magenta", justify="right")
        table_lengths.add_column("Max", style="magenta", justify="right")
        table_lengths.add_column("Trung bình", style="yellow", justify="right")
        
        for field, col in [("Context (Ngữ cảnh)", "ctx_len"), ("Question (Câu hỏi)", "q_len"), ("Answer (Đáp án)", "a_len")]:
            table_lengths.add_row(
                field,
                f"{df_split[col].min()}",
                f"{df_split[col].max()}",
                f"{df_split[col].mean():.1f}"
            )
        
        console.print(table_lengths)
        console.print()
        
        # 3. Quét tìm các vấn đề của dữ liệu (Data Issues) và Phân tích lỗi không khớp ngữ cảnh
        null_ctx = df_split['context'].isnull().sum()
        null_q = df_split['question'].isnull().sum()
        null_a = df_split['answer_text'].isnull().sum()
        dup_questions = df_split.duplicated(subset=['question']).sum()
        
        # Phân tích sâu các lý do câu trả lời không xuất hiện trong ngữ cảnh
        out_of_context = 0
        case_mismatch = 0
        nfc_mismatch = 0
        whitespace_mismatch = 0
        total_mismatch = 0  # Do lỗi dịch máy, paraphrase, hoặc thiếu hẳn từ ngữ
        
        split_records = df_split[['context', 'question', 'answer_text']].to_dict('records')
        
        # Nếu là tập train, ta gán vào train_records để phần hiển thị mẫu điển hình phía sau hoạt động
        if split_name == 'train':
            train_records = split_records
            train_ctx_len = df_split['ctx_len'].values
            train_q_len = df_split['q_len'].values
            train_a_len = df_split['a_len'].values
            
        for idx, row in enumerate(split_records):
            ctx = str(row['context'])
            ans = str(row['answer_text'])
            q = str(row['question'])
            
            if ans not in ctx:
                out_of_context += 1
                
                # 1. Thử khớp không phân biệt chữ hoa/thường
                if ans.lower() in ctx.lower():
                    case_mismatch += 1
                    if len(examples_case) < 2:
                        examples_case.append({"idx": idx, "ctx": ctx, "ans": ans, "q": q, "type": f"Lệch chữ hoa/thường ({split_name})"})
                    continue
                    
                # 2. Thử khớp khi loại bỏ khoảng trắng đầu/cuối của đáp án
                if ans.strip() in ctx:
                    whitespace_mismatch += 1
                    if len(examples_whitespace) < 2:
                        examples_whitespace.append({"idx": idx, "ctx": ctx, "ans": ans, "q": q, "type": f"Thừa khoảng trắng ({split_name})"})
                    continue
                    
                # 3. Thử khớp khi chuẩn hóa Unicode NFC
                norm_ans = unicodedata.normalize('NFC', ans)
                norm_ctx = unicodedata.normalize('NFC', ctx)
                if norm_ans in norm_ctx:
                    nfc_mismatch += 1
                    if len(examples_nfc) < 2:
                        examples_nfc.append({"idx": idx, "ctx": ctx, "ans": ans, "q": q, "type": f"Lệch Unicode ({split_name})"})
                    continue
                    
                # 4. Thử khớp kết hợp cả ba
                norm_ans_clean = norm_ans.strip().lower()
                norm_ctx_clean = norm_ctx.strip().lower()
                if norm_ans_clean in norm_ctx_clean:
                    case_mismatch += 1
                    if len(examples_case) < 2:
                        examples_case.append({"idx": idx, "ctx": ctx, "ans": ans, "q": q, "type": f"Lệch chữ hoa/thường (kết hợp, {split_name})"})
                    continue
                    
                # Nếu vẫn không khớp -> Lỗi dịch máy hoặc câu trả lời không khớp ngữ cảnh thực tế
                total_mismatch += 1
                if len(examples_mismatch) < 2:
                    examples_mismatch.append({"idx": idx, "ctx": ctx, "ans": ans, "q": q, "type": f"Lỗi dịch máy / Không tồn tại ({split_name})"})
        
        table_issues = Table(title=f"[bold red]QUÉT VÀ PHÂN TÍCH VẤN ĐỀ CỦA TẬP {str(split_name).upper()}[/bold red]", box=box.ROUNDED)
        table_issues.add_column("Vấn đề / Loại lỗi", style="cyan")
        table_issues.add_column("Số lượng", style="magenta", justify="right")
        table_issues.add_column("Tỷ lệ % trên tập", style="yellow", justify="right")
        table_issues.add_column("Đánh giá & Hướng xử lý", style="green")
        
        total_len = len(df_split)
        table_issues.add_row(
            "Mẫu bị rỗng (Null) - Context",
            f"{null_ctx}",
            f"{null_ctx / total_len * 100:.3f}%" if total_len > 0 else "0%",
            "Bình thường" if null_ctx == 0 else "Cần loại bỏ mẫu"
        )
        table_issues.add_row(
            "Mẫu bị rỗng (Null) - Question",
            f"{null_q}",
            f"{null_q / total_len * 100:.3f}%" if total_len > 0 else "0%",
            "Bình thường" if null_q == 0 else "Cần loại bỏ mẫu"
        )
        table_issues.add_row(
            "Mẫu bị rỗng (Null) - Answer",
            f"{null_a}",
            f"{null_a / total_len * 100:.3f}%" if total_len > 0 else "0%",
            "Bình thường" if null_a == 0 else "Cần điền hoặc loại bỏ mẫu"
        )
        table_issues.add_row(
            "Câu hỏi bị trùng lặp",
            f"{dup_questions}",
            f"{dup_questions / total_len * 100:.2f}%" if total_len > 0 else "0%",
            "Một câu hỏi áp cho nhiều ngữ cảnh (Không nhất thiết là lỗi)"
        )
        table_issues.add_row(
            "Đáp án không khớp ngữ cảnh (Tổng cộng)",
            f"{out_of_context}",
            f"{out_of_context / total_len * 100:.2f}%" if total_len > 0 else "0%",
            "[bold red]Cực kỳ nghiêm trọng[/bold red] (Chi tiết phân tích bên dưới)"
        )
        
        console.print(table_issues)
        console.print()
        
        # In bảng phân tích chi tiết lỗi không khớp ngữ cảnh
        table_mismatch_detail = Table(title=f"[bold yellow]CHI TIẾT LỖI ĐÁP ÁN KHÔNG KHỚP TRÊN TẬP {str(split_name).upper()}[/bold yellow]", box=box.ROUNDED)
        table_mismatch_detail.add_column("Nguyên nhân lỗi", style="cyan")
        table_mismatch_detail.add_column("Số lượng mẫu", style="magenta", justify="right")
        table_mismatch_detail.add_column("Tỷ lệ % trong nhóm lỗi", style="yellow", justify="right")
        table_mismatch_detail.add_column("Ví dụ minh họa & Giải pháp", style="green")
        
        if out_of_context > 0:
            table_mismatch_detail.add_row(
                "Lệch chữ hoa / chữ thường",
                f"{case_mismatch:,}",
                f"{case_mismatch / out_of_context * 100:.1f}%",
                "Ví dụ: 'Bảo Phúc' vs 'bảo phúc'. Giải pháp: lowercase trước khi khớp."
            )
            table_mismatch_detail.add_row(
                "Lỗi khoảng trắng thừa (Whitespace)",
                f"{whitespace_mismatch:,}",
                f"{whitespace_mismatch / out_of_context * 100:.1f}%",
                "Ví dụ: ' 9 p. m ' vs '9 p. m'. Giải pháp: strip() đáp án."
            )
            table_mismatch_detail.add_row(
                "Lệch chuẩn hóa Unicode (NFC vs NFD)",
                f"{nfc_mismatch:,}",
                f"{nfc_mismatch / out_of_context * 100:.1f}%",
                "Ví dụ: Tổ hợp vs Dựng sẵn. Giải pháp: unicodedata.normalize('NFC')."
            )
            table_mismatch_detail.add_row(
                "Lỗi dịch máy / Paraphrase thực tế",
                f"{total_mismatch:,}",
                f"{total_mismatch / out_of_context * 100:.1f}%",
                "Ví dụ: 'lactobacilli' (đáp án) bị dịch thành 'cái' (ngữ cảnh). Cần loại bỏ mẫu lỗi này."
            )
            console.print(table_mismatch_detail)
            console.print()


    # --- PHẦN MẪU DỮ LIỆU ĐIỂN HÌNH (HỢP LỆ) ---
    console.print("[bold green][VÍ DỤ MẪU DỮ LIỆU ĐIỂN HÌNH (HỢP LỆ)][/bold green]\n")

    valid_examples = []
    for idx, row in enumerate(train_records):
        ctx = str(row['context'])
        ans = str(row['answer_text'])
        q = str(row['question'])
        if ans in ctx:
            valid_examples.append({"idx": idx, "ctx": ctx, "ans": ans, "q": q})
            if len(valid_examples) == 2:
                break

    for i, ex in enumerate(valid_examples):
        ctx = ex["ctx"]
        q = ex["q"]
        ans = ex["ans"]
        
        truncated_ctx = ctx
        is_truncated = False
        ans_idx = ctx.find(ans)
        if ans_idx != -1:
            start_snippet = max(0, ans_idx - 120)
            end_snippet = min(len(ctx), ans_idx + len(ans) + 120)
            truncated_ctx = ctx[start_snippet:end_snippet]
            if start_snippet > 0:
                truncated_ctx = "... " + truncated_ctx
            if end_snippet < len(ctx):
                truncated_ctx = truncated_ctx + " ..."
            is_truncated = True
        elif len(ctx) > 300:
            truncated_ctx = ctx[:300] + "..."
            is_truncated = True
            
        panel_content = Text()
        panel_content.append("Ngữ cảnh (Context): ", style="bold cyan")
        
        if ans in truncated_ctx:
            parts = truncated_ctx.split(ans, 1)
            panel_content.append(parts[0], style="white")
            panel_content.append(ans, style="bold green underline")
            panel_content.append(parts[1], style="white")
        else:
            panel_content.append(truncated_ctx, style="white")
            
        panel_content.append("\n\nCâu hỏi (Question): ", style="bold cyan")
        panel_content.append(q, style="yellow")
        panel_content.append("\n\nĐáp án thực tế (Answer): ", style="bold cyan")
        panel_content.append(f"'{ans}'", style="bold green")
        
        console.print(Panel(panel_content, title=f"[bold green]Mẫu Điển Hình #{i+1} (ID: {ex['idx']})[/bold green]", border_style="green", box=box.ROUNDED))
    console.print()


    # --- PHẦN TRỰC QUAN HÓA CÁC MẪU LỖI ---
    console.print("[bold red][MINH HỌA TRỰC QUAN CÁC MẪU BỊ LỖI PHÂN LỚP][/bold red]\n")

    def show_visual_example(example, color, highlight_fn):
        idx = example["idx"]
        ctx = example["ctx"]
        ans = example["ans"]
        q = example["q"]
        err_type = example["type"]
        
        panel_text = Text()
        panel_text.append(f"Mẫu số trong tập dữ liệu: {idx}\n", style="bold cyan")
        panel_text.append(f"Câu hỏi: ", style="bold yellow")
        panel_text.append(f"{q}\n", style="yellow")
        panel_text.append(f"Đáp án cần tìm: ", style="bold green")
        panel_text.append(f"'{ans}'\n\n", style="green")
        
        panel_text.append("Trực quan ngữ cảnh (chỗ bị lệch được làm nổi bật):\n", style="bold white")
        panel_text.append(highlight_fn(ctx, ans))
        
        console.print(Panel(
            panel_text,
            title=f"[bold {color}]LỖI: {err_type.upper()}[/bold {color}]",
            border_style=color,
            box=box.ROUNDED
        ))
        console.print()

    # 1. Hàm highlight cho Case Mismatch
    def highlight_case(ctx, ans):
        start_idx = ctx.lower().find(ans.lower())
        if start_idx != -1:
            end_idx = start_idx + len(ans)
            start_snippet = max(0, start_idx - 120)
            end_snippet = min(len(ctx), end_idx + 120)
            
            prefix = ctx[start_snippet:start_idx]
            match = ctx[start_idx:end_idx]
            suffix = ctx[end_idx:end_snippet]
            
            res = Text()
            if start_snippet > 0:
                res.append("... ")
            res.append(prefix, style="dim white")
            res.append(match, style="bold black on yellow")
            res.append(suffix, style="dim white")
            if end_snippet < len(ctx):
                res.append(" ...")
            return res
        return Text(ctx[:200] + "...")

    # 2. Hàm highlight cho Whitespace Mismatch
    def highlight_whitespace(ctx, ans):
        stripped_ans = ans.strip()
        start_idx = ctx.find(stripped_ans)
        if start_idx != -1:
            end_idx = start_idx + len(stripped_ans)
            start_snippet = max(0, start_idx - 120)
            end_snippet = min(len(ctx), end_idx + 120)
            
            prefix = ctx[start_snippet:start_idx]
            match = ctx[start_idx:end_idx]
            suffix = ctx[end_idx:end_snippet]
            
            # Thay thế khoảng trắng bằng ký hiệu dễ thấy
            display_ans = ans.replace(" ", "•")
            display_match = match.replace(" ", "•")
            
            res = Text()
            res.append(f"-> Chuỗi đáp án thực tế có khoảng trắng: ", style="bold cyan")
            res.append(f"'{display_ans}'\n", style="magenta")
            res.append(f"-> Chuỗi trong ngữ cảnh tìm thấy: ", style="bold cyan")
            res.append(f"'{display_match}'\n\n", style="magenta")
            
            if start_snippet > 0:
                res.append("... ")
            res.append(prefix, style="dim white")
            res.append(match, style="bold white on magenta")
            res.append(suffix, style="dim white")
            if end_snippet < len(ctx):
                res.append(" ...")
            return res
        return Text(ctx[:200] + "...")

    # 3. Hàm highlight cho Unicode Mismatch
    def highlight_unicode(ctx, ans):
        # Tìm theo chuẩn hóa NFC
        norm_ans = unicodedata.normalize('NFC', ans)
        norm_ctx = unicodedata.normalize('NFC', ctx)
        start_idx = norm_ctx.find(norm_ans)
        
        if start_idx != -1:
            end_idx = start_idx + len(norm_ans)
            start_snippet = max(0, start_idx - 120)
            end_snippet = min(len(norm_ctx), end_idx + 120)
            
            prefix = norm_ctx[start_snippet:start_idx]
            match = norm_ctx[start_idx:end_idx]
            suffix = norm_ctx[end_idx:end_snippet]
            
            # Lấy unicode code point của đáp án gốc để đối chiếu
            ans_codes = " ".join([f"U+{ord(c):04X}" for c in ans])
            
            res = Text()
            res.append(f"-> Mã Code Points của đáp án: {ans_codes}\n\n", style="cyan")
            
            if start_snippet > 0:
                res.append("... ")
            res.append(prefix, style="dim white")
            res.append(match, style="bold black on cyan")
            res.append(suffix, style="dim white")
            if end_snippet < len(norm_ctx):
                res.append(" ...")
            return res
        return Text(ctx[:200] + "...")

    # 4. Hàm highlight cho Translation / Mismatch
    def highlight_translation(ctx, ans):
        # Tìm kiếm các từ của đáp án có xuất hiện rải rác trong ngữ cảnh không
        ans_words = set(ans.lower().split())
        
        # Cắt một đoạn ngữ cảnh ngẫu nhiên hoặc hiển thị đoạn đầu
        # Để trực quan, ta tìm câu có nhiều từ trùng nhất với đáp án
        sentences = ctx.split('.')
        best_sentence = ""
        max_overlap = 0
        for s in sentences:
            overlap = len(ans_words.intersection(set(s.lower().split())))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s
                
        res = Text()
        res.append("-> Không tìm thấy đáp án khớp cụm từ trong ngữ cảnh (lỗi dịch máy hoặc paraphrase)\n", style="bold red")
        
        if best_sentence and max_overlap > 0:
            res.append(f"-> Câu có độ tương đồng cao nhất trong ngữ cảnh (chứa từ khóa trùng): \n", style="bold yellow")
            words = best_sentence.split()
            for w in words:
                w_clean = w.strip(".,!?()\"'-").lower()
                if w_clean in ans_words:
                    res.append(w + " ", style="bold red underline")
                else:
                    res.append(w + " ", style="white")
        else:
            res.append("-> Ngữ cảnh (300 ký tự đầu):\n", style="bold yellow")
            res.append(ctx[:300] + "...", style="dim white")
            
        return res

    # Hiển thị lỗi Chữ hoa/Chữ thường
    if examples_case:
        for ex in examples_case:
            show_visual_example(ex, "yellow", highlight_case)

    # Hiển thị lỗi Khoảng trắng
    if examples_whitespace:
        for ex in examples_whitespace:
            show_visual_example(ex, "magenta", highlight_whitespace)

    # Hiển thị lỗi Unicode
    if examples_nfc:
        for ex in examples_nfc:
            show_visual_example(ex, "cyan", highlight_unicode)

    # Hiển thị lỗi Không khớp hoàn toàn (Dịch máy/Paraphrase)
    if examples_mismatch:
        for ex in examples_mismatch:
            show_visual_example(ex, "red", highlight_translation)

    # 4. Trực quan hóa phân phối độ dài token
    if HAS_PLOT and train_ctx_len is not None:
        plot_token_distributions(train_ctx_len, train_q_len, train_a_len)

if __name__ == "__main__":
    run_eda()
