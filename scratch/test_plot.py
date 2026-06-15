import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from datasets import load_dataset
from pathlib import Path

def main():
    print("Loading dataset...")
    raw_datasets = load_dataset("ntphuc149/ViSpanExtractQA")
    df_train = raw_datasets["train"].to_pandas()
    
    # Compute lengths
    ctx_len = df_train['context'].apply(lambda x: len(str(x).split())).values
    q_len = df_train['question'].apply(lambda x: len(str(x).split())).values
    a_len = df_train['answer_text'].apply(lambda x: len(str(x).split())).values
    
    # Setup styling using Seaborn
    sns.set_style("white")
    
    # Setup font
    for _font in ["Arial", "Tahoma", "DejaVu Sans"]:
        try:
            matplotlib.rcParams["font.family"] = _font
            break
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # xlim_max for context
    xlim_max = int(np.percentile(ctx_len, 99.5))
    x_grid = np.linspace(0, xlim_max, 500)
    
    # Compute KDEs using scipy's standard gaussian_kde
    kde_ctx = gaussian_kde(ctx_len)
    kde_q = gaussian_kde(q_len)
    kde_a = gaussian_kde(a_len)
    
    # Evaluate and normalize peaks to 1.0
    y_ctx = kde_ctx(x_grid)
    y_q = kde_q(x_grid)
    y_a = kde_a(x_grid)
    
    norm_ctx = y_ctx / y_ctx.max()
    norm_q = y_q / y_q.max()
    norm_a = y_a / y_a.max()
    
    # Plot curves with colors requested:
    # Màu xanh dương (Context): #1E88E5
    # Màu cam (Question): #FF8F00
    # Màu xanh lá (Answer Span): #4CAF50
    ax.plot(x_grid, norm_ctx, label="Ngữ cảnh (Context)", color="#1E88E5", linewidth=2.5)
    ax.fill_between(x_grid, norm_ctx, alpha=0.15, color="#1E88E5")
    
    ax.plot(x_grid, norm_q, label="Câu hỏi (Question)", color="#FF8F00", linewidth=2.2)
    ax.fill_between(x_grid, norm_q, alpha=0.12, color="#FF8F00")
    
    ax.plot(x_grid, norm_a, label="Đáp án (Answer / Span)", color="#4CAF50", linewidth=2.2)
    ax.fill_between(x_grid, norm_a, alpha=0.10, color="#4CAF50")
    
    # Customize axes
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, 1.1)
    
    # Add title and labels
    ax.set_title("Phân phối Độ dài Token của các Trường Dữ liệu (Tập Train)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Độ dài văn bản (Số lượng Token) →", fontsize=10, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Số mẫu (Mật độ chuẩn hóa) ↑", fontsize=10, fontweight="semibold", labelpad=10)
    
    # Hide the default y ticks because it is normalized
    ax.set_yticks([])
    
    # Stylize grid and spines
    ax.grid(True, linestyle="--", linewidth=0.5, color="#E2E8F0", alpha=0.6)
    sns.despine(left=False, bottom=False)  # standard Seaborn spine removal
    
    # Customize spine colors
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    
    ax.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=9)
    
    plt.tight_layout()
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "token_length_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Plot saved successfully to {out_path}")

if __name__ == "__main__":
    main()
