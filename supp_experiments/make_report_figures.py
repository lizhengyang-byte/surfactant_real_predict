"""
make_report_figures.py — 生成报告配图（静态 PNG，浅色主题，色板已验证 CVD 安全）
=================================================================================
  figures/exp1_r2_heatmap.png   6模型×6目标 R² 热力图（顺序蓝色，白≈近零/失败）
  figures/exp4_perm_delta.png   ΔR²（排列-基线）分组柱状图（误差棒=5次排列 spread）

用法：python supp_experiments/make_report_figures.py
（需要先运行 experiment1.py 与 experiment4.py）
"""
import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# —— 中文字体（Windows）：matplotlib 默认 DejaVu Sans 无中日韩字形，
#    标题/轴标签会显示成豆腐块。优先微软雅黑，回退 SimHei。（2026-10-11 修复）
import matplotlib.font_manager as fm
_CN_CANDIDATES = ["Microsoft YaHei", "SimHei", "NSimSun", "SimSun"]
_CN_FONT = None
for _f in _CN_CANDIDATES:
    if any(_f in getattr(p, "name", "") for p in fm.fontManager.ttflist):
        _CN_FONT = _f
        break
if _CN_FONT is None:
    _CN_FONT = "DejaVu Sans"  # 兜底（不理想但不会崩）
plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免 CJK 字形缺失告警

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, TARGET_NAMES, TARGET_CSV_COLS

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# —— 已验证色板（dataviz skill reference palette）——
INK = "#0b0b0b"          # primary ink
INK2 = "#52514e"         # secondary ink
GRID = "#e1e0d9"         # gridline
SERIES_RNN = "#2a78d6"   # categorical slot 1 (blue)
SERIES_TRANS = "#eb6834" # categorical slot 2 (orange)
# 顺序蓝（R² 热力图）(light -> dark)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

ORDER = ["MLP-original", "MLP-small", "RNN-original", "RNN-large",
         "Transformer-original", "Transformer-large"]
DISPLAY = {"MLP-original": "MLP-original", "MLP-small": "MLP-small",
           "RNN-original": "RNN-original", "RNN-large": "RNN-large",
           "Transformer-original": "Transformer-original",
           "Transformer-large": "Transformer-large"}
TARGET_DISPLAY = dict(zip(TARGET_NAMES, TARGET_CSV_COLS))


def fmt_r2(v):
    return f"{v:+.2f}" if v < -0.005 else f"{v:.2f}"


def fig1(df_pivot, out):
    """R² 热力图：行=模型，列=目标。"""
    mat = df_pivot[list(TARGET_NAMES)].values.astype(float)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    vmin, vmax = 0.0, 1.0   # 0 映射到最浅（≈白，接近失败），1 最深
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=200)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9.5, color=INK)
                continue
            txt_color = INK if v < 0.30 else "white"
            ax.text(j, i, fmt_r2(v), ha="center", va="center", fontsize=9.5,
                    color=txt_color, fontweight="bold")
    ax.set_xticks(range(len(TARGET_NAMES)))
    ax.set_xticklabels([TARGET_DISPLAY[t] for t in TARGET_NAMES], fontsize=10)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([DISPLAY[m] for m in ORDER], fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("目标属性", fontsize=11, color=INK2)
    ax.set_title("实验一 · 容量匹配消融：各模型在各目标的测试 R²",
                 fontsize=11.5, color=INK, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("R²", fontsize=10, color=INK2)
    cbar.outline.set_linewidth(0)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def fig2(exp4_summary, exp4_perm, out):
    """ΔR² = mean(perm) - baseline 柱状图，误差棒为 5 次排列的 min~max。"""
    mods = ["RNN-original", "Transformer-original"]
    colors = {"RNN-original": SERIES_RNN, "Transformer-original": SERIES_TRANS}
    ts = TARGET_NAMES
    x = np.arange(len(ts))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=200)
    for k, mod in enumerate(mods):
        means, lows, highs = [], [], []
        for j, t in enumerate(ts):
            b = exp4_summary[(exp4_summary["Model"] == mod) &
                             (exp4_summary["Target"] == t)]["baseline_R2"].iloc[0]
            perm_r2 = np.array([exp4_perm[f"perm_{p+1}"][mod][t]["R2"] for p in range(5)])
            means.append(perm_r2.mean() - b)
            lows.append(perm_r2.min() - b)
            highs.append(perm_r2.max() - b)
        ax.bar(x + width * (k - 0.5), means, width, yerr=[np.array(means) - np.array(lows),
                                                          np.array(highs) - np.array(means)],
               color=colors[mod], label=mod, capsize=3, linewidth=0, alpha=0.9,
               error_kw={"ecolor": INK, "elinewidth": 1.0})
        for xi, m in zip(x + width * (k - 0.5), means):
            ax.text(xi, m + (0.004 if m >= 0 else -0.007), f"{m:+.3f}",
                    ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=GRID, linewidth=1.2)
    ax.axhspan(-0.03, 0.03, color=GRID, alpha=0.35, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_DISPLAY[t] for t in ts], fontsize=10)
    ax.set_xlim(-0.5, len(ts) - 0.5)
    ax.set_ylabel("ΔR²（排列均值 − 基线）", fontsize=10.5, color=INK2)
    ax.set_title("实验四 · 特征排列随机化：5 次随机列置换相对基线的 R² 变化（灰带 = ±0.03 判据）",
                 fontsize=11.5, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def main():
    pivot_path = os.path.join(RESULTS_DIR, "exp1_capacity_pivot.csv")
    if not os.path.isfile(pivot_path):
        raise SystemExit("缺少 exp1_capacity_pivot.csv，请先运行 experiment1.py")
    dfp = pd.read_csv(pivot_path, index_col=0)
    dfp = dfp.reindex([m for m in ORDER if m in dfp.index])
    fig1(dfp, os.path.join(FIG_DIR, "exp1_r2_heatmap.png"))

    sum_path = os.path.join(RESULTS_DIR, "exp4_summary.csv")
    perm_path = os.path.join(RESULTS_DIR, "exp4_permutation.json")
    if not (os.path.isfile(sum_path) and os.path.isfile(perm_path)):
        print("缺少实验四结果，跳过 fig2")
        return
    s = pd.read_csv(sum_path)
    perm = json.load(open(perm_path, encoding="utf-8"))
    fig2(s, perm, os.path.join(FIG_DIR, "exp4_perm_delta.png"))


if __name__ == "__main__":
    main()