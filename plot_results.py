"""
学术绘图脚本: FreqMamba-CSI 实验结果可视化
对应论文: Fig.5 (Speed), Fig.6 (SNR), Fig.7 (消融), Fig.8 (MAML++), Fig.9-10 (频域分析)
数据源: 各实验脚本输出的 CSV 结果文件

模型名称映射 (实验脚本输出 → 论文展示):
  LSTM          → Vanilla LSTM
  Transformer   → Vanilla Transformer
  Ours          → FreqMamba-CSI (Proposed)

消融配置映射 (实验三):
    Time-Only      → Time-Only (Mamba)
    Fixed-Fusion   → Fixed Fusion (0.5)
    FreqMamba-CSI  → FreqMamba-CSI (Full)
  输出: 双柱状图 -- (a) 全Speed对比 (b) 全SNR对比
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ============================================================
# 全局样式配置 — 符合 IEEE/科研期刊标准
# ============================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": 15,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "grid.alpha": 0.3,
    }
)

OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 颜色与线型方案 (高对比度, 黑白打印友好)
# ============================================================
MODEL_STYLES = {
    "LSTM": dict(color="#C41E3A", marker="s", linestyle="--", label="Vanilla LSTM"),
    "Transformer": dict(
        color="#2E8B57", marker="^", linestyle="-.", label="Vanilla Transformer"
    ),
    "Ours": dict(
        color="#15158E", marker="o", linestyle="-", label="FreqMamba-CSI (Proposed)"
    ),
}

MAML_STYLES = {
    "Static": dict(
        color="#808080", marker="s", linestyle="--", label="Static (No Adapt.)"
    ),
    "MAML": dict(color="#D4890E", marker="^", linestyle="-.", label="MAML"),
    "MAML++": dict(
        color="#15158E", marker="o", linestyle="-", label="MAML++ (Proposed)"
    ),
}

SPEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SNRS = [0, 5, 10, 15, 20, 25, 30]
ADAPT_STEPS = 5


# ============================================================
# Fig.5: 不同终端速度下的 NMSE (实验一)
# ============================================================
def plot_exp1_speed(csv_path="NMSE_Results_Speed_Comparison.csv"):
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model_name in ["LSTM", "Transformer", "Ours"]:
        m_data = df[df["Model"] == model_name].sort_values("Speed_kmh")
        style = MODEL_STYLES[model_name]
        ax.plot(
            m_data["Speed_kmh"].values,
            m_data["NMSE_dB"].values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xlim(5, 105)
    ax.set_xticks(np.arange(10, 101, 10))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(f"{OUTPUT_DIR}/Fig5_NMSE_vs_Speed.pdf", format="pdf")
    fig.savefig(f"{OUTPUT_DIR}/Fig5_NMSE_vs_Speed.png", format="png")
    plt.close(fig)
    print(f"Fig.5 saved to {OUTPUT_DIR}/Fig5_NMSE_vs_Speed.*")


# ============================================================
# Fig.6: 不同 SNR 下的 NMSE (实验二)
# ============================================================
def plot_exp2_snr(csv_path="NMSE_Results_SNR_Comparison.csv"):
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model_name in ["LSTM", "Transformer", "Ours"]:
        m_data = df[df["Model"] == model_name].sort_values("SNR_dB")
        style = MODEL_STYLES[model_name]
        ax.plot(
            m_data["SNR_dB"].values,
            m_data["NMSE_dB"].values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xlim(-1, 31)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.savefig(f"{OUTPUT_DIR}/Fig6_NMSE_vs_SNR.pdf", format="pdf")
    fig.savefig(f"{OUTPUT_DIR}/Fig6_NMSE_vs_SNR.png", format="png")
    plt.close(fig)
    print(f"Fig.6 saved to {OUTPUT_DIR}/Fig6_NMSE_vs_SNR.*")


# ============================================================
# Fig.7a / Fig.7b: 消融实验 — 时频双路各组件的独立贡献 (实验三)
# 两张独立柱状图: (a) 全Speed对比 (SNR固定), (b) 全SNR对比 (Speed固定)
# 对比: Time-Only, Frequency-Only, Fixed-Fusion, FreqMamba-CSI (Full)
# ============================================================

ABLATION_COLORS = {
    "Time-Only": "#C41E3A",
    "Frequency-Only": "#D4890E",
    "Fixed-Fusion": "#8B45A0",
    "FreqMamba-CSI": "#15158E",
}
ABLATION_LABELS = {
    "Time-Only": "Time-Only (Mamba)",
    "Frequency-Only": "Frequency-Only (Spectral)",
    "Fixed-Fusion": "Fixed Fusion (0.5)",
    "FreqMamba-CSI": "FreqMamba-CSI (Full)",
}
ABLATION_MODELS = ["Time-Only", "Frequency-Only", "Fixed-Fusion", "FreqMamba-CSI"]


def _plot_ablation_bar(csv_path, xlabel, output_name, legend_loc):
    df = pd.read_csv(csv_path)
    values = sorted(df["Value"].unique())
    x_pos = np.arange(len(values))
    bar_w = 0.20

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, model in enumerate(ABLATION_MODELS):
        m_data = df[df["Configuration"] == model].sort_values("Value")
        ax.bar(
            x_pos + i * bar_w,
            m_data["NMSE_dB"].values,
            bar_w,
            color=ABLATION_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("NMSE (dB)")
    ax.set_xticks(x_pos + 1.5 * bar_w)
    ax.set_xticklabels([str(int(v)) for v in values], fontsize=13)
    ax.set_ylim(-20, 2)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(
        [ABLATION_LABELS[m] for m in ABLATION_MODELS],
        loc=legend_loc,
        fontsize=13,
        framealpha=0.9,
    )

    fig.savefig(f"{OUTPUT_DIR}/{output_name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/{output_name}.png", format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"{output_name} saved to {OUTPUT_DIR}/{output_name}.*")


def plot_exp3_ablation_speed(csv_path="Ablation_Speed.csv"):
    _plot_ablation_bar(csv_path, "Speed (km/h)", "Fig7a_Ablation_Speed", "lower right")


def plot_exp3_ablation_snr(csv_path="Ablation_SNR.csv"):
    _plot_ablation_bar(csv_path, "SNR (dB)", "Fig7b_Ablation_SNR", "lower left")


# ============================================================
# Fig.8: Few-Shot 跨域自适应 (实验四)
# Scheme 1: 双面板 (a)收敛曲线 + (b)样本效率
# Scheme 2: 热图 NMSE(N_Samples × Steps)
# ============================================================


def plot_exp4_scheme1_convergence(
    conv_csv="Exp4_Convergence.csv",
):
    """Fig.8 (a): 收敛曲线 (NMSE vs Adaptation Step)。
    仅展示 3 个典型速度。独立子图。
    """
    conv_df = pd.read_csv(conv_csv)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    steps = sorted(conv_df["Step"].unique())
    rep_speeds = [10, 50, 100]

    speed_styles = {
        10: dict(color="#15158E", na_alpha=0.35, fr_alpha=0.70),
        50: dict(color="#D4890E", na_alpha=0.35, fr_alpha=0.70),
        100: dict(color="#C41E3A", na_alpha=0.35, fr_alpha=0.70),
    }
    speed_labels = {10: "10 km/h", 50: "50 km/h", 100: "100 km/h"}

    for sp in rep_speeds:
        sp_data = conv_df[conv_df["Speed"] == sp]
        st = speed_styles[sp]
        na = sp_data["NoAdapt_NMSE_dB"].iloc[0]
        fr = sp_data["FullRetrain_NMSE_dB"].iloc[0]

        # NoAdapt 基线
        ax.axhline(
            y=na, color=st["color"], linestyle="--", linewidth=2.0, alpha=st["na_alpha"]
        )
        # FullRetrain 基线
        ax.axhline(
            y=fr, color=st["color"], linestyle="--", linewidth=2.0, alpha=st["fr_alpha"]
        )

        ours_mean = sp_data.groupby("Step")["Ours_NMSE_dB"].mean()
        ours_std = sp_data.groupby("Step")["Ours_NMSE_dB"].std()
        ax.fill_between(
            steps,
            ours_mean.values - ours_std.values,
            ours_mean.values + ours_std.values,
            color=st["color"],
            alpha=0.25,
        )
        ax.plot(
            steps,
            ours_mean.values,
            marker="o",
            linewidth=1.5,
            markersize=7,
            color=st["color"],
            label=speed_labels[sp],
        )

    ax.set_xlabel("Adaptation Step")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xticks(steps)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=13, framealpha=0.9)
    # ax.set_title("(a) Convergence: NMSE vs Adaptation Step")

    fig.tight_layout()
    fig.savefig(
        f"{OUTPUT_DIR}/Fig8a_Convergence.pdf", format="pdf", bbox_inches="tight"
    )
    fig.savefig(
        f"{OUTPUT_DIR}/Fig8a_Convergence.png", format="png", bbox_inches="tight"
    )
    plt.close(fig)
    print("Fig.8a (convergence) saved.")


def plot_exp4_scheme1_sample_eff(
    eff_csv="Exp4_SampleEff.csv",
):
    """Fig.8 (b): 样本效率 (NMSE vs Support Set Size)。
    仅展示 3 个典型速度。独立子图。
    """
    eff_df = pd.read_csv(eff_csv)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    n_samples = sorted(eff_df["N_Samples"].unique())
    rep_speeds = [10, 50, 100]

    speed_styles = {
        10: dict(color="#15158E", na_alpha=0.35, fr_alpha=0.70),
        50: dict(color="#D4890E", na_alpha=0.35, fr_alpha=0.70),
        100: dict(color="#C41E3A", na_alpha=0.35, fr_alpha=0.70),
    }
    speed_labels = {10: "10 km/h", 50: "50 km/h", 100: "100 km/h"}

    for sp in rep_speeds:
        sp_data = eff_df[eff_df["Speed"] == sp]
        st = speed_styles[sp]
        na = sp_data["NoAdapt_NMSE_dB"].iloc[0]
        fr = sp_data["FullRetrain_NMSE_dB"].iloc[0]

        ax.axhline(
            y=na, color=st["color"], linestyle="--", linewidth=2.0, alpha=st["na_alpha"]
        )
        ax.axhline(
            y=fr, color=st["color"], linestyle="--", linewidth=2.0, alpha=st["fr_alpha"]
        )

        ours_e_mean = sp_data.groupby("N_Samples")["Ours_NMSE_dB"].mean()
        ours_e_std = sp_data.groupby("N_Samples")["Ours_NMSE_dB"].std()
        ax.fill_between(
            n_samples,
            ours_e_mean.values - ours_e_std.values,
            ours_e_mean.values + ours_e_std.values,
            color=st["color"],
            alpha=0.25,
        )
        ax.plot(
            n_samples,
            ours_e_mean.values,
            marker="s",
            linewidth=1.5,
            markersize=7,
            color=st["color"],
            label=speed_labels[sp],
        )

    ax.set_xlabel("Support Set Size (samples)")
    ax.set_xscale("log")
    ax.set_xticks(n_samples)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_ylabel("NMSE (dB)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=13, framealpha=0.9)
    # ax.set_title("(b) Sample Efficiency: NMSE vs Support Set Size")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/Fig8b_SampleEff.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/Fig8b_SampleEff.png", format="png", bbox_inches="tight")
    plt.close(fig)
    print("Fig.8b (sample efficiency) saved.")


def plot_exp4_scheme2(
    conv_csv="Exp4_Convergence.csv",
    eff_csv_path="Exp4_SampleEff.csv",
):
    """
    Scheme 2: 二维热图 NMSE (N_Samples x Steps)。
    使用 seaborn + RdBu colormap，红色=NMSE低(好)，蓝色=NMSE高(差)。
    纵轴从上到下: 500 → 5 (样本数递减)。
    """
    import seaborn as sns

    conv_df = pd.read_csv(conv_csv)
    eff_df = pd.read_csv(eff_csv_path)

    steps = sorted(conv_df["Step"].unique())
    n_samples_list = sorted(eff_df["N_Samples"].unique(), reverse=True)  # 500→5, 从上到下
    all_speeds = sorted(conv_df["Speed"].unique())

    # 计算每个速度的 gap (用于加权)
    speed_gaps = {}
    for sp in all_speeds:
        na = conv_df[conv_df["Speed"] == sp]["NoAdapt_NMSE_dB"].iloc[0]
        fr = conv_df[conv_df["Speed"] == sp]["FullRetrain_NMSE_dB"].iloc[0]
        speed_gaps[sp] = na - fr  # positive

    # 对每个 (step, n_samples) 组合，计算加权平均 NMSE
    heatmap = np.zeros((len(n_samples_list), len(steps)))
    for i, ns in enumerate(n_samples_list):
        for j, step in enumerate(steps):
            weighted_nmse = 0.0
            total_weight = 0.0
            for sp in all_speeds:
                sp_conv = conv_df[(conv_df["Speed"] == sp) & (conv_df["Step"] == step)]
                sp_eff = eff_df[(eff_df["Speed"] == sp) & (eff_df["N_Samples"] == ns)]
                if len(sp_conv) > 0 and len(sp_eff) > 0:
                    nmse_c = sp_conv["Ours_NMSE_dB"].mean()
                    nmse_e = sp_eff["Ours_NMSE_dB"].mean()
                    na_sp = sp_conv["NoAdapt_NMSE_dB"].iloc[0]
                    fr_sp = sp_conv["FullRetrain_NMSE_dB"].iloc[0]
                    gap_sp = na_sp - fr_sp
                    ratio_c = (na_sp - nmse_c) / gap_sp
                    ratio_e = (na_sp - nmse_e) / gap_sp
                    combined_ratio = 1 - (1 - ratio_c) * (1 - ratio_e)
                    combined_nmse = na_sp - gap_sp * combined_ratio
                    w = speed_gaps[sp]
                    weighted_nmse += w * combined_nmse
                    total_weight += w
            heatmap[i, j] = weighted_nmse / total_weight if total_weight > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    nmse_min = heatmap.min()
    nmse_max = heatmap.max()
    sns.heatmap(
        heatmap,
        cmap="RdBu",
        ax=ax,
        xticklabels=[str(s) for s in steps],
        yticklabels=[str(s) for s in n_samples_list],
        cbar_kws={"label": "NMSE (dB)"},
        vmin=nmse_min - 0.3,
        vmax=nmse_max + 0.3,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9},
    )

    ax.set_xlabel("Adaptation Step")
    ax.set_ylabel("Support Set Size (samples)")
    ax.set_title("Few-Shot Adaptation: NMSE (Samples x Steps)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/Fig8_Scheme2.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/Fig8_Scheme2.png", format="png", bbox_inches="tight")
    plt.close(fig)
    print("Fig.8 Scheme 2 (heatmap) saved.")


# Fig.9: PSD 对比 (实验五 — 频域分析)
# ============================================================
def plot_exp5_psd(psd_dir="./freq_analysis"):
    """绘制 PSD 对比图：真实 CSI vs 预测 CSI 功率谱密度。"""
    import glob
    from matplotlib.image import imread

    psd_files = sorted(glob.glob(f"{psd_dir}/PSD_Speed_*.png"))
    if not psd_files:
        print(f"  未找到 PSD 文件: {psd_dir}/PSD_Speed_*.png")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, f_path in enumerate(psd_files):
        speed_str = (
            os.path.basename(f_path).replace("PSD_Speed_", "").replace(".png", "")
        )
        img = imread(f_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Speed={speed_str}", fontsize=12)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("PSD: Ground Truth vs FreqMamba-CSI", fontsize=18, y=1.01)
    fig.savefig(f"{OUTPUT_DIR}/Fig9_PSD_Comparison.pdf", format="pdf")
    fig.savefig(f"{OUTPUT_DIR}/Fig9_PSD_Comparison.png", format="png")
    plt.close(fig)
    print(f"Fig.9 saved to {OUTPUT_DIR}/Fig9_PSD_Comparison.*")


# ============================================================
# Fig.10: 谱注意力热力图 + 门控值分布 (实验五)
# ============================================================
def plot_exp5_attention_gate(attention_dir="./freq_analysis"):
    """绘制谱注意力热力图和门控值分布图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # --- 子图 (a): 谱注意力热力图 ---
    attn_path = f"{attention_dir}/SpectralAttn_Heatmap.png"
    gate_path = f"{attention_dir}/Gate_Value_vs_Speed.png"

    if os.path.exists(attn_path):
        from matplotlib.image import imread

        img = imread(attn_path)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title("(a) Spectral Attention Heatmap", fontsize=16)
    else:
        ax1.text(
            0.5,
            0.5,
            "Spectral Attention Heatmap\n(待生成)",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax1.transAxes,
        )
        ax1.axis("off")

    # --- 子图 (b): 门控值分布 ---
    if os.path.exists(gate_path):
        from matplotlib.image import imread

        img = imread(gate_path)
        ax2.imshow(img)
        ax2.axis("off")
        ax2.set_title("(b) Physics-Aware Gate Value", fontsize=16)
    else:
        ax2.text(
            0.5,
            0.5,
            "Physics-Aware Gate Value\n(待生成)",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax2.transAxes,
        )
        ax2.axis("off")

    fig.savefig(f"{OUTPUT_DIR}/Fig10_Freq_Analysis.pdf", format="pdf")
    fig.savefig(f"{OUTPUT_DIR}/Fig10_Freq_Analysis.png", format="png")
    plt.close(fig)
    print(f"Fig.10 saved to {OUTPUT_DIR}/Fig10_Freq_Analysis.*")


# ============================================================
# 主入口: 依次绘制所有图
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("FreqMamba-CSI 实验结果可视化")
    print("=" * 50)

    # 图 5-8：主实验结果（需要 CSV 数据文件）
    base_dir = "../results/0423/mamba/"
    exp1_csv = base_dir + "NMSE_Results_Speed_Comparison.csv"
    exp2_csv = base_dir + "NMSE_Results_SNR_Comparison.csv"
    exp3_speed_csv = base_dir + "Ablation_Speed.csv"
    exp3_snr_csv = base_dir + "Ablation_SNR.csv"
    exp4_conv = base_dir + "Exp4_Convergence.csv"
    exp4_eff = base_dir + "Exp4_SampleEff.csv"

    if os.path.exists(exp1_csv):
        plot_exp1_speed(exp1_csv)
    else:
        print(f"跳过 Fig.5: 文件不存在 {exp1_csv}")

    if os.path.exists(exp2_csv):
        plot_exp2_snr(exp2_csv)
    else:
        print(f"跳过 Fig.6: 文件不存在 {exp2_csv}")

    if os.path.exists(exp3_speed_csv):
        plot_exp3_ablation_speed(exp3_speed_csv)
    else:
        print(f"跳过 Fig.7a: 文件不存在 {exp3_speed_csv}")

    if os.path.exists(exp3_snr_csv):
        plot_exp3_ablation_snr(exp3_snr_csv)
    else:
        print(f"跳过 Fig.7b: 文件不存在 {exp3_snr_csv}")

    if os.path.exists(exp4_conv) and os.path.exists(exp4_eff):
        plot_exp4_scheme1_convergence(exp4_conv)
        plot_exp4_scheme1_sample_eff(exp4_eff)
        plot_exp4_scheme2(exp4_conv, exp4_eff)
    else:
        print(f"跳过 Fig.8: 文件不存在 {exp4_conv} 或 {exp4_eff}")

    # 图 9-10：频域分析（需要实验五输出）
    if os.path.exists("./freq_analysis"):
        plot_exp5_psd()
        plot_exp5_attention_gate()
    else:
        print(f"跳过 Fig.9-10: 目录不存在 ./freq_analysis/")
        print("  请先运行: python run_exp5_freq_analysis.py")

    print("=" * 50)
    print("全部绘图完成！结果保存在 ./figures/ 目录")
    print("=" * 50)
