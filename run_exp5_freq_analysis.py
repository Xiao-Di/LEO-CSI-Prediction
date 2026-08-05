"""
实验五: 频域分析可视化 (对应论文 Fig.9-10)

生成两类可视化:
1. PSD 对比: 真实 CSI vs 预测 CSI 的功率谱密度曲线 (各速度下)
2. 谱注意力权重热力图: FreqMamba-CSI 频域分支的 spectral_mlp 权重分布
3. 物理门控值分布: gate value vs speed (验证物理感知门控的物理合理性)

所有可视化基于单星数据。
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import (
    DEVICE, NUM_ANTENNAS, BATCH_SIZE,
    FREQ_MAMBA_D_MODEL, HISTORY_STEPS, FUTURE_STEPS,
)
from freq_mamba import FreqMambaCSI
from dataset import CSIDataset
from metrics import calculate_nmse_db

SPEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SNR_DB = 20
H_COLS = [f"H_Real_{i}" for i in range(NUM_ANTENNAS)] + [
    f"H_Imag_{i}" for i in range(NUM_ANTENNAS)
]


def load_model_and_data(model_path, speed):
    """加载预训练模型和对应速度的测试集。"""
    model = FreqMambaCSI().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    csv_path = f"./dataset/Exp1_Speed_{speed}_SNR{SNR_DB}_test.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"测试文件不存在: {csv_path}")

    # 加载数据集（需要训练集标准化参数）
    train_csv = "./dataset/Exp3_Mixed_train.csv"
    if not os.path.exists(train_csv):
        train_csv = f"./dataset/Exp1_Speed_{speed}_SNR{SNR_DB}_train.csv"

    train_df = pd.read_csv(train_csv)
    all_csi = train_df[H_COLS].values
    csi_mean = all_csi.mean(axis=0)
    csi_std = all_csi.std(axis=0) + 1e-12
    train_stats = (csi_mean, csi_std)

    dataset = CSIDataset(csv_path, stats=train_stats)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 反标准化参数
    denorm_stats = (csi_mean, csi_std)

    return model, loader, denorm_stats


def _denormalize(pred_norm, target_norm, denorm_stats):
    """将标准化后的预测/真实值还原到原始尺度。"""
    mean, std = denorm_stats
    mean_t = torch.tensor(mean, device=pred_norm.device, dtype=pred_norm.dtype).view(1, 1, -1)
    std_t = torch.tensor(std, device=pred_norm.device, dtype=pred_norm.dtype).view(1, 1, -1)
    pred = pred_norm * std_t + mean_t
    target = target_norm * std_t + mean_t
    return pred, target


def collect_predictions(model, loader, denorm_stats, max_batches=5):
    """收集模型预测结果（原始尺度），返回 pred/target 列表。"""
    preds, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x, aux, y = batch
            x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
            pred = model(x, aux)
            pred_raw, target_raw = _denormalize(pred, y, denorm_stats)
            preds.append(pred_raw.cpu())
            targets.append(target_raw.cpu())
    return preds, targets


# ============================================================================
# 可视化 1: PSD 对比
# ============================================================================

def plot_psd_comparison(preds, targets, speed, save_dir="./freq_analysis"):
    """
    绘制真实 CSI vs 预测 CSI 的功率谱密度对比。
    对每个天线分别画 PSD 曲线，取平均。
    """
    os.makedirs(save_dir, exist_ok=True)

    pred_all = torch.cat(preds, dim=0)  # [B, T_fut, CSI_dim]
    target_all = torch.cat(targets, dim=0)

    # 沿预测时间维度做 FFT
    pred_fft = torch.fft.rfft(pred_all, dim=1)  # [B, F, D]
    target_fft = torch.fft.rfft(target_all, dim=1)

    pred_psd = pred_fft.abs().pow(2).mean(dim=0)  # [F, D]  平均过 batch
    target_psd = target_fft.abs().pow(2).mean(dim=0)

    F_bins = pred_psd.size(0)
    freq_axis = np.linspace(0, 1 / (0.5e-3) / 2, F_bins)  # 基于 DT=0.5ms

    # 画 4 根天线作为代表 (0, 4, 8, 12)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    antennas = [0, 4, 8, 12]

    for j, ant_idx in enumerate(antennas):
        ax = axes[j]
        ax.semilogy(
            freq_axis, target_psd[:, ant_idx].numpy() + 1e-15,
            label="Ground Truth", linewidth=1.5, color="#1f77b4",
        )
        ax.semilogy(
            freq_axis, pred_psd[:, ant_idx].numpy() + 1e-15,
            label="FreqMamba-CSI", linewidth=1.5, color="#ff7f0e",
            linestyle="--",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title(f"Antenna {ant_idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PSD Comparison: Ground Truth vs FreqMamba-CSI (Speed={speed} km/h)")
    plt.tight_layout()
    save_path = f"{save_dir}/PSD_Speed_{speed}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PSD 图已保存: {save_path}")


def plot_psd_summary(speeds, all_preds_targets, save_dir="./freq_analysis"):
    """
    汇总所有速度的平均 PSD 误差（单图）。
    X 轴: 频率 bin, Y 轴: |PSD_pred - PSD_true| 的平均值。
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(speeds)))

    for idx, speed in enumerate(speeds):
        preds, targets = all_preds_targets[speed]
        pred_all = torch.cat(preds, dim=0)
        target_all = torch.cat(targets, dim=0)

        pred_psd = torch.fft.rfft(pred_all, dim=1).abs().pow(2).mean(dim=0).mean(dim=1)
        target_psd = torch.fft.rfft(target_all, dim=1).abs().pow(2).mean(dim=0).mean(dim=1)

        psd_error = (pred_psd - target_psd).abs().numpy()
        F_bins = len(psd_error)
        freq_axis = np.linspace(0, 1 / (0.5e-3) / 2, F_bins)

        ax.semilogy(freq_axis, psd_error + 1e-15, linewidth=1.5,
                    color=colors[idx], label=f"{speed} km/h", alpha=0.8)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean |PSD Error|")
    ax.set_title("PSD Prediction Error vs Speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = f"{save_dir}/PSD_Error_Summary.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PSD 误差汇总图已保存: {save_path}")


# ============================================================================
# 可视化 2: 谱注意力权重
# ============================================================================

def plot_spectral_attention(model, loader, speed, save_dir="./freq_analysis"):
    """
    提取并可视化频域分支的谱注意力权重。
    展示不同速度下 spectral_mlp 输出的频域权重分布。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 获取一批数据
    with torch.no_grad():
        x, aux, y = next(iter(loader))
        x, aux = x.to(DEVICE), aux.to(DEVICE)

        # 手动前向传播以提取谱注意力权重
        h_freq = model.freq_encoder(x)  # [B, T, D]

        # 重新计算谱权重
        enc_x = model.freq_encoder.enc(x) + model.freq_encoder.pos  # [B, T, D]
        X = torch.fft.rfft(enc_x, dim=1)  # [B, F, D]
        F_bins = X.size(1)
        P = X.abs()
        spec_input = P.mean(dim=2, keepdim=True)  # [B, F, 1]
        spec_weights = model.freq_encoder.spectral_mlp(spec_input)  # [B, F, 1]
        spec_weights_norm = torch.softmax(spec_weights, dim=1)  # [B, F, 1]

        # 平均过 batch 和 feature 维度
        avg_weights = spec_weights_norm.squeeze(-1).mean(dim=0).cpu().numpy()  # [F]

    freq_axis = np.linspace(0, 1 / (0.5e-3) / 2, F_bins)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(freq_axis, avg_weights, width=freq_axis[1] - freq_axis[0],
           alpha=0.7, color="#2ca02c", edgecolor="none")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"Spectral Attention Weights (Speed={speed} km/h)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_path = f"{save_dir}/SpectralAttn_Speed_{speed}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  谱注意力图已保存: {save_path}")


def plot_spectral_attention_heatmap(speeds, all_models_loaders, save_dir="./freq_analysis"):
    """
    绘制热力图: X 轴=频率 bin, Y 轴=速度, 颜色=谱注意力权重。
    展示不同速度下频域关注点的变化。
    """
    os.makedirs(save_dir, exist_ok=True)

    weight_matrix = []
    for speed in speeds:
        model, loader, _ = all_models_loaders[speed]
        with torch.no_grad():
            x, aux, y = next(iter(loader))
            x, aux = x.to(DEVICE), aux.to(DEVICE)

            enc_x = model.freq_encoder.enc(x) + model.freq_encoder.pos
            X = torch.fft.rfft(enc_x, dim=1)
            P = X.abs()
            spec_input = P.mean(dim=2, keepdim=True)
            spec_weights = model.freq_encoder.spectral_mlp(spec_input)
            spec_weights_norm = torch.softmax(spec_weights, dim=1)
            avg_w = spec_weights_norm.squeeze(-1).mean(dim=0).cpu().numpy()
            weight_matrix.append(avg_w)

    weight_matrix = np.array(weight_matrix)  # [N_speeds, F_bins]
    F_bins = weight_matrix.shape[1]
    freq_axis = np.linspace(0, 1 / (0.5e-3) / 2, F_bins)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        weight_matrix.T, aspect="auto", origin="lower",
        extent=[speeds[0], speeds[-1], freq_axis[0], freq_axis[-1]],
        cmap="YlOrRd",
    )
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectral Attention Heatmap vs Speed")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    save_path = f"{save_dir}/SpectralAttn_Heatmap.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  谱注意力热力图已保存: {save_path}")


# ============================================================================
# 可视化 3: 物理门控值分布
# ============================================================================

def plot_gate_distribution(speeds, all_models_loaders, save_dir="./freq_analysis"):
    """
    绘制物理感知门控值随速度的变化。
    验证: gate → 1 (高速依赖时域), gate → 0 (低速依赖频域)。
    """
    os.makedirs(save_dir, exist_ok=True)

    gate_values = []
    for speed in speeds:
        model, loader, _ = all_models_loaders[speed]
        gates = []
        with torch.no_grad():
            for batch in loader:
                x, aux, y = batch
                x, aux = x.to(DEVICE), aux.to(DEVICE)
                if aux.ndim == 3:
                    aux_mean = aux.mean(dim=1)
                else:
                    aux_mean = aux
                g = torch.sigmoid(model.gate.gate_net(aux_mean))
                gates.extend(g.cpu().numpy().flatten())
        gate_values.append({
            "Speed": speed,
            "Mean": np.mean(gates),
            "Std": np.std(gates),
            "Min": np.min(gates),
            "Max": np.max(gates),
            "All": gates,
        })

    # 门控值 vs 速度曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    means = [g["Mean"] for g in gate_values]
    stds = [g["Std"] for g in gate_values]
    ax.plot(speeds, means, marker="o", linewidth=2, markersize=6,
            color="#d62728", label="Gate Value (Time Domain Weight)")
    ax.fill_between(speeds,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color="#d62728", alpha=0.2)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="0.5 (Equal Fusion)")
    ax.set_xlabel("Terminal Speed (km/h)")
    ax.set_ylabel("Gate Value")
    ax.set_title("Physics-Aware Gate: Time vs Frequency Domain Weight")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_path = f"{save_dir}/Gate_Value_vs_Speed.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  门控值分布图已保存: {save_path}")

    # 门控值箱线图
    fig, ax = plt.subplots(figsize=(8, 4))
    data_for_box = [g["All"] for g in gate_values]
    bp = ax.boxplot(data_for_box, labels=speeds, patch_artist=True,
                    medianprops=dict(color="black"), showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.6)
    ax.set_xlabel("Terminal Speed (km/h)")
    ax.set_ylabel("Gate Value")
    ax.set_title("Gate Value Distribution vs Speed")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_path = f"{save_dir}/Gate_Boxplot.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  门控值箱线图已保存: {save_path}")

    # 保存门控值统计数据
    gate_df = pd.DataFrame([
        {"Speed": g["Speed"], "Mean": g["Mean"], "Std": g["Std"],
         "Min": g["Min"], "Max": g["Max"]}
        for g in gate_values
    ])
    gate_df.to_csv(f"{save_dir}/Gate_Values.csv", index=False)


# ============================================================================
# 主流程
# ============================================================================

def run_exp5_freq_analysis(model_path=None, speeds=None, output_dir="./freq_analysis"):
    """
    实验五完整流程。

    参数:
        model_path: 预训练模型路径 (默认: Exp1_Ours_pretrained.pth)
        speeds: 速度列表 (默认全部 10-100)
        output_dir: 输出目录
    """
    if model_path is None:
        model_path = "Exp1_Ours_pretrained.pth"
    if speeds is None:
        speeds = SPEEDS

    os.makedirs(output_dir, exist_ok=True)
    print(f"{'='*50}")
    print(f"实验五: 频域分析")
    print(f"模型: {model_path}")
    print(f"速度: {speeds}")
    print(f"{'='*50}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行 run_exp1_speed.py 或 run_exp3_ablation.py 生成预训练权重。")
        return

    all_models_loaders = {}
    all_preds_targets = {}

    for speed in speeds:
        print(f"\n>>> 加载速度 {speed} km/h 的数据...")
        try:
            model, loader, denorm_stats = load_model_and_data(model_path, speed)
            all_models_loaders[speed] = (model, loader, denorm_stats)

            # 收集预测结果
            preds, targets = collect_predictions(model, loader, denorm_stats)
            all_preds_targets[speed] = (preds, targets)

            nmse_batch = []
            with torch.no_grad():
                for batch in DataLoader(CSIDataset(
                    f"./dataset/Exp1_Speed_{speed}_SNR{SNR_DB}_test.csv",
                    stats=(denorm_stats[0], denorm_stats[1])
                ), batch_size=BATCH_SIZE):
                    x, aux, y = [b.to(DEVICE) for b in batch]
                    pred = model(x, aux)
                    nmse_batch.append(calculate_nmse_db(pred, y))
            print(f"  NMSE: {np.mean(nmse_batch):.4f} dB")
        except FileNotFoundError as e:
            print(f"  跳过: {e}")
            continue

    if not all_models_loaders:
        print("没有可分析的数据，退出。")
        return

    # --- 可视化 1: PSD 对比 ---
    print("\n[1/3] 绘制 PSD 对比图...")
    for speed in speeds:
        if speed in all_preds_targets:
            preds, targets = all_preds_targets[speed]
            plot_psd_comparison(preds, targets, speed, output_dir)

    plot_psd_summary(
        [s for s in speeds if s in all_preds_targets],
        all_preds_targets, output_dir,
    )

    # --- 可视化 2: 谱注意力权重 ---
    print("\n[2/3] 绘制谱注意力权重...")
    for speed in speeds:
        if speed in all_models_loaders:
            model, loader, _ = all_models_loaders[speed]
            plot_spectral_attention(model, loader, speed, output_dir)

    plot_spectral_attention_heatmap(
        [s for s in speeds if s in all_models_loaders],
        all_models_loaders, output_dir,
    )

    # --- 可视化 3: 物理门控值分布 ---
    print("\n[3/3] 绘制物理门控值分布...")
    plot_gate_distribution(
        [s for s in speeds if s in all_models_loaders],
        all_models_loaders, output_dir,
    )

    print(f"\n{'='*50}")
    print(f"实验五完成！所有图已保存至 {output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_exp5_freq_analysis()
