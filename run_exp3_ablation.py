"""
实验三: 消融实验 (对应论文 Fig.7)
量化时频双路各组件的独立贡献。

对比配置 (3个):
  Time-Only:       仅时域 Mamba (gate=1, 无物理损失)
  Fixed-Fusion:    固定 0.5 融合 (gate=0.5, 有物理损失)
  FreqMamba-CSI:   完整模型 (物理感知门控 + 物理损失)

输出两组柱状图:
  (a) 全 Speed 对比 (SNR=20dB 固定)
  (b) 全 SNR 对比 (Speed=50 km/h 固定)
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd

from config import DEVICE, NUM_ANTENNAS, BATCH_SIZE, EARLY_STOP_PATIENCE, LAMBDA_PHYS
from freq_mamba import FreqMambaCSI
from dataset import CSIDataset
from losses import SAGINPhysicsLoss
from train_utils import train_model, evaluate_model

H_COLS = [f"H_Real_{i}" for i in range(NUM_ANTENNAS)] + [
    f"H_Imag_{i}" for i in range(NUM_ANTENNAS)
]

ABLATION_CONFIGS = [
    {"name": "Time-Only",    "model_class": lambda: FreqMambaCSI(time_only=True),  "use_phys": False},
    {"name": "Fixed-Fusion", "model_class": lambda: FreqMambaCSI(no_gate=True),    "use_phys": True},
    {"name": "FreqMamba-CSI","model_class": lambda: FreqMambaCSI(),                "use_phys": True},
]


def _build_dataset(data_dir="./dataset"):
    """构建混合训练集并返回统计量。"""
    speeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    source_files = [f"{data_dir}/Exp1_Speed_{v}_SNR20_train.csv" for v in speeds]
    mixed_csv = "./dataset/Exp3_Mixed_train.csv"

    if not os.path.exists(mixed_csv):
        dfs = [pd.read_csv(f) for f in source_files if os.path.exists(f)]
        pd.concat(dfs, ignore_index=True).to_csv(mixed_csv, index=False)

    df = pd.read_csv(mixed_csv)
    all_csi = df[H_COLS].values
    stats = (all_csi.mean(axis=0), all_csi.std(axis=0) + 1e-12)
    return mixed_csv, stats


def _train_and_eval(cfg, train_csv, stats, test_files, config_name):
    """训练单个消融配置并在所有测试文件上评估。"""
    full_set = CSIDataset(train_csv, stats=stats)
    n_train = int(0.8 * len(full_set))
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(full_set))

    train_loader = DataLoader(
        Subset(full_set, idx[:n_train]), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        Subset(full_set, idx[n_train:]), batch_size=BATCH_SIZE, shuffle=False
    )

    model = cfg["model_class"]().to(DEVICE)
    criterion = (
        SAGINPhysicsLoss(lambda_phys=LAMBDA_PHYS)
        if cfg["use_phys"]
        else torch.nn.MSELoss()
    )

    print(f"\n>>> 训练模型: {cfg['name']}")
    loss_curve, stopped_epoch = train_model(
        model, train_loader, criterion, cfg["use_phys"],
        val_loader=val_loader, early_stop=EARLY_STOP_PATIENCE,
    )
    torch.save(model.state_dict(), f"Ablation_{cfg['name']}_pretrained.pth")

    # 测试
    model.eval()
    results = []
    for f_path in test_files:
        config_val = int(f_path.split("_")[-1].replace(".csv", "").replace("Speed_", "").replace("SNR", ""))
        # Extract config value more robustly
        for part in os.path.basename(f_path).split("_"):
            if part.startswith("Speed") or part.startswith("SNR"):
                config_val = int(part.replace("Speed_", "").replace("SNR", ""))
                break

        test_loader = DataLoader(
            CSIDataset(f_path, stats=stats), batch_size=BATCH_SIZE
        )
        avg_nmse = evaluate_model(model, test_loader)
        results.append({"Configuration": cfg["name"], config_name: config_val, "NMSE_dB": avg_nmse})
        print(f"  {cfg['name']} @ {config_name}={config_val}: NMSE = {avg_nmse:.2f} dB")
    return results


def run_exp3_ablation_speed(data_dir="./dataset"):
    """消融实验: 全 Speed 对比 (SNR=20dB 固定)。"""
    train_csv, stats = _build_dataset(data_dir)
    all_results = []

    speeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    test_files = [f"{data_dir}/Exp1_Speed_{v}_SNR20_test.csv" for v in speeds]

    for cfg in ABLATION_CONFIGS:
        results = _train_and_eval(cfg, train_csv, stats, test_files, "Speed_kmh")
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df["Dimension"] = "Speed"
    results_df.rename(columns={"Speed_kmh": "Value"}, inplace=True)
    return results_df


def run_exp3_ablation_snr(data_dir="./dataset"):
    """消融实验: 全 SNR 对比 (Speed=50 km/h 固定)。"""
    train_csv, stats = _build_dataset(data_dir)
    all_results = []

    snrs = [0, 5, 10, 15, 20, 25, 30]
    test_files = [f"{data_dir}/Exp2_SNR_{v}_Speed50_test.csv" for v in snrs]

    for cfg in ABLATION_CONFIGS:
        results = _train_and_eval(cfg, train_csv, stats, test_files, "SNR_dB")
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df["Dimension"] = "SNR"
    results_df.rename(columns={"SNR_dB": "Value"}, inplace=True)
    return results_df


if __name__ == "__main__":
    print("=" * 50)
    print("实验三: 消融实验 (双维度柱状图)")
    print("=" * 50)

    speed_results = run_exp3_ablation_speed()
    snr_results = run_exp3_ablation_snr()

    combined = pd.concat([speed_results, snr_results], ignore_index=True)
    combined.to_csv("Ablation_Study_Results.csv", index=False)

    print("\n=== 消融实验结果 ===")
    print(combined.to_string(index=False))
    print("\n实验三完成！")
