"""
实验一: 不同终端速度下的 NMSE (对应论文 Fig.5)
训练策略: 全部速度混合训练 → 所有速度直接推理 (无 MAML)
数据划分: 每个速度 10000 样本 → train 8000 (80/20 训练/验证) + test 2000

对比模型:
- VanillaLSTM: 纯 LSTM 基线
- VanillaTransformer: 纯 Transformer 基线
- FreqMamba-CSI (Ours): 时频双路 Mamba
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd

from config import DEVICE, NUM_ANTENNAS, BATCH_SIZE, EARLY_STOP_PATIENCE, LAMBDA_PHYS
from models import VanillaLSTM, VanillaTransformer
from freq_mamba import FreqMambaCSI
from dataset import CSIDataset
from losses import SAGINPhysicsLoss
from train_utils import train_model, evaluate_model, parse_config_value

H_COLS = [f"H_Real_{i}" for i in range(NUM_ANTENNAS)] + [
    f"H_Imag_{i}" for i in range(NUM_ANTENNAS)
]
AUX_COLS = ["Distance_km", "Doppler_Hz", "Rain_Att_dB"]
MIXED_CSV = "./dataset/Exp1_Mixed_train.csv"


def _build_mixed_csv(source_files, output_path):
    """合并多个场景 CSV 为一个混合文件。"""
    if os.path.exists(output_path):
        print(f"混合文件已存在: {output_path}，跳过合并")
        return
    print(f"合并 {len(source_files)} 个文件 → {output_path}")
    dfs = [pd.read_csv(f) for f in source_files]
    pd.concat(dfs, ignore_index=True).to_csv(output_path, index=False)


def run_exp1(speeds=None, speed_dir="./dataset"):
    if speeds is None:
        speeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    train_files = [f"{speed_dir}/Exp1_Speed_{v}_SNR20_train.csv" for v in speeds]
    test_files = [f"{speed_dir}/Exp1_Speed_{v}_SNR20_test.csv" for v in speeds]
    config_name = "Speed_kmh"

    model_classes = {
        "LSTM": VanillaLSTM,
        "Transformer": VanillaTransformer,
        "Ours": FreqMambaCSI,
    }

    final_results = []
    all_details = []

    # --- 离线预训练: 混合 train 文件 ---
    _build_mixed_csv(train_files, MIXED_CSV)
    df = pd.read_csv(MIXED_CSV)
    all_csi = df[H_COLS].values
    global_mean = all_csi.mean(axis=0)
    global_std = all_csi.std(axis=0) + 1e-12
    train_stats = (global_mean, global_std)

    # 单星数据集（所有模型统一使用）
    full_set = CSIDataset(MIXED_CSV, stats=train_stats)
    n_total = len(full_set)
    n_train = int(0.8 * n_total)

    rng = np.random.RandomState(42)
    idx = rng.permutation(n_total)

    train_set = Subset(full_set, idx[:n_train])
    val_set = Subset(full_set, idx[n_train:])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    for m_name, m_class in model_classes.items():
        print(f"\n>>> 实验一: 正在训练模型 {m_name}")
        model = m_class().to(DEVICE)

        use_phys = m_name == "Ours"
        criterion = (
            SAGINPhysicsLoss(lambda_phys=LAMBDA_PHYS)
            if use_phys
            else torch.nn.MSELoss()
        )

        loss_curve, stopped_epoch = train_model(
            model, train_loader, criterion, use_phys,
            val_loader=val_loader, early_stop=EARLY_STOP_PATIENCE,
        )
        pd.DataFrame({"Epoch": range(1, stopped_epoch + 1), "Loss": loss_curve}).to_csv(
            f"Exp1_{m_name}_Loss.csv", index=False
        )
        torch.save(model.state_dict(), f"Exp1_{m_name}_pretrained.pth")

        # --- 测试: 在独立 test 集上推理 ---
        model.eval()
        for f_path in test_files:
            config_val = parse_config_value(f_path)
            dataset = CSIDataset(f_path, stats=train_stats)
            test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            print(f">>> 测试 {m_name} @ {config_name}={config_val}")

            avg_nmse, details = evaluate_model(
                model,
                test_loader,
                return_details=True,
                config_label=config_name,
                config_value=config_val,
            )

            final_results.append(
                {"Model": m_name, config_name: config_val, "NMSE_dB": avg_nmse}
            )
            all_details.append(details)

            print(
                f"| {m_name} | {config_name}={config_val} | NMSE: {avg_nmse:.5f} dB |"
            )

    nmse_df = pd.DataFrame(final_results)
    details_df = pd.concat(all_details, ignore_index=True)

    nmse_df.to_csv("NMSE_Results_Speed_Comparison.csv", index=False)
    details_df.to_csv("Exp1_Real_vs_Pred.csv", index=False)
    print("\n实验一完成！")
    return nmse_df, details_df


if __name__ == "__main__":
    run_exp1()
