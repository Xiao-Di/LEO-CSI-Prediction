"""
实验四: Few-Shot 跨域自适应 (对应论文 Fig.8)

核心叙事: 预训练模型(城市) → 目标域(丛林) 性能下降，用 MAML 框架在极少样本、极少步数
内完成适配，性能逼近从头重新训练。

对比方法:
  No Adaptation: 城市预训练权重直接推理，零适配（下界）
  Ours:          MAML 内循环适配，双域损失 + 参数正则（本文方法）
  Full Retrain:  在丛林数据上从头训练至收敛（上界）

输出:
  Exp4_Convergence.csv  — 适配步数 vs NMSE（固定样本量=20）
  Exp4_SampleEff.csv    — 支持集样本数 vs NMSE（固定步数=5）
"""

import os
import time
import copy
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from config import (
    DEVICE, LR_MAML, BATCH_SIZE, LAMBDA_PHYS,
    LAMBDA_SPEC, LAMBDA_REG_MAML,
)
from freq_mamba import FreqMambaCSI
from dataset import CSIDataset
from losses import SAGINPhysicsLoss
from spectral_loss import SpectralConsistencyLoss
from metrics import calculate_nmse_db

H_COLS = [f"H_Real_{i}" for i in range(2 * 16)]

# 收敛曲线: 测试的步数点
CONVERGENCE_STEPS = [1, 2, 3, 4, 5, 8, 10]
# 样本效率: 测试的支持集大小
SAMPLE_SIZES = [5, 10, 20, 50, 100, 200, 500]
# 固定交叉参数
FIXED_N_SAMPLES = 20    # 收敛曲线固定用 20 个样本
FIXED_N_STEPS = 5       # 样本效率曲线固定用 5 步

SPEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _evaluate(model, loader):
    """评估数据集的平均 NMSE (dB)。"""
    model.eval()
    nmse_list = []
    with torch.no_grad():
        for batch in loader:
            x, aux, y = batch
            x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
            pred = model(x, aux)
            nmse_list.append(calculate_nmse_db(pred, y))
    return sum(nmse_list) / len(nmse_list)


def _load_pretrained(path):
    """加载预训练权重并冻结为 deepcopy。"""
    model = FreqMambaCSI().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model


def _adapt_model(
    model, support_loader, query_loader, n_steps,
    criterion_phys, criterion_spec, init_params=None,
    use_spectral=True, use_reg=True,
):
    """
    执行内循环适配。
    返回: (step_nmse_list, step_time_list, final_nmse)
    """
    opt = torch.optim.SGD(model.parameters(), lr=LR_MAML)
    model.train()
    step_nmse_list = []
    step_time_list = []
    cum_time = 0.0

    batch = next(iter(support_loader))
    sx, s_aux, sy = [t.to(DEVICE) for t in batch]

    for step in range(n_steps):
        t0 = time.time()
        opt.zero_grad()
        pred = model(sx, s_aux)
        loss_nmse, _, _ = criterion_phys(pred, sy, s_aux, sx[:, -1, :])
        loss = loss_nmse

        if use_spectral:
            l_spec = criterion_spec(pred, sy)
            loss = loss + LAMBDA_SPEC * l_spec

        if use_reg and init_params is not None:
            loss_reg = sum(
                (p - init_params[n]).pow(2).sum()
                for n, p in model.named_parameters()
            )
            loss = loss + LAMBDA_REG_MAML * loss_reg

        loss.backward()
        opt.step()
        cum_time += time.time() - t0

        model.eval()
        nmse = _evaluate(model, query_loader)
        model.train()

        step_nmse_list.append(nmse)
        step_time_list.append(cum_time)

    return step_nmse_list, step_time_list, step_nmse_list[-1]


def run_exp4(
    pretrained_path,
    jungle_dir="./dataset",
    urban_train_stats=None,
    snr_db=20,
    n_repeats=3,
):
    """
    实验四: 生成收敛曲线 + 样本效率曲线数据。
    对每个速度执行 No Adapt / Ours / Full Retrain 三级对比。
    """
    # 城市训练集归一化参数
    if urban_train_stats is None:
        urban_csv = "./dataset/Exp1_Mixed_train.csv"
        if os.path.exists(urban_csv):
            urban_df = pd.read_csv(urban_csv)
            all_csi = urban_df[H_COLS].values
            urban_train_stats = (all_csi.mean(axis=0), all_csi.std(axis=0) + 1e-12)
        else:
            urban_train_stats = (None, None)

    csi_stats = urban_train_stats

    criterion_phys = SAGINPhysicsLoss(lambda_phys=LAMBDA_PHYS)
    criterion_spec = SpectralConsistencyLoss()
    criterion_mse = torch.nn.MSELoss()

    convergence_rows = []  # Step, Speed, NoAdapt, Ours, FullRetrain
    sample_eff_rows = []   # N_Samples, Speed, NoAdapt, Ours, FullRetrain

    for speed in SPEEDS:
        target_csv = f"{jungle_dir}/Exp4_Jungle_Speed_{speed}_SNR{snr_db}_test.csv"
        if not os.path.exists(target_csv):
            print(f"跳过: {target_csv}")
            continue

        target_dataset = CSIDataset(target_csv, stats=csi_stats)
        full_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"\n=== Speed {speed} km/h ===")

        # ---- No Adaptation ----
        no_adapt_model = _load_pretrained(pretrained_path)
        no_adapt_nmse = _evaluate(no_adapt_model, full_loader)
        print(f"  No Adapt NMSE: {no_adapt_nmse:.4f} dB")

        # ---- Full Retrain ----
        full_model = _load_pretrained(pretrained_path)
        full_opt = torch.optim.Adam(full_model.parameters(), lr=1e-4)
        retrain_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)
        full_model.train()
        for _ in range(50):
            for batch in retrain_loader:
                full_opt.zero_grad()
                x, aux, y = [t.to(DEVICE) for t in batch]
                loss = criterion_mse(full_model(x, aux), y)
                loss.backward()
                full_opt.step()
        full_model.eval()
        full_nmse = _evaluate(full_model, full_loader)
        print(f"  Full Retrain NMSE: {full_nmse:.4f} dB")

        # ---- Convergence curve (fixed samples=20, vary steps) ----
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(target_dataset))
        support_indices = indices[:FIXED_N_SAMPLES].tolist()
        query_indices = indices[FIXED_N_SAMPLES:].tolist()

        support_ds = torch.utils.data.Subset(target_dataset, support_indices)
        query_ds = torch.utils.data.Subset(target_dataset, query_indices)
        support_loader = DataLoader(support_ds, batch_size=min(20, FIXED_N_SAMPLES), shuffle=True)
        query_loader = DataLoader(query_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Ours 收敛曲线 (多次随机重复)
        for rep in range(n_repeats):
            rng_rep = np.random.RandomState(42 + rep * 1000)
            rep_indices = rng_rep.permutation(len(target_dataset))
            rep_sup_idx = rep_indices[:FIXED_N_SAMPLES].tolist()
            rep_qry_idx = rep_indices[FIXED_N_SAMPLES:].tolist()

            rep_sup_ds = torch.utils.data.Subset(target_dataset, rep_sup_idx)
            rep_qry_ds = torch.utils.data.Subset(target_dataset, rep_qry_idx)
            rep_sup_loader = DataLoader(rep_sup_ds, batch_size=min(20, FIXED_N_SAMPLES), shuffle=True)
            rep_qry_loader = DataLoader(rep_qry_ds, batch_size=BATCH_SIZE, shuffle=False)

            model = _load_pretrained(pretrained_path)
            init_params = {n: p.clone() for n, p in model.named_parameters()}
            step_nmse_list, _, _ = _adapt_model(
                model, rep_sup_loader, rep_qry_loader,
                n_steps=CONVERGENCE_STEPS[-1],
                criterion_phys=criterion_phys,
                criterion_spec=criterion_spec,
                init_params=init_params,
                use_spectral=True, use_reg=True,
            )
            for i, s in enumerate(CONVERGENCE_STEPS):
                convergence_rows.append({
                    "Speed": speed, "Repeat": rep, "Step": s,
                    "NoAdapt_NMSE_dB": no_adapt_nmse,
                    "Ours_NMSE_dB": step_nmse_list[s - 1],
                    "FullRetrain_NMSE_dB": full_nmse,
                })
        print(f"  Ours convergence done (steps={CONVERGENCE_STEPS})")

        # ---- Sample efficiency (fixed steps=5, vary samples) ----
        for n_samples in SAMPLE_SIZES:
            if n_samples >= len(target_dataset):
                continue
            rng2 = np.random.RandomState(42)
            indices2 = rng2.permutation(len(target_dataset))
            sup_idx = indices2[:n_samples].tolist()
            qry_idx = indices2[n_samples:].tolist()

            sup_ds = torch.utils.data.Subset(target_dataset, sup_idx)
            qry_ds = torch.utils.data.Subset(target_dataset, qry_idx)
            sup_loader = DataLoader(sup_ds, batch_size=min(n_samples, BATCH_SIZE), shuffle=True)
            qry_loader = DataLoader(qry_ds, batch_size=BATCH_SIZE, shuffle=False)

            for rep in range(n_repeats):
                rng_rep2 = np.random.RandomState(42 + rep * 1000 + n_samples)
                rep_idx2 = rng_rep2.permutation(len(target_dataset))
                rep_sup2 = rep_idx2[:n_samples].tolist()
                rep_qry2 = rep_idx2[n_samples:].tolist()

                rep_sup_ds2 = torch.utils.data.Subset(target_dataset, rep_sup2)
                rep_qry_ds2 = torch.utils.data.Subset(target_dataset, rep_qry2)
                rep_sup_loader2 = DataLoader(rep_sup_ds2, batch_size=min(n_samples, BATCH_SIZE), shuffle=True)
                rep_qry_loader2 = DataLoader(rep_qry_ds2, batch_size=BATCH_SIZE, shuffle=False)

                model = _load_pretrained(pretrained_path)
                init_params = {n: p.clone() for n, p in model.named_parameters()}
                _, _, final_nmse = _adapt_model(
                    model, rep_sup_loader2, rep_qry_loader2,
                    n_steps=FIXED_N_STEPS,
                    criterion_phys=criterion_phys,
                    criterion_spec=criterion_spec,
                    init_params=init_params,
                    use_spectral=True, use_reg=True,
                )
                sample_eff_rows.append({
                    "Speed": speed, "Repeat": rep, "N_Samples": n_samples,
                    "NoAdapt_NMSE_dB": no_adapt_nmse,
                    "Ours_NMSE_dB": final_nmse,
                    "FullRetrain_NMSE_dB": full_nmse,
                })
        print(f"  Ours sample efficiency done (sizes={SAMPLE_SIZES})")

    conv_df = pd.DataFrame(convergence_rows)
    eff_df = pd.DataFrame(sample_eff_rows)
    return conv_df, eff_df


if __name__ == "__main__":
    PRETRAINED_PATH = "Exp1_Ours_pretrained.pth"
    JUNGLE_DIR = "./dataset"
    SNR = 20

    conv_df, eff_df = run_exp4(
        pretrained_path=PRETRAINED_PATH,
        jungle_dir=JUNGLE_DIR,
        snr_db=SNR,
        n_repeats=3,
    )

    conv_df.to_csv("Exp4_Convergence.csv", index=False)
    eff_df.to_csv("Exp4_SampleEff.csv", index=False)

    print("\n=== 实验四完成 ===")
    print("收敛曲线 -> Exp4_Convergence.csv")
    print("样本效率 -> Exp4_SampleEff.csv")
