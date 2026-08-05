import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# --- 1. 全局仿真参数配置 (参考报告 Table I [6]) ---
TP, TF = 16, 4  # 历史 16 步, 预测 4 步
N_T = 16  # 天线数 (N)
D_MODEL = 128  # 统一隐藏层维度
N_HEAD = 4  # 注意力头数 (必须为偶数)
BATCH_SIZE = 64
LR_OFFLINE = 1e-4  # 离线预训练学习率
LR_MAML = 1e-5  # MAML在线更新学习率 [4]
# TODO
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 模型构建 (Hybrid, LSTM, Transformer) ---


# [模型 A] 双尺度混合模型: 微观(LSTM) + 宏观(Transformer) [Source 139]
class HybridCSIModel(nn.Module):
    def __init__(self):
        super(HybridCSIModel, self).__init__()
        # 微观路径 (LSTM): 捕捉短期相位漂移和小尺度随机分量 S(t)
        self.micro_path = nn.LSTM(2 * N_T, D_MODEL, num_layers=2, batch_first=True)
        # 宏观路径 (Transformer): 捕捉轨道几何驱动的大尺度分量 L(t)
        # 辅助信息: [Distance_km, Doppler_Hz, Rain_Att_dB]
        self.macro_enc = nn.Linear(3, D_MODEL)
        self.macro_path = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, batch_first=True), num_layers=2
        )
        # 融合层与解码器 (CSI Decoder [7])
        self.fusion = nn.Linear(D_MODEL * 2, D_MODEL)
        self.decoder = nn.Linear(D_MODEL, TF * 2 * N_T)

    def forward(self, csi_seq, aux_info):
        _, (h_n, _) = self.micro_path(csi_seq)
        micro_feat = h_n[-1]  # 取最后时刻隐含状态
        macro_feat = torch.mean(self.macro_path(self.macro_enc(aux_info)), dim=1)
        fused = self.fusion(torch.cat((micro_feat, macro_feat), dim=1))
        return self.decoder(fused).view(-1, TF, 2 * N_T)


# [模型 B] Baseline: Vanilla LSTM [Source 83]
class VanillaLSTM(nn.Module):
    def __init__(self):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(2 * N_T, D_MODEL, num_layers=4, batch_first=True)
        self.fc = nn.Linear(D_MODEL, TF * 2 * N_T)

    def forward(self, x, _):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).view(-1, TF, 2 * N_T)

    # [模型 C] Baseline: Vanilla Transformer [Source 83]


class VanillaTransformer(nn.Module):
    def __init__(self):
        super(VanillaTransformer, self).__init__()
        self.enc = nn.Linear(2 * N_T, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, TP, D_MODEL))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, batch_first=True), num_layers=4
        )
        self.fc = nn.Linear(D_MODEL, TF * 2 * N_T)

    def forward(self, x, _):
        x = self.enc(x) + self.pos
        feat = self.transformer(x)
        return self.fc(torch.mean(feat, dim=1)).view(-1, TF, 2 * N_T)


class CSIDataset(Dataset):
    def __init__(self, csv_file, stats=None):
        df = pd.read_csv(csv_file)
        h_cols = [f"H_Real_{i}" for i in range(N_T)] + [
            f"H_Imag_{i}" for i in range(N_T)
        ]

        # 解决 NMSE +170dB 的核心：数据标准化
        raw_csi = df[h_cols].values
        if stats is None:
            self.mean = raw_csi.mean()
            self.std = raw_csi.std() + 1e-12
        else:
            self.mean, self.std = stats

        self.samples = []
        for _, group in df.groupby("Sample_ID"):
            if len(group) < (TP + TF):
                continue
            # 区分历史观测与未来标签 [8]
            hist = group[group["Is_Future"] == 0].iloc[:TP]
            future = group[group["Is_Future"] == 1].iloc[:TF]

            x_csi = (hist[h_cols].values - self.mean) / self.std
            y_csi = (future[h_cols].values - self.mean) / self.std
            # 引入 Rain_Att_dB 替换 Sat_Lat [9, 10]
            aux = hist[["Distance_km", "Doppler_Hz", "Rain_Att_dB"]].values

            self.samples.append(
                (
                    torch.tensor(x_csi, dtype=torch.float32),
                    torch.tensor(aux, dtype=torch.float32),
                    torch.tensor(y_csi, dtype=torch.float32),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_stats(self):
        return self.mean, self.std

    # --- 4. 训练、测试与保存逻辑 ---


def calculate_nmse_db(pred, target):
    """计算 dB 域的 NMSE [11, 12]"""
    mse = torch.sum((pred - target) ** 2)
    ref = torch.sum(target**2)
    return 10 * torch.log10(mse / ref).item()


# 损失函数定义
class SAGINPhysicsLoss(nn.Module):
    def __init__(self, lambda_nmse=1.0, lambda_phys=0.5, delta_t=0.5e-3):
        super(SAGINPhysicsLoss, self).__init__()
        self.lambda_nmse = lambda_nmse  # NMSE回归权重
        self.lambda_phys = lambda_phys  # 物理一致性权重 [Source 386]
        self.dt = delta_t  # 采样间隔 (默认0.5ms) [Source 327]

    def forward(self, pred, target, aux_info, last_hist_csi):
        """
        参数:
        pred: 模型输出 [Batch, TF, 32] (标准化后的)
        target: 标签数据 [Batch, TF, 32]
        aux_info: 辅助信息 [Batch, TP, 3]，第1维是Doppler_Hz [Source 318]
        last_hist_csi: 历史序列最后一步 [Batch, 32] 用于计算相位起点
        """
        # 1. 计算 L_NMSE (回归误差基础约束) [Source 387]
        # 公式: ||H_pred - H_true||^2 / ||H_true||^2
        mse = torch.sum((pred - target) ** 2, dim=(1, 2))
        ref = torch.sum(target**2, dim=(1, 2)) + 1e-12
        l_nmse = torch.mean(mse / ref)

        # 2. 计算 L_Physics (相位演进一致性约束) [Source 387]
        # 将 pred 还原为复数 [Real: 0~15, Imag: 16~31]
        p_real, p_imag = pred[:, :, :16], pred[:, :, 16:]
        t_real, t_imag = target[:, :, :16], target[:, :, 16:]

        # 计算预测相位和真实相位
        pred_phase = torch.atan2(p_imag, p_real)  # [Batch, TF, 16]
        true_phase = torch.atan2(t_imag, t_real)

        # 提取当前多普勒频移 (取辅助信息最后一步的 Doppler)
        doppler_hz = aux_info[:, -1, 1].unsqueeze(-1).unsqueeze(-1)  # [Batch, 1, 1]

        # 计算基于 SGP4 的理论相位旋转 Delta_Phi = 2 * pi * f_d * dt [Source 384, 387]
        # 预测第 t 步相对于历史最后一帧的累计旋转
        steps = torch.arange(1, pred.size(1) + 1, device=pred.device).view(1, -1, 1)
        theoretical_rotation = 2 * 3.1415926 * doppler_hz * (steps * self.dt)

        # 提取历史最后一步的相位作为基准
        h_last_real, h_last_imag = last_hist_csi[:, :16], last_hist_csi[:, 16:]
        base_phase = torch.atan2(h_last_imag, h_last_real).unsqueeze(
            1
        )  # [Batch, 1, 16]

        # 物理一致性正则项: 强制要求相位演变符合多普勒规律
        expected_phase = base_phase + theoretical_rotation
        l_physics = F.mse_loss(pred_phase, expected_phase)

        # 3. 综合总损失 [Source 386]
        total_loss = self.lambda_nmse * l_nmse + self.lambda_phys * l_physics

        return total_loss, l_nmse, l_physics


def run_experiment_pipeline(exp_id, file_list, config_name):
    # 初始化三个对比模型
    model_classes = {
        "Hybrid": HybridCSIModel,
        "LSTM": VanillaLSTM,
        "Transformer": VanillaTransformer,
    }
    final_nmse_results = []
    csi_prediction_details = []

    # 离线预训练阶段 (使用中值文件)
    train_file = file_list[len(file_list) // 2]
    train_set = CSIDataset(train_file)
    train_stats = train_set.get_stats()
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)

    for m_name, m_class in model_classes.items():
        print(f"\n>>> 实验 {exp_id}: 正在训练模型 {m_name}")
        model = m_class().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_OFFLINE)
        criterion = (
            SAGINPhysicsLoss(lambda_nmse=1.0, lambda_phys=0.8)
            if m_name == "Hybrid"
            else nn.MSELoss()
        )

        # 离线充分训练
        loss_curve = []
        model.train()
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS} - ", end="")
            epoch_loss = 0
            for x, aux, y in train_loader:
                x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                if m_name == "Hybrid":
                    loss, nmse_val, phys_val = criterion(
                        model(x, aux), y, aux, x[:, -1, :]
                    )
                else:
                    loss = criterion(model(x, aux), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_curve.append(epoch_loss / len(train_loader))

        # 保存 Loss 曲线数据
        pd.DataFrame({"Epoch": range(1, EPOCHS + 1), "Loss": loss_curve}).to_csv(
            f"Exp{exp_id}_{m_name}_Loss.csv", index=False
        )
        # 保存离线模型权重
        torch.save(model.state_dict(), f"Exp{exp_id}_{m_name}_pretrained.pth")

        # 测试阶段 (含 MAML 在线适配 [4, 5])
        for f_path in file_list:
            config_val = (
                f_path.split("_")[-1].replace(".csv", "")
                if "SNR" not in f_path
                else f_path.split("_")[-2]
            )
            dataset = CSIDataset(f_path, stats=train_stats)
            print(
                f">>> 实验 {exp_id}: 模型 {m_name} 适配并测试配置 {config_name}={config_val}"
            )

            # --- MAML 在线演进 (Inner Loop) ---
            # 使用该新场景的前 5 个样本 (Support Set) 进行极少量梯度更新
            model.load_state_dict(
                torch.load(f"Exp{exp_id}_{m_name}_pretrained.pth")
            )  # 重置为预训练状态
            model.train()
            maml_opt = torch.optim.SGD(model.parameters(), lr=LR_MAML)
            support_loader = DataLoader(dataset, batch_size=5, shuffle=False)
            sx, s_aux, sy = next(iter(support_loader))
            sx, s_aux, sy = sx.to(DEVICE), s_aux.to(DEVICE), sy.to(DEVICE)
            maml_opt.zero_grad()
            if m_name == "Hybrid":
                maml_loss, _, _ = criterion(model(sx, s_aux), sy, s_aux, sx[:, -1, :])
            else:
                maml_loss = criterion(model(sx, s_aux), sy)
            maml_loss.backward()
            maml_opt.step()

            # --- 评估 (Query Set) ---
            model.eval()
            test_loader = DataLoader(dataset, batch_size=1)
            batch_nmse = []
            with torch.no_grad():
                for i, (x, aux, y) in enumerate(test_loader):
                    x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
                    pred = model(x, aux)
                    batch_nmse.append(calculate_nmse_db(pred, y))

                    # 记录预测细节 (修复 Scalar 报错: 提取特定天线和时间步 [13])
                    if i < 3:  # 记录每种配置前 3 个样本
                        detail = {
                            "Model": m_name,
                            config_name: config_val,
                            "Sample_ID": i,
                        }
                        for t in range(TF):
                            # y[batch, time, feature] -> y[0, t, 0] 是天线 0 的实部
                            detail[f"Real_H0_T{t}"] = y[0, t, 0].item()
                            detail[f"Pred_H0_T{t}"] = pred[0, t, 0].item()
                        csi_prediction_details.append(detail)

            avg_nmse = np.mean(batch_nmse)
            final_nmse_results.append(
                {"Model": m_name, config_name: config_val, "NMSE_dB": avg_nmse}
            )
            print(
                f"| {m_name} | {config_name}={config_val} | NMSE: {avg_nmse:.5f} dB |"
            )

    return pd.DataFrame(final_nmse_results), pd.DataFrame(csi_prediction_details)


# --- 5. 消融实验流水线 (实验三) ---
def run_ablation_study(file_list, config_name):
    """消融实验：量化双尺度架构与物理损失的独立贡献。"""
    # 配置 A: Pure Data (Hybrid + MSELoss)
    # 配置 B: Single-Scale (VanillaLSTM + SAGINPhysicsLoss)
    # 配置 C: Full (Hybrid + SAGINPhysicsLoss)
    ablation_configs = [
        {
            "name": "A_PureData",
            "model_class": HybridCSIModel,
            "use_physics_loss": False,
        },
        {
            "name": "B_SingleScale",
            "model_class": VanillaLSTM,
            "use_physics_loss": True,
        },
        {
            "name": "C_GeoHMCP_Full",
            "model_class": HybridCSIModel,
            "use_physics_loss": True,
        },
    ]

    final_results = []

    # 离线预训练（使用中值文件）
    train_file = file_list[len(file_list) // 2]
    train_set = CSIDataset(train_file)
    train_stats = train_set.get_stats()
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)

    for cfg in ablation_configs:
        m_name = cfg["name"]
        use_phys = cfg["use_physics_loss"]
        model = cfg["model_class"]().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_OFFLINE)
        criterion = (
            SAGINPhysicsLoss(lambda_nmse=1.0, lambda_phys=0.8)
            if use_phys
            else nn.MSELoss()
        )

        print(f"\n>>> 消融实验: 正在离线训练模型 {m_name}")
        loss_curve = []
        model.train()
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS} - ", end="")
            epoch_loss = 0
            for x, aux, y in train_loader:
                x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                if use_phys:
                    loss, _, _ = criterion(model(x, aux), y, aux, x[:, -1, :])
                else:
                    loss = criterion(model(x, aux), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_curve.append(epoch_loss / len(train_loader))
            print(f"Loss: {loss_curve[-1]:.6f}")

        # 保存权重
        torch.save(model.state_dict(), f"Ablation_{m_name}_pretrained.pth")
        pd.DataFrame({"Epoch": range(1, EPOCHS + 1), "Loss": loss_curve}).to_csv(
            f"Ablation_{m_name}_Loss.csv", index=False
        )

        # 测试 + MAML 在线适配
        for f_path in file_list:
            config_val = (
                f_path.split("_")[-1].replace(".csv", "")
                if "SNR" not in f_path
                else f_path.split("_")[-2]
            )
            dataset = CSIDataset(f_path, stats=train_stats)
            print(f">>> 消融实验: 模型 {m_name} 适配配置 {config_name}={config_val}")

            # --- MAML Inner Loop (5-shot) ---
            model.load_state_dict(torch.load(f"Ablation_{m_name}_pretrained.pth"))
            model.train()
            maml_opt = torch.optim.SGD(model.parameters(), lr=LR_MAML)
            support_loader = DataLoader(dataset, batch_size=5, shuffle=False)
            sx, s_aux, sy = next(iter(support_loader))
            sx, s_aux, sy = sx.to(DEVICE), s_aux.to(DEVICE), sy.to(DEVICE)
            maml_opt.zero_grad()
            if use_phys:
                maml_loss, _, _ = criterion(model(sx, s_aux), sy, s_aux, sx[:, -1, :])
            else:
                maml_loss = criterion(model(sx, s_aux), sy)
            maml_loss.backward()
            maml_opt.step()

            # --- Query Set 评估 ---
            model.eval()
            test_loader = DataLoader(dataset, batch_size=1)
            batch_nmse = []
            with torch.no_grad():
                for x, aux, y in test_loader:
                    x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
                    pred = model(x, aux)
                    batch_nmse.append(calculate_nmse_db(pred, y))

            avg_nmse = np.mean(batch_nmse)
            final_results.append(
                {"Configuration": m_name, config_name: config_val, "NMSE_dB": avg_nmse}
            )
            print(
                f"| {m_name} | {config_name}={config_val} | NMSE: {avg_nmse:.5f} dB |"
            )

    return pd.DataFrame(final_results)


if __name__ == "__main__":
    # --- 6. 执行主流程 ---
    # 实验一: 不同终端速度下的 NMSE (对应论文 Fig 5 [14])
    speeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    speed_files = [f"./dataset/Exp1_Speed_{v}_SNR0.csv" for v in speeds]
    nmse_v, detail_v = run_experiment_pipeline(1, speed_files, "Speed_kmh")
    nmse_v.to_csv("NMSE_Results_Speed_Comparison.csv", index=False)

    # 实验二: SNR (Fig.6)
    snrs = [0, 5, 10, 15, 20, 25, 30]
    snr_files = [f"./dataset/Exp2_Speed100_SNR_{s}.csv" for s in snrs]
    nmse_s, detail_s = run_experiment_pipeline(2, snr_files, "SNR_dB")
    nmse_s.to_csv("NMSE_Results_SNR_Comparison.csv", index=False)

    # 保存拼接后的真实 vs 预测细节用于绘图
    pd.concat([detail_v, detail_s]).to_csv("CSI_Real_vs_Pred_Adaptive.csv", index=False)

    # --- 实验三: 消融实验 (对应论文 Fig.7) ---
    ablation_speed_files = [
        f"./dataset/Exp1_Speed_{v}_SNR0.csv" for v in speeds
    ]  # 复用速度场景文件
    nmse_ablation = run_ablation_study(ablation_speed_files, "Speed_kmh")
    nmse_ablation.to_csv("Ablation_Study_Results.csv", index=False)

    print("\n实验完成！")
