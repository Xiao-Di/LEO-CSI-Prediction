"""
损失函数: SAGINPhysicsLoss
复合损失: L_total = λ_NMSE * L_NMSE + λ_Phys * L_Physics

物理约束逻辑:
- 用 cosine 相似度约束预测 CSI 的相位演进方向与真实 CSI 一致
- 使用 wrapped phase difference 避免 2π 模环绕问题
- 物理损失作为正则项，量纲与 NMSE 一致 (无量纲)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_ANTENNAS, DT, LAMBDA_NMSE, LAMBDA_PHYS


class SAGINPhysicsLoss(nn.Module):
    """
    SAGIN 物理一致性损失函数。
    融合 NMSE 回归误差与多普勒相位演进正则项。
    """

    def __init__(
        self,
        lambda_nmse=LAMBDA_NMSE,
        lambda_phys=LAMBDA_PHYS,
        delta_t=None,
    ):
        super(SAGINPhysicsLoss, self).__init__()
        self.lambda_nmse = lambda_nmse
        self.lambda_phys = lambda_phys
        self.dt = delta_t if delta_t is not None else DT

    def forward(self, pred, target, aux_info, last_hist_csi):
        """
        参数:
            pred:          模型输出 [Batch, TF, 32] (标准化后)
            target:        真实标签 [Batch, TF, 32] (标准化后)
            aux_info:      辅助信息 [Batch, TP, 3]，第1维是Doppler_Hz
            last_hist_csi: 历史序列最后一步 [Batch, 32]，用于相位一致性检查

        返回:
            (total_loss, l_nmse, l_physics)
        """
        # 1. L_NMSE: 回归误差基础约束
        #    公式: ||H_pred - H_true||^2 / ||H_true||^2
        mse = torch.sum((pred - target) ** 2, dim=(1, 2))
        ref = torch.sum(target**2, dim=(1, 2)) + 1e-12
        l_nmse = torch.mean(mse / ref)

        # 2. L_Physics: 相位演进一致性约束
        # 提取预测和真实的实部/虚部
        p_real, p_imag = pred[:, :, :NUM_ANTENNAS], pred[:, :, NUM_ANTENNAS:]
        t_real, t_imag = target[:, :, :NUM_ANTENNAS], target[:, :, NUM_ANTENNAS:]

        pred_phase = torch.atan2(p_imag, p_real)    # [Batch, TF, N_T]
        true_phase = torch.atan2(t_imag, t_real)     # [Batch, TF, N_T]

        # --- 修复 1: 使用 wrapped phase difference ---
        # 预测序列内部相位差分: Δφ[t] = wrap(φ[t] - φ[t-1])
        pred_phase_raw = pred_phase[:, 1:, :] - pred_phase[:, :-1, :]  # [B, TF-1, N_T]
        true_phase_raw = true_phase[:, 1:, :] - true_phase[:, :-1, :]  # [B, TF-1, N_T]

        # Wrapped difference: 将差值限制在 [-π, π]
        pred_phase_diff = torch.atan2(torch.sin(pred_phase_raw), torch.cos(pred_phase_raw))
        true_phase_diff = torch.atan2(torch.sin(true_phase_raw), torch.cos(true_phase_raw))

        # --- 修复 2: 用 cosine 相似度而非 MSE ---
        # cos(Δφ_pred - Δφ_true) = 1 表示完全一致, = -1 表示完全相反
        # L = 1 - cos(...) 范围 [0, 2], 无量纲
        phase_diff_error = pred_phase_diff - true_phase_diff  # [B, TF-1, N_T]
        l_physics_internal = torch.mean(1 - torch.cos(phase_diff_error))

        # --- 历史→预测连接点的相位一致性 ---
        h_last_real, h_last_imag = last_hist_csi[:, :NUM_ANTENNAS], last_hist_csi[:, NUM_ANTENNAS:]
        base_phase = torch.atan2(h_last_imag, h_last_real)  # [Batch, N_T]
        hist_to_pred_raw = pred_phase[:, 0, :] - base_phase  # [Batch, N_T]
        hist_to_pred = torch.atan2(torch.sin(hist_to_pred_raw), torch.cos(hist_to_pred_raw))

        true_boundary_raw = true_phase[:, 0, :] - base_phase
        true_boundary = torch.atan2(torch.sin(true_boundary_raw), torch.cos(true_boundary_raw))

        l_physics_boundary = torch.mean(1 - torch.cos(hist_to_pred - true_boundary))

        # 加权组合
        l_physics = 0.7 * l_physics_internal + 0.3 * l_physics_boundary

        # 3. 综合总损失
        total_loss = self.lambda_nmse * l_nmse + self.lambda_phys * l_physics

        return total_loss, l_nmse, l_physics
