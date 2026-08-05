"""
谱一致性损失 (Spectral Consistency Loss)。

用于 MAML++ 内循环：约束预测 CSI 与真实 CSI 的功率谱密度一致，
确保模型在频域上也保持物理合理性。
"""

import torch
import torch.nn as nn


class SpectralConsistencyLoss(nn.Module):
    """
    功率谱密度一致性损失。

    计算预测和真实 CSI 序列的功率谱密度 (PSD) 之间的 MSE。
    PSD 在时间维度上计算（对每个天线/特征独立）。

    输入:
        pred:   [B, T_fut, CSI_dim]  预测 CSI
        target: [B, T_fut, CSI_dim]  真实 CSI
    返回:
        l_spectral: 标量，PSD MSE
    """

    def forward(self, pred, target):
        # FFT 沿时间维度
        pred_fft = torch.fft.rfft(pred, dim=1)  # [B, T//2+1, CSI_dim]
        target_fft = torch.fft.rfft(target, dim=1)

        # 功率谱密度
        pred_psd = pred_fft.abs().pow(2)  # [B, T//2+1, CSI_dim]
        target_psd = target_fft.abs().pow(2)

        # 归一化
        pred_psd_norm = pred_psd / (pred_psd.sum(dim=1, keepdim=True) + 1e-12)
        target_psd_norm = target_psd / (target_psd.sum(dim=1, keepdim=True) + 1e-12)

        # MSE between normalized PSDs
        l_spectral = torch.mean((pred_psd_norm - target_psd_norm) ** 2)
        return l_spectral
