"""
评估指标: NMSE (dB) 计算
"""

import torch


def calculate_nmse_db(pred, target):
    """
    计算 dB 域的 NMSE。
    公式: 10 * log10( ||H_pred - H_true||^2 / ||H_true||^2 )
    """
    mse = torch.sum((pred - target) ** 2)
    ref = torch.sum(target**2)
    return 10 * torch.log10(mse / ref).item()
