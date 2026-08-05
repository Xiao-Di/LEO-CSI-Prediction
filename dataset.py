"""
数据集加载: CSIDataset (单星基线), MultiSatCSIDataset (多星并发)
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from config import (
    NUM_ANTENNAS, HISTORY_STEPS, FUTURE_STEPS,
    MAX_SATELLITES_PER_SAMPLE, GNO_NODE_DIM, AUX_DIM,
)

H_COLS = [f"H_Real_{i}" for i in range(NUM_ANTENNAS)] + [
    f"H_Imag_{i}" for i in range(NUM_ANTENNAS)
]
AUX_COLS = [
    "Distance_km", "Doppler_Hz", "Rain_Att_dB",  # 原始特征
    "cos_el", "dir_cos",                          # 派生几何特征
]

# 多星节点属性: [Distance_km, Doppler_Hz, Rain_Att_dB, cos_el, dir_cos, Sat_Azimuth, Sat_Elevation]
NODE_ATTR_COLS = [
    "Distance_km", "Doppler_Hz", "Rain_Att_dB",
    "cos_el", "dir_cos", "Sat_Azimuth", "Sat_Elevation",
]


class CSIDataset(Dataset):
    """
    单星 CSI 数据集 (用于基线模型)。
    按 Sample_ID 分组，区分历史观测 (Is_Future==0) 与未来标签 (Is_Future==1)
    """

    def __init__(self, csv_file, stats=None):
        df = pd.read_csv(csv_file)

        # 兼容多星 CSV: 仅取主星
        if "Sat_Index" in df.columns:
            df = df[df["Sat_Index"] == 0].reset_index(drop=True)

        raw_csi = df[H_COLS].values
        if stats is None:
            self.mean = np.zeros(raw_csi.shape[1])
            self.std = raw_csi.std(axis=0) + 1e-12
        else:
            _, self.std = stats
            self.mean = np.zeros(raw_csi.shape[1])

        self.samples = []
        for _, group in df.groupby("Sample_ID"):
            if len(group) < (HISTORY_STEPS + FUTURE_STEPS):
                continue
            hist = group[group["Is_Future"] == 0].iloc[:HISTORY_STEPS]
            future = group[group["Is_Future"] == 1].iloc[:FUTURE_STEPS]

            x_csi = (hist[H_COLS].values - self.mean) / self.std
            y_csi = (future[H_COLS].values - self.mean) / self.std
            aux = hist[AUX_COLS].values

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


class MultiSatCSIDataset(Dataset):
    """
    多星并发 CSI 数据集 (用于 HybridV2 GNO 模型)。

    按 (Sample_ID) 分组，每样本包含 N_sat 颗卫星 (1..MAX_SATELLITES_PER_SAMPLE)。
    对每颗卫星提取历史 CSI + 节点属性，仅主星 (Sat_Index=0) 作为预测目标。

    返回:
        csi_hist:   [N_max, T_hist, CSI_dim]  所有卫星历史 CSI (pad=0)
        node_attrs: [N_max, AUX_DIM]           所有卫星节点属性 (pad=0)
        y_csi:      [T_fut, CSI_dim]           主星未来 CSI
        mask:       [N_max]                    有效卫星掩码 (1=valid, 0=pad)
    """

    def __init__(self, csv_file, csi_stats=None, attr_stats=None):
        df = pd.read_csv(csv_file)

        assert "Sat_Index" in df.columns, "MultiSatCSIDataset requires Sat_Index column"

        # CSI 标准化
        raw_csi = df[H_COLS].values
        if csi_stats is None:
            self.csi_mean = np.zeros(raw_csi.shape[1])
            self.csi_std = raw_csi.std(axis=0) + 1e-12
        else:
            _, self.csi_std = csi_stats
            self.csi_mean = np.zeros(raw_csi.shape[1])

        # 节点属性标准化 (从训练集计算)
        if attr_stats is None:
            self.attr_mean = None
            self.attr_std = None
        else:
            self.attr_mean, self.attr_std = attr_stats

        self.n_max = MAX_SATELLITES_PER_SAMPLE
        self.samples = []

        for sample_id, group in df.groupby("Sample_ID"):
            sats_in_sample = group["Sat_Index"].nunique()
            if sats_in_sample < 1:
                continue

            # 检查是否有足够的历史+未来步
            sat0 = group[group["Sat_Index"] == 0]
            if len(sat0) < (HISTORY_STEPS + FUTURE_STEPS):
                continue

            # 主星未来标签
            future = sat0[sat0["Is_Future"] == 1].iloc[:FUTURE_STEPS]
            y_csi = (future[H_COLS].values - self.csi_mean) / self.csi_std

            # 每颗卫星的历史 CSI + 节点属性
            csi_hist_list = []
            attr_list = []
            mask_list = []

            for sat_idx in range(self.n_max):
                sat_data = group[group["Sat_Index"] == sat_idx]
                if len(sat_data) == 0:
                    # 填充
                    csi_hist_list.append(np.zeros((HISTORY_STEPS, len(H_COLS))))
                    attr_list.append(np.zeros(AUX_DIM))
                    mask_list.append(0)
                else:
                    hist = sat_data[sat_data["Is_Future"] == 0].iloc[:HISTORY_STEPS]
                    if len(hist) < HISTORY_STEPS:
                        continue  # 跳过不完整的

                    x_csi = (hist[H_COLS].values - self.csi_mean) / self.csi_std
                    # pad 如果不足
                    if len(x_csi) < HISTORY_STEPS:
                        pad = np.zeros((HISTORY_STEPS - len(x_csi), x_csi.shape[1]))
                        x_csi = np.vstack([x_csi, pad])

                    # 节点属性 (取第一行，假设宏观参数不随时间变化)
                    attr = hist[NODE_ATTR_COLS].iloc[0].values.astype(np.float32)

                    csi_hist_list.append(x_csi)
                    attr_list.append(attr)
                    mask_list.append(1)

            if len(csi_hist_list) == 0:
                continue

            # 堆叠 + padding
            csi_hist = np.stack(csi_hist_list)  # [n_valid, T_hist, CSI_dim]
            attrs = np.stack(attr_list)  # [n_valid, AUX_DIM]
            mask = np.array(mask_list, dtype=np.float32)

            # Pad 到 n_max
            n_valid = len(csi_hist)
            if n_valid < self.n_max:
                pad_csi = np.zeros((self.n_max - n_valid, HISTORY_STEPS, len(H_COLS)))
                csi_hist = np.vstack([csi_hist, pad_csi])
                pad_attr = np.zeros((self.n_max - n_valid, AUX_DIM))
                attrs = np.vstack([attrs, pad_attr])
                pad_mask = np.zeros(self.n_max - n_valid, dtype=np.float32)
                mask = np.concatenate([mask, pad_mask])

            # 属性标准化
            if self.attr_mean is not None:
                attrs = (attrs - self.attr_mean) / (self.attr_std + 1e-12)

            self.samples.append(
                (
                    torch.tensor(csi_hist, dtype=torch.float32),
                    torch.tensor(attrs, dtype=torch.float32),
                    torch.tensor(y_csi, dtype=torch.float32),
                    torch.tensor(mask, dtype=torch.float32),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_csi_stats(self):
        return self.csi_mean, self.csi_std

    def get_attr_stats(self):
        return self.attr_mean, self.attr_std

    @staticmethod
    def compute_attr_stats_from_df(df):
        """从训练集 DataFrame 计算节点属性的 mean/std (用于归一化)。"""
        attrs = df[NODE_ATTR_COLS].values.astype(np.float64)
        return attrs.mean(axis=0), attrs.std(axis=0) + 1e-12
