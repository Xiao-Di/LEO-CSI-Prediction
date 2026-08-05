"""
全局仿真参数配置
参考论文: Parameter Settings for LEO Satellite Communication System (Table I)
"""

import numpy as np

# --- 系统参数 (与论文 Table I 严格一致) ---
CARRIER_FREQ = 5e9  # 载波频率 5 GHz
LIGHT_SPEED = 3e8  # 光速 m/s
SATELLITE_HEIGHT = 600e3  # 卫星轨道高度 600 km (m)
SATELLITE_VEL = 7500  # 卫星运行速度 7.5 km/s (m/s)
NUM_ANTENNAS = 16  # 天线数 N
NUM_PATHS = 6  # 散射路径数 L_k

# --- 时序参数 (与论文 Table I 严格一致) ---
HISTORY_STEPS = 16  # 历史观测步长 T_P
FUTURE_STEPS = 4  # 预测步长 T_F
DT = 0.5e-3  # 时隙间隔 0.5 ms
# 说明: DT 是预测任务的"时隙间隔"，不是载波采样率。
# CSI 数据为基带观测（载波同步后），相位变化来自残差多普勒。
# 5 GHz 下最大多普勒 ~125 kHz，残差多普勒约 0.1%~1% (125~1250 Hz)。
# 取 RESIDUAL_VEL_FACTOR=0.005，残差 fd ≈ 625 Hz，
# 每步相位变化 Δφ = 2π×625×0.5ms ≈ 1.96 rad (112.5°)，数值稳定。

K_FACTOR = 10  # Rician K 因子 (dB)

# --- 残差多普勒比例 (基带观测) ---
# 接收机载波同步后剩余的相位漂移占设备多普勒的比例
# 取 10% 以保证相位演化在时间窗口内可被模型学习
# Speed=10: fd_dev≈35 Hz → residual≈3.5 Hz → Δφ/step≈0.6° (16步累计≈10°)
# Speed=100: fd_dev≈350 Hz → residual≈35 Hz → Δφ/step≈6.3° (16步累计≈101°)
RESIDUAL_VEL_FACTOR = 0.10  # 10%

# --- 训练超参数 ---
D_MODEL = 32  # 统一隐藏层维度（减小以防过拟合）
N_HEAD = 4  # Transformer 注意力头数
BATCH_SIZE = 64
LR_OFFLINE = 1e-4  # 离线预训练学习率
LR_MAML = 1e-5  # MAML 在线更新学习率
EPOCHS = 100

# --- 物理损失权重 ---
LAMBDA_NMSE = 1.0  # NMSE 回归项权重
LAMBDA_PHYS = 0.5  # 物理一致性正则项权重（GNO-Hybrid 使用增强权重）

# --- Walker Constellation ---
NUM_PLANES = 6              # Walker Delta constellation planes
SATS_PER_PLANE = 4          # Satellites per plane
INCLINATION = np.radians(53)  # Orbital inclination 53°
MIN_ELEVATION_VIS = np.radians(5)  # Min visible satellite elevation 5°
MAX_SATELLITES_PER_SAMPLE = 4  # Max co-visible satellites per sample (padding size)
EARTH_RADIUS = 6371e3       # Earth radius (m)
EARTH_MU = 3.986e14         # Earth gravitational parameter (m³/s²)

# Ground terminal location ranges (urban)
TERMINAL_LAT_MIN = 20       # degrees
TERMINAL_LAT_MAX = 55
TERMINAL_LON_MIN = 0
TERMINAL_LON_MAX = 360

# --- GNO (per-sample satellite graph) ---
AUX_DIM = 7              # Per-satellite aux: Dist, Doppler, Rain, cos_el, dir_cos, az, el
GNO_NODE_DIM = 5         # Per-node features for GNO: Dist, Doppler, cos_el, dir_cos, elevation
ORIG_AUX_DIM = 5         # Dataset aux columns: Dist, Doppler, Rain, cos_el, dir_cos
GNO_AUX_DIM = 4          # GNO kernel features: Dist, Doppler, cos_el, dir_cos
GNO_AUX_INDICES = [0, 1, 3, 4]  # Indices in AUX_DIM vector to use for GNO kernel
GNO_KERNEL_DIM = 16      # GNO kernel hidden dim（减小以防过拟合）
GNO_NUM_LAYERS = 2       # GNO layers

# --- Node attribute normalization ---
NODE_ATTR_CLIP = 1e3     # Clip distance/etc for numerical stability before normalization

# --- FreqMamba-CSI ---
FREQ_MAMBA_D_MODEL = 48
FREQ_MAMBA_N_HEAD = 4
FREQ_MAMBA_MAMBA_BLOCKS = 2
FREQ_MAMBA_D_STATE = 16
FREQ_MAMBA_D_CONV = 4
FREQ_MAMBA_EXPAND = 2
LAMBDA_SPEC = 0.3          # Spectral consistency loss weight
LAMBDA_REG_MAML = 0.01     # MAML++ parameter regularization weight

# --- 模型架构参数 ---
NUM_LSTM_LAYERS = 2      # LSTM/TransformerEncoder 层数（减小以防过拟合）
MACRO_TRANSFORMER_LAYERS = 2  # Macro-path TransformerEncoder 层数

# --- 学习率调度参数 ---
LR_SCHEDULER_FACTOR = 0.5       # 学习率衰减因子
LR_SCHEDULER_PATIENCE = 10      # 学习率衰减 patience
LR_MIN = 1e-6                   # 最小学习率

# --- 早停参数 ---
EARLY_STOP_PATIENCE = 15        # 早停 patience（验证 loss 无改善的轮数）

# --- 运行环境 ---
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
