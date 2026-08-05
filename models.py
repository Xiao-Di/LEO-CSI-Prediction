"""
模型定义: HybridCSIModel, VanillaLSTM, VanillaTransformer
"""

import torch
import torch.nn as nn
from config import NUM_ANTENNAS, D_MODEL, N_HEAD, FUTURE_STEPS, HISTORY_STEPS, NUM_LSTM_LAYERS, MACRO_TRANSFORMER_LAYERS, ORIG_AUX_DIM

CSI_DIM = 2 * NUM_ANTENNAS  # 实部+虚部拼接


class HybridCSIModel(nn.Module):
    """
    双尺度混合模型:
    - 微观路径 (LSTM): 捕捉短期相位漂移和小尺度多径干扰
    - 宏观路径 (Transformer): 捕捉轨道几何变化与环境演变
    """

    def __init__(self):
        super(HybridCSIModel, self).__init__()
        # 微观路径: LSTM 处理 CSI 序列
        self.micro_path = nn.LSTM(CSI_DIM, D_MODEL, num_layers=NUM_LSTM_LAYERS, batch_first=True)

        # 宏观路径: Transformer 处理辅助信息 [Distance_km, Doppler_Hz, Rain_Att_dB]
        self.macro_enc = nn.Linear(ORIG_AUX_DIM, D_MODEL)
        self.macro_path = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, batch_first=True),
            num_layers=MACRO_TRANSFORMER_LAYERS,
        )

        # 融合解码层
        self.fusion = nn.Linear(D_MODEL * 2, D_MODEL)
        self.decoder = nn.Linear(D_MODEL, FUTURE_STEPS * CSI_DIM)

    def forward(self, csi_seq, aux_info):
        _, (h_n, _) = self.micro_path(csi_seq)
        micro_feat = h_n[-1]  # 取最后时刻隐含状态
        macro_feat = torch.mean(self.macro_path(self.macro_enc(aux_info)), dim=1)
        fused = self.fusion(torch.cat((micro_feat, macro_feat), dim=1))
        return self.decoder(fused).view(-1, FUTURE_STEPS, CSI_DIM)


class VanillaLSTM(nn.Module):
    """基准模型: 纯数据驱动 LSTM，忽略辅助信息"""

    def __init__(self):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(CSI_DIM, D_MODEL, num_layers=NUM_LSTM_LAYERS, batch_first=True)
        self.decoder = nn.Linear(D_MODEL, FUTURE_STEPS * CSI_DIM)

    def forward(self, csi_seq, _aux):
        _, (h_n, _) = self.lstm(csi_seq)
        return self.decoder(h_n[-1]).view(-1, FUTURE_STEPS, CSI_DIM)


class VanillaTransformer(nn.Module):
    """基准模型: 纯 Transformer，忽略辅助信息"""

    def __init__(self):
        super(VanillaTransformer, self).__init__()
        self.enc = nn.Linear(CSI_DIM, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, HISTORY_STEPS, D_MODEL))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, batch_first=True),
            num_layers=MACRO_TRANSFORMER_LAYERS,
        )
        self.decoder = nn.Linear(D_MODEL, FUTURE_STEPS * CSI_DIM)

    def forward(self, csi_seq, _aux):
        x = self.enc(csi_seq) + self.pos
        feat = self.transformer(x)
        return self.decoder(torch.mean(feat, dim=1)).view(-1, FUTURE_STEPS, CSI_DIM)
