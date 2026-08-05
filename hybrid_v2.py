"""
GNO-enhanced Hybrid CSI prediction model (HybridV2) — 多星并发版本。

架构:
1. 每颗卫星独立 LSTM 编码: [B, N, T_hist, CSI_dim] → [B, N, D]
2. GNO 卫星图消息传递: [B, N, D] → [B, N, T_hist, D]
3. 提取主星特征 (Sat_Index=0): [B, T_hist, D]
4. Macro-path: TransformerEncoder on 主星辅助特征 → [B, D]
5. Cross-attention + 解码 → [B, 4, 32]
"""

import torch
import torch.nn as nn
from config import (
    NUM_ANTENNAS, D_MODEL, N_HEAD, FUTURE_STEPS, HISTORY_STEPS,
    AUX_DIM, GNO_KERNEL_DIM, GNO_NUM_LAYERS, GNO_AUX_DIM, GNO_AUX_INDICES,
    NUM_LSTM_LAYERS, MACRO_TRANSFORMER_LAYERS, GNO_NODE_DIM,
)
from gno_module import GNOBlock

CSI_DIM = 2 * NUM_ANTENNAS


class HybridV2(nn.Module):
    """
    多星并发 GNO 模型。

    输入:
        csi_hist:   [B, N, T_hist, CSI_dim]  所有卫星历史 CSI
        node_attrs: [B, N, AUX_DIM]           所有卫星节点属性
        y_csi:      [B, T_fut, CSI_dim]       主星未来 CSI (训练时)
        mask:       [B, N]                    有效卫星掩码

    流程:
        1. 每颗卫星 LSTM 编码: [B, N, T_hist, CSI_dim] → [B, N, D]
        2. GNO 消息传递: [B, N, D] + [B, N, A] + [B, N] → [B, N, T_hist, D]
        3. 提取主星: [B, T_hist, D]
        4. Macro-path: Transformer on 主星辅助 → [B, D]
        5. Cross-attention + 解码 → [B, T_fut, CSI_dim]
    """

    def __init__(
        self,
        aux_dim=AUX_DIM,
        gno_aux_dim=GNO_AUX_DIM,
        d_model=D_MODEL,
        gno_hidden=GNO_KERNEL_DIM,
        gno_layers=GNO_NUM_LAYERS,
        n_head=N_HEAD,
        node_attr_dim=GNO_NODE_DIM,
    ):
        super().__init__()

        # 1. 每颗卫星的 LSTM 编码器 (共享权重)
        self.sat_lstm = nn.LSTM(
            CSI_DIM, d_model, num_layers=NUM_LSTM_LAYERS, batch_first=True
        )

        # 2. CSI 序列嵌入 (用于 GNO 输入)
        self.csi_enc = nn.Linear(CSI_DIM, d_model)
        self.csi_pos = nn.Parameter(torch.randn(1, 1, HISTORY_STEPS, d_model))

        # 3. GNO 卫星图消息传递
        self.gno = GNOBlock(
            gno_aux_dim, d_model,
            kernel_hidden=gno_hidden, num_layers=gno_layers,
        )

        # 4. Macro-path: TransformerEncoder on 主星辅助特征
        self.macro_enc = nn.Linear(aux_dim, d_model)
        self.macro_path = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head, batch_first=True),
            num_layers=MACRO_TRANSFORMER_LAYERS,
        )

        # 5. Cross-attention: GNO 主星特征 query macro
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # 6. 融合 + 解码
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, FUTURE_STEPS * CSI_DIM),
        )

    def forward(self, csi_hist, node_attrs, mask):
        """
        csi_hist:   [B, N, T_hist, CSI_dim]
        node_attrs: [B, N, AUX_DIM]
        mask:       [B, N]  (1=valid, 0=pad)
        Returns:    [B, T_fut, CSI_dim]
        """
        B, N, T, _ = csi_hist.shape

        # 1. 每颗卫星独立 LSTM 编码
        csi_flat = csi_hist.reshape(B * N, T, CSI_DIM)  # [B*N, T, CSI_dim]
        _, (h_n, _) = self.sat_lstm(csi_flat)
        sat_feat = h_n[-1].reshape(B, N, -1)  # [B, N, D]

        # 2. CSI 序列编码 + 位置编码 (用于 GNO)
        csi_embed = self.csi_enc(csi_hist) + self.csi_pos  # [B, N, T, D]

        # 3. GNO 节点属性提取 (排除 Rain_Att_dB)
        gno_attrs = node_attrs[:, :, GNO_AUX_INDICES]  # [B, N, GNO_AUX_DIM]

        # GNO 消息传递
        gno_out = self.gno(gno_attrs, csi_embed, mask)  # [B, N, T, D]
        gno_feat = gno_out.mean(dim=2)  # [B, N, D] 时间维度平均池化

        # 4. 提取主星 (Sat_Index=0, 即第一个有效节点)
        # mask 确保我们取的是有效的主星
        primary_mask = mask[:, 0:1].unsqueeze(2)  # [B, 1, 1]
        primary_feat = gno_feat[:, 0:1, :]  # [B, 1, D]
        primary_feat = primary_feat.squeeze(1)  # [B, D]

        # 主星序列 (用于 macro-path)
        primary_seq = csi_embed[:, 0, :, :]  # [B, T, D]

        # 5. Macro-path: Transformer on 主星辅助特征
        # 取主星的辅助属性 (从 node_attrs 扩展为序列)
        primary_attrs = node_attrs[:, 0, :]  # [B, AUX_DIM]
        # 扩展为序列 (每步相同)
        primary_attrs_seq = primary_attrs.unsqueeze(1).expand(B, T, -1)  # [B, T, AUX_DIM]
        macro_enc = self.macro_enc(primary_attrs_seq)  # [B, T, D]
        macro_out = self.macro_path(macro_enc)
        macro_feat = macro_out.mean(dim=1)  # [B, D]

        # 6. Cross-attention: 主星 GNO 特征 query macro
        ca_out, _ = self.cross_attn(
            primary_feat.unsqueeze(1),  # query [B, 1, D]
            macro_out,                    # key   [B, T, D]
            macro_out,                    # value [B, T, D]
        )
        ca_out = self.cross_attn_norm(ca_out.squeeze(1))  # [B, D]

        # 7. 融合 + 解码
        fused = self.fusion(torch.cat([primary_feat, ca_out], dim=-1))  # [B, D]
        return self.decoder(fused).view(B, FUTURE_STEPS, CSI_DIM)
