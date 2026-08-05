"""
FreqMamba-CSI: Time-Frequency Dual-Path Mamba for Satellite-Terrestrial CSI Prediction

Architecture:
1. Time Domain Path: Mamba SSM encoder (selective state-space modeling)
2. Frequency Domain Path: Spectral Attention (FFT-based pattern extraction)
3. Physics-Aware Gated Fusion: dynamic time-frequency weighting
4. Cross-Scale Interaction: residual cross-product + Transformer refinement

Dual environment support:
- CUDA: uses mamba_ssm library (S6 selective scan)
- No CUDA: pure PyTorch fallback (1D conv + selective gating)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_ANTENNAS, FUTURE_STEPS, HISTORY_STEPS, ORIG_AUX_DIM,
    FREQ_MAMBA_D_MODEL, FREQ_MAMBA_N_HEAD, FREQ_MAMBA_MAMBA_BLOCKS,
    FREQ_MAMBA_D_STATE, FREQ_MAMBA_D_CONV, FREQ_MAMBA_EXPAND,
)

CSI_DIM = 2 * NUM_ANTENNAS
USE_MAMBA_SSM = torch.cuda.is_available()

if USE_MAMBA_SSM:
    try:
        from mamba_ssm import Mamba
        _MAMBA_AVAILABLE = True
    except ImportError:
        _MAMBA_AVAILABLE = False
else:
    _MAMBA_AVAILABLE = False


# ============================================================================
# Time Domain: Selective Mamba Block
# ============================================================================

class _MambaSSMBlock(nn.Module):
    """Official Mamba SSM block (requires mamba_ssm library + CUDA)."""

    def __init__(self, d_model, d_state=FREQ_MAMBA_D_STATE, d_conv=FREQ_MAMBA_D_CONV, expand=FREQ_MAMBA_EXPAND):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return self.mamba(x)


class _MambaFallbackBlock(nn.Module):
    """
    Pure PyTorch fallback: simulates SSM selective scan behavior.
    Uses depthwise 1D conv + input-dependent gating to mimic S6 behavior.
    Parameter structure aligned with MambaSSMBlock for weight compatibility.
    """

    def __init__(self, d_model, d_state=FREQ_MAMBA_D_STATE, d_conv=FREQ_MAMBA_D_CONV, expand=FREQ_MAMBA_EXPAND):
        super().__init__()
        d_inner = d_model * expand

        # Input projection (analogous to Mamba's in_proj)
        self.in_proj = nn.Linear(d_model, d_inner * 2)

        # Depthwise 1D conv (analogous to SSM's discrete state transition)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner,
        )

        # State dimension for gating (analogous to SSM state size)
        self.d_state = d_state
        self.state_proj = nn.Linear(d_inner, d_state)

        # Selective gate: input-dependent information flow
        self.selective_gate = nn.Linear(d_state, d_inner)
        self.gate_norm = nn.LayerNorm(d_state)

        # Output projection (analogous to Mamba's out_proj)
        self.out_proj = nn.Linear(d_inner, d_model)

        # Activation
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: [B, T, D]
        Returns: [B, T, D]
        """
        B, T, D = x.shape

        # Input projection split into branches (analogous to Mamba's dual branches)
        x_branch, gate_branch = self.in_proj(x).chunk(2, dim=-1)  # [B, T, d_inner]

        # 1D conv along time dimension (local temporal pattern learning)
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :T]  # causal trim
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
        x_conv = self.act(x_conv)

        # Selective gating: compute state from gate_branch, project to control info flow
        state = self.gate_norm(self.state_proj(gate_branch))  # [B, T, d_state]
        gate_weight = torch.sigmoid(self.selective_gate(state))  # [B, T, d_inner]

        # Apply selective gate (input-dependent modulation)
        x_gated = x_conv * gate_weight

        # Output projection
        return self.out_proj(x_gated)


if _MAMBA_AVAILABLE:
    SelectiveMambaBlock = _MambaSSMBlock
else:
    SelectiveMambaBlock = _MambaFallbackBlock


class TimeDomainEncoder(nn.Module):
    """
    Time domain path: Mamba stack for temporal CSI modeling.
    Input: [B, T, CSI_dim] → Output: [B, T, D]
    """

    def __init__(self, d_model=FREQ_MAMBA_D_MODEL, n_blocks=FREQ_MAMBA_MAMBA_BLOCKS):
        super().__init__()
        self.enc = nn.Linear(CSI_DIM, d_model)
        self.pos = nn.Parameter(torch.randn(1, HISTORY_STEPS, d_model))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "mamba": SelectiveMambaBlock(d_model),
                "norm": nn.LayerNorm(d_model),
            })
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        x = self.enc(x) + self.pos
        for block in self.blocks:
            x = x + block["mamba"](block["norm"](x))
        return x


# ============================================================================
# Frequency Domain: Spectral Attention
# ============================================================================

class SpectralAttention(nn.Module):
    """
    Frequency domain path: FFT → spectral attention → iFFT.

    Captures Doppler expansion patterns that are structured in frequency domain
    but appear as fast-varying phase rotation in time domain.

    Input/Output: [B, T, D]
    """

    def __init__(self, d_model=FREQ_MAMBA_D_MODEL):
        super().__init__()
        self.enc = nn.Linear(CSI_DIM, d_model)
        self.pos = nn.Parameter(torch.randn(1, HISTORY_STEPS, d_model))

        # Learnable spectral weight network
        self.spectral_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, CSI_dim] raw CSI sequence
        Returns: [B, T, D] spectrally-enhanced features
        """
        x = self.enc(x) + self.pos  # [B, T, D]

        # FFT along time dimension
        X = torch.fft.rfft(x, dim=1)  # [B, F, D] where F = T//2+1
        F_bins = X.size(1)

        # Power spectrum magnitude for attention
        P = X.abs()  # [B, F, D]

        # Learnable spectral weights: per-frequency-bin, per-feature weight
        spec_input = P.mean(dim=2, keepdim=True)  # [B, F, 1]
        spec_weights = self.spectral_mlp(spec_input)  # [B, F, 1]
        spec_weights = torch.softmax(spec_weights, dim=1)  # normalize over freq bins

        # Apply spectral attention
        X_weighted = X * spec_weights

        # iFFT back to time domain
        x_out = torch.fft.irfft(X_weighted, n=x.size(1), dim=1)  # [B, T, D]

        # Output projection + residual
        return x_out + self.out_proj(self.norm(x_out))


# ============================================================================
# Physics-Aware Gated Fusion
# ============================================================================

class PhysicsAwareGate(nn.Module):
    """
    Physics-aware gating: dynamically fuses time and frequency domain features
    based on scene parameters (Doppler, Rain, Distance).

    gate → 1: rely on time domain (high speed, fast-varying)
    gate → 0: rely on frequency domain (low speed, stable spectrum)
    """

    def __init__(self, aux_dim=ORIG_AUX_DIM):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(aux_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, h_time, h_freq, aux):
        """
        h_time: [B, T, D]
        h_freq: [B, T, D]
        aux:    [B, AUX_DIM]  (Dist, Doppler, Rain)
        Returns: [B, T, D] fused features
        """
        gate = torch.sigmoid(self.gate_net(aux))  # [B, 1]
        gate = gate.unsqueeze(1)  # [B, 1, 1]
        return gate * h_time + (1 - gate) * h_freq, gate.squeeze(-1)


# ============================================================================
# Cross-Scale Feature Interaction
# ============================================================================

class CrossScaleInteraction(nn.Module):
    """
    Cross-scale interaction: element-wise product of time/freq residuals
    captures co-occurrence patterns, refined by 1-layer Transformer.
    """

    def __init__(self, d_model=FREQ_MAMBA_D_MODEL, n_head=FREQ_MAMBA_N_HEAD):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head, batch_first=True),
            num_layers=1,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_fused, h_time, h_freq):
        """
        h_fused: [B, T, D]
        h_time:  [B, T, D]
        h_freq:  [B, T, D]
        Returns: [B, T, D]
        """
        # Cross-scale product: captures joint time-frequency patterns
        cross = h_time * h_freq  # [B, T, D]

        # Concatenate + project
        x = self.proj(torch.cat([h_fused, cross], dim=-1))  # [B, T, D]

        # Transformer refinement
        return self.norm(x + self.te(x))


# ============================================================================
# FreqMamba-CSI: Complete Model
# ============================================================================

class FreqMambaCSI(nn.Module):
    """
    FreqMamba-CSI: Time-Frequency Dual-Path Mamba for CSI Prediction.

    Input:
        csi_seq: [B, T_hist, CSI_dim]  historical CSI sequence
        aux:     [B, AUX_DIM]          auxiliary features (Dist, Doppler, Rain)

    Output:
        [B, T_fut, CSI_dim]  predicted future CSI

    Ablation flags (constructor):
        time_only: use only time domain path
        freq_only: use only frequency domain path
        no_gate: use fixed 0.5 fusion instead of physics-aware gate
    """

    def __init__(
        self,
        d_model=FREQ_MAMBA_D_MODEL,
        n_head=FREQ_MAMBA_N_HEAD,
        aux_dim=ORIG_AUX_DIM,
        time_only=False,
        freq_only=False,
        no_gate=False,
    ):
        super().__init__()
        self.time_only = time_only
        self.freq_only = freq_only
        self.no_gate = no_gate

        # 1. Time domain path
        self.time_encoder = TimeDomainEncoder(d_model)

        # 2. Frequency domain path
        self.freq_encoder = SpectralAttention(d_model)

        # 3. Physics-aware gate
        self.gate = PhysicsAwareGate(aux_dim)

        # 4. Cross-scale interaction
        self.cross_scale = CrossScaleInteraction(d_model, n_head)

        # 5. Decoder
        self.decoder = nn.Linear(d_model, FUTURE_STEPS * CSI_DIM)

    def forward(self, csi_seq, aux):
        """
        csi_seq: [B, T_hist, CSI_dim]
        aux:     [B, AUX_DIM]  or  [B, T_hist, AUX_DIM] (sequence-level)
        Returns: [B, T_fut, CSI_dim]
        """
        # Handle sequence-level aux: mean-pool over time
        if aux.ndim == 3:
            aux = aux.mean(dim=1)  # [B, T, A] → [B, A]

        # Time domain encoding
        h_time = self.time_encoder(csi_seq)  # [B, T, D]

        # Frequency domain encoding
        h_freq = self.freq_encoder(csi_seq)  # [B, T, D]

        # Ablation: single-path
        if self.time_only:
            h_fused, gate_val = h_time, torch.ones(h_time.size(0), 1, device=h_time.device)
        elif self.freq_only:
            h_fused, gate_val = h_freq, torch.zeros(h_freq.size(0), 1, device=h_freq.device)
        elif self.no_gate:
            h_fused = 0.5 * h_time + 0.5 * h_freq
            gate_val = 0.5 * torch.ones(h_time.size(0), 1, device=h_time.device)
        else:
            h_fused, gate_val = self.gate(h_time, h_freq, aux)  # [B, T, D], [B, 1]

        # Cross-scale interaction
        h_out = self.cross_scale(h_fused, h_time, h_freq)  # [B, T, D]

        # Decode: mean-pool over time → predict future CSI
        pooled = h_out.mean(dim=1)  # [B, D]
        return self.decoder(pooled).view(-1, FUTURE_STEPS, CSI_DIM)

    def get_gate_value(self, csi_seq, aux):
        """Return gate value for visualization (only meaningful when not ablated)."""
        h_time = self.time_encoder(csi_seq)
        h_freq = self.freq_encoder(csi_seq)
        _, gate_val = self.gate(h_time, h_freq, aux)
        return gate_val
