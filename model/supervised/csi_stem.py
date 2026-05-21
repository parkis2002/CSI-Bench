"""
Shared CSI input preprocessing modules for Methods 1–3.

Background
----------
CSI-Bench HAR training data contains two frequency bands:
  - 5G: 232 real subcarrier channels, amplitude scale ~1–5
  - 2G: 56 real subcarrier channels, zero-padded to 232 by the data loader

This means 71% of training batches have exactly 0.0 in subcarrier columns
56–231.  All three modules below are designed to handle this heterogeneity.

Method 2 — Subcarrier Validity Masking (in SubcarrierGroupStem)
    After group-averaging, any group whose energy is below a threshold
    (i.e. all constituent subcarriers were zero-padded) is gated to zero
    before projection.  This prevents the Linear projection's bias terms
    from contributing spurious activations for padded channels.

Method 3 — Subcarrier-Group Pooling Stem (SubcarrierGroupStem)
    Adjacent subcarriers are averaged into groups of `group_size` before
    the linear projection.  For feature_size=232, group_size=8 → 29 groups.
    This reduces the input projection from Linear(232, d) to Linear(29, d),
    ties weights across physically adjacent (correlated) subcarriers, and
    makes the 56/232 boundary land cleanly on group 7 (subcarriers 56–63
    are the first group that may be zero-padded for 2G samples).

Method 1 — FFT Frequency Branch (FFTBranch)
    Computes per-group magnitude spectra along the time axis and projects
    the low-frequency component (first `n_fft_keep` bins) into a compact
    feature vector appended before the classification head.  Activity
    frequencies: breathing ≈ 0.3 Hz, walking ≈ 1 Hz, running ≈ 2 Hz.
    For a 500-sample window at 50 Hz CSI rate, 32 bins covers 0–3.2 Hz.
"""

import torch
import torch.nn as nn


class SubcarrierGroupStem(nn.Module):
    """
    Methods 2 + 3: grouped subcarrier pooling with validity masking.

    Input:  [B, T, F]   where F = feature_size (232)
    Output: [B, T, d_model]

    Parameters
    ----------
    feature_size : number of subcarrier channels (must be divisible by group_size).
    group_size   : adjacent subcarriers averaged per group. Default 8 → 29 groups.
    d_model      : output feature dimension per timestep.
    dropout      : applied to the projected output.
    """

    def __init__(
        self,
        feature_size: int = 232,
        group_size: int = 8,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_size % group_size != 0:
            raise ValueError(
                f"feature_size ({feature_size}) must be divisible by group_size ({group_size})"
            )
        self.feature_size = feature_size
        self.group_size = group_size
        self.n_groups = feature_size // group_size  # 29 for 232/8

        self.proj = nn.Linear(self.n_groups, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape

        # Method 3: average each contiguous group of `group_size` subcarriers
        grouped = x.view(B, T, self.n_groups, self.group_size).mean(dim=-1)  # [B, T, n_groups]

        # Method 2: gate groups whose energy is negligible (zero-padded subcarriers)
        # energy: RMS over the time axis, one value per (sample, group)
        energy = grouped.detach().abs().mean(dim=1, keepdim=True)  # [B, 1, n_groups]
        gate = (energy > 1e-4).float()                             # hard, non-differentiable
        grouped = grouped * gate                                    # [B, T, n_groups]

        return self.drop(self.norm(self.proj(grouped)))             # [B, T, d_model]


class FFTBranch(nn.Module):
    """
    Method 1: frequency-domain branch producing a global frequency summary.

    Computes the magnitude spectrum of each subcarrier group along the time
    axis, retains the `n_fft_keep` low-frequency bins, and projects the
    result to a compact `d_fft`-dimensional vector.  The output is
    concatenated with the pooled temporal representation just before the
    classification head of each model.

    Input:  [B, T, F]  (same raw tensor passed to SubcarrierGroupStem)
    Output: [B, d_fft]

    Parameters
    ----------
    win_len     : temporal window length (500 for CSI-Bench).
    feature_size: number of subcarrier channels (232).
    group_size  : must match SubcarrierGroupStem.group_size (default 8).
    n_fft_keep  : low-frequency FFT bins to retain (default 32 covers 0–3.2 Hz
                  at a 50 Hz CSI sampling rate, capturing all activity rhythms).
    d_fft       : output dimension.
    dropout     : applied after the GELU activation.
    """

    def __init__(
        self,
        win_len: int = 500,
        feature_size: int = 232,
        group_size: int = 8,
        n_fft_keep: int = 32,
        d_fft: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_size % group_size != 0:
            raise ValueError(
                f"feature_size ({feature_size}) must be divisible by group_size ({group_size})"
            )
        self.group_size = group_size
        self.n_groups = feature_size // group_size
        self.n_fft_keep = min(n_fft_keep, win_len // 2 + 1)

        in_dim = self.n_fft_keep * self.n_groups  # 32 * 29 = 928
        self.proj = nn.Linear(in_dim, d_fft)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape

        # Group subcarriers (mirrors SubcarrierGroupStem)
        grouped = x.view(B, T, self.n_groups, self.group_size).mean(dim=-1)  # [B, T, n_groups]

        # Validity gate (prevents zero-padded groups from contributing FFT energy)
        energy = grouped.detach().abs().mean(dim=1, keepdim=True)  # [B, 1, n_groups]
        gate = (energy > 1e-4).float()
        grouped = grouped * gate                                    # [B, T, n_groups]

        # FFT along the time axis → magnitude spectrum
        fft_mag = torch.fft.rfft(grouped, dim=1).abs()             # [B, T//2+1, n_groups]

        # Retain low-frequency bins and flatten
        fft_feat = fft_mag[:, :self.n_fft_keep, :].reshape(B, -1)  # [B, n_fft_keep * n_groups]

        return self.norm(self.drop(self.act(self.proj(fft_feat))))  # [B, d_fft]
