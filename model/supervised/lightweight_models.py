"""
Lightweight CSI-HAR models for comparison against PrunedAttentionGRU.

Two architectures:

  LightTCN  (improved)
    Temporal Convolutional Network with exponentially dilated depthwise
    separable convolutions and Squeeze-and-Excitation channel attention.
    Improvements over v1:
      - kernel_size 3→5: wider per-block receptive field
      - tcn_layers 3→5: RF = 1 + 4*(1+2+4+8+16) = 125 steps (25 % of window)
      - tcn_channels 64→128: more representational capacity per timestep
      - Global average pool replaced by learnable temporal attention pool:
        a linear layer scores each timestep and the weighted sum is taken,
        allowing the model to focus on the sharpest discriminative moment
        (e.g. jump apex, wave onset) rather than blurring it with the mean.
    Inspirations:
      - TCN: Bai et al., "An Empirical Evaluation of Generic Convolutional and
             Recurrent Networks for Sequence Modeling" (2018)
      - SE-Net: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
      - Depthwise separable conv: Howard et al., MobileNets (2017)

  InceptionCSI  (improved)
    InceptionTime-style multi-scale CNN.
    Improvements over v1:
      - Kernel sizes (5,11,23)→(9,19,39): larger multi-scale receptive fields
        better suited to the 500-step window (39 steps ≈ 8 % of window)
      - depth 3→6: matches the InceptionTime paper and builds a deeper
        temporal hierarchy, allowing second-order pattern learning
      - Per-3-block residual shortcuts: residual is applied every 3 blocks
        (matching InceptionTime's original design) rather than one global
        shortcut; intermediate shortcuts stabilise gradient flow through deep
        stacks and reduce the spiky validation loss seen at depth 3
    Inspiration:
      - InceptionTime: Fawaz et al., "InceptionTime: Finding AlexNet for Time
                       Series Classification" (Data Mining and Knowledge
                       Discovery, 2020)

Both accept CSI-Bench 4-D input [B, 1, win_len, feature_size] and output
[B, num_classes].  Both implement get_init_params() for few-shot cloning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.supervised.csi_stem import FFTBranch, SubcarrierGroupStem


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

class _TemporalAttentionPool(nn.Module):
    """Soft attention pooling over the time axis.

    A linear layer scores each timestep; the softmax-normalised weights are
    used to compute a weighted sum that collapses the T dimension.
    Compared with global average pooling, this lets the model focus on the
    single sharpest discriminative moment in the sequence — essential for
    impulsive activities (jumping peak, wave onset) that avg-pool smooths out.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        e = self.score(x.transpose(1, 2))       # [B, T, 1]
        w = torch.softmax(e, dim=1)             # [B, T, 1]
        return (x.transpose(1, 2) * w).sum(1)  # [B, C]


# ---------------------------------------------------------------------------
# LightTCN components
# ---------------------------------------------------------------------------

class _SEBlock1d(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        s = self.pool(x).squeeze(-1)    # [B, C]
        e = self.fc(s).unsqueeze(-1)    # [B, C, 1]
        return x * e


class _TCNBlock(nn.Module):
    """Dilated depthwise-separable conv + SE + residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        se_reduction: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dw = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=padding,
            groups=channels, bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.se = _SEBlock1d(channels, se_reduction)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.pw(self.dw(x))))
        out = self.se(out)
        out = self.drop(out)
        return self.relu(out + x)


# ---------------------------------------------------------------------------
# LightTCN
# ---------------------------------------------------------------------------

class LightTCN(nn.Module):
    """
    Lightweight Temporal Convolutional Network for CSI-HAR.

    Treat the feature (subcarrier) axis as channels and apply 1-D dilated
    depthwise-separable convolutions along the time axis.  Exponentially
    growing dilation doubles the temporal receptive field per block without
    adding parameters.  Squeeze-and-Excitation re-weights each channel after
    each block, focusing on the most informative subcarrier groups.
    Temporal attention pooling replaces global average pooling, allowing the
    model to focus on the single most discriminative time window.

    Input:  [B, 1, win_len, feature_size]
    Output: [B, num_classes]

    Default parameter count: ~310 K (tcn_channels=128, tcn_layers=5, kernel=5)
    Receptive field: 1 + 4*(1+2+4+8+16) = 125 steps (25 % of 500-step window)
    """

    def __init__(
        self,
        feature_size: int = 232,
        tcn_channels: int = 128,
        tcn_layers: int = 5,
        se_reduction: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
        win_len: int = 500,
        d_fft: int = 32,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.tcn_channels = tcn_channels
        self.tcn_layers = tcn_layers
        self.num_classes = num_classes
        self.win_len = win_len
        self.d_fft = d_fft

        # Method 3 + 2: grouped subcarrier stem replaces Conv1d(F→C) input projection
        self.stem = SubcarrierGroupStem(feature_size, group_size=8,
                                        d_model=tcn_channels, dropout=dropout)

        # Method 1: FFT frequency branch
        self.fft_branch = FFTBranch(win_len=win_len, feature_size=feature_size,
                                     group_size=8, d_fft=d_fft, dropout=dropout)

        # Kernel size 5, dilations 1, 2, 4, 8, 16 → RF = 125 steps at 5 layers
        self.blocks = nn.ModuleList([
            _TCNBlock(
                tcn_channels,
                kernel_size=5,
                dilation=2 ** i,
                se_reduction=se_reduction,
                dropout=dropout,
            )
            for i in range(tcn_layers)
        ])

        # Attention pool: focuses on the sharpest discriminative moment
        self.pool = _TemporalAttentionPool(tcn_channels)
        self.backbone_dim = tcn_channels + d_fft
        self.head = nn.Linear(self.backbone_dim, num_classes)

    def get_init_params(self) -> dict:
        return {
            "feature_size": self.feature_size,
            "tcn_channels": self.tcn_channels,
            "tcn_layers": self.tcn_layers,
            "num_classes": self.num_classes,
            "win_len": self.win_len,
            "d_fft": self.d_fft,
        }

    def forward_features(self, x: torch.Tensor):
        """Return (logits, feat) where feat [B, backbone_dim] is the pre-head embedding."""
        if x.dim() == 4:
            x = x.squeeze(1)                   # [B, T, F]
        if x.shape[-1] != self.feature_size:
            x = x.transpose(1, 2)             # [B, T, F]

        fft_feat = self.fft_branch(x)          # [B, d_fft]

        x = self.stem(x)                       # [B, T, C]
        x = x.transpose(1, 2)                  # [B, C, T]

        for block in self.blocks:
            x = block(x)                       # [B, C, T]
        x = self.pool(x)                       # [B, C]

        feat = torch.cat([x, fft_feat], dim=-1)   # [B, backbone_dim]
        return self.head(feat), feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits


# ---------------------------------------------------------------------------
# InceptionCSI components
# ---------------------------------------------------------------------------

class _InceptionBlock1d(nn.Module):
    """
    Multi-scale inception module for 1-D sequences.

    Four parallel branches are computed and concatenated:
      - Three convolution branches (after a shared bottleneck) at kernel
        sizes 9, 19, and 39 to capture short-, medium-, and long-range
        temporal patterns (larger kernels than v1 suit the 500-step window).
      - One max-pool + pointwise branch as a skip connection that preserves
        high-frequency information without parameter cost.
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 24,
        kernels: tuple = (9, 19, 39),
        bottleneck_size: int = 24,
    ):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, 1, bias=False)

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(bottleneck_size, nb_filters, k, padding=k // 2, bias=False)
            for k in kernels
        ])

        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, 1, bias=False),
        )

        n_out = nb_filters * (len(kernels) + 1)
        self.bn = nn.BatchNorm1d(n_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, in_ch, T]
        bn = self.bottleneck(x)
        outs = [b(bn) for b in self.conv_branches]
        outs.append(self.pool_branch(x))
        return self.relu(self.bn(torch.cat(outs, dim=1)))  # [B, n_out, T]


# ---------------------------------------------------------------------------
# InceptionCSI
# ---------------------------------------------------------------------------

class InceptionCSI(nn.Module):
    """
    InceptionTime-inspired multi-scale 1-D CNN for CSI-HAR.

    The feature axis is treated as channels; InceptionBlocks apply parallel
    convolutions at temporal scales of 9, 19, and 39 timesteps.  Residual
    shortcuts are applied every 3 blocks (matching InceptionTime's design),
    providing gradient highways through the deeper 6-block stack.

    Input:  [B, 1, win_len, feature_size]
    Output: [B, num_classes]

    Default parameter count: ~415 K (nb_filters=24, depth=6)
    """

    _KERNELS = (9, 19, 39)

    def __init__(
        self,
        feature_size: int = 232,
        nb_filters: int = 24,
        depth: int = 6,
        dropout: float = 0.1,
        num_classes: int = 2,
        win_len: int = 500,
        d_fft: int = 32,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.nb_filters = nb_filters
        self.depth = depth
        self.num_classes = num_classes
        self.win_len = win_len
        self.d_fft = d_fft

        out_ch = nb_filters * (len(self._KERNELS) + 1)  # 4 × nb_filters = 96

        # Method 3 + 2: grouped subcarrier stem replaces Conv1d(F→out_ch)
        self.stem = SubcarrierGroupStem(feature_size, group_size=8,
                                        d_model=out_ch, dropout=dropout)

        # Method 1: FFT frequency branch
        self.fft_branch = FFTBranch(win_len=win_len, feature_size=feature_size,
                                     group_size=8, d_fft=d_fft, dropout=dropout)

        self.blocks = nn.ModuleList([
            _InceptionBlock1d(out_ch, nb_filters, self._KERNELS, nb_filters)
            for _ in range(depth)
        ])

        # One residual BN per group of 3 blocks (e.g. depth=6 → 2 shortcuts)
        n_shortcuts = depth // 3
        self.res_bns = nn.ModuleList([
            nn.BatchNorm1d(out_ch) for _ in range(n_shortcuts)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.backbone_dim = out_ch + d_fft
        self.head = nn.Linear(self.backbone_dim, num_classes)

    def get_init_params(self) -> dict:
        return {
            "feature_size": self.feature_size,
            "nb_filters": self.nb_filters,
            "depth": self.depth,
            "num_classes": self.num_classes,
            "win_len": self.win_len,
            "d_fft": self.d_fft,
        }

    def forward_features(self, x: torch.Tensor):
        """Return (logits, feat) where feat [B, backbone_dim] is the pre-head embedding."""
        if x.dim() == 4:
            x = x.squeeze(1)                   # [B, T, F]
        if x.shape[-1] != self.feature_size:
            x = x.transpose(1, 2)             # [B, T, F]

        fft_feat = self.fft_branch(x)          # [B, d_fft]

        x = self.stem(x)                       # [B, T, out_ch]
        x = x.transpose(1, 2)                  # [B, out_ch, T]

        residual = x
        shortcut_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % 3 == 0 and shortcut_idx < len(self.res_bns):
                x = F.relu(self.res_bns[shortcut_idx](x + residual))
                residual = x
                shortcut_idx += 1

        x = self.pool(x).squeeze(-1)           # [B, out_ch]
        x = self.drop(x)
        feat = torch.cat([x, fft_feat], dim=-1)   # [B, backbone_dim]
        return self.head(feat), feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits
