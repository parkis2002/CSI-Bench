"""
DisentangledCSI — Disentangled 1D-CNN student model for cross-domain CSI-HAR.

Architecture:
  SharedEncoder → [MotionBranch, DomainBranch]

SharedEncoder
  Lightweight stack of InstanceNorm1d Conv1d + depthwise-separable blocks.
  InstanceNorm1d is used throughout instead of BatchNorm to prevent cross-domain
  statistics contamination: each sample is normalised independently so running
  stats from one domain do not pollute another.

MotionBranch  (the main output)
  Depthwise-separable blocks + Squeeze-and-Excitation channel attention + GAP.
  Exposes backbone_dim (= motion_channels) and forward_features() so
  KnowledgeDistillationTrainer can be used directly as a drop-in; the
  DisentangledTrainer extends this with domain CE, adversarial, and orthogonal
  losses.

DomainBranch  (the noise absorber)
  Lighter depthwise-separable blocks + GAP → per-domain classifiers.
  Training it with domain CE pulls domain-specific variance into this branch,
  freeing the motion branch of it.

Disentanglement enforced by DisentangledTrainer:
  - Orthogonal regularisation  : motion_feat ⊥ domain_feat
  - Adversarial GRL            : separate domain discriminators on motion_feat
  - Domain CE                  : domain branch heads train normally

Input:  [B, 1, T, F] or [B, T, F]   (CSI-Bench standard 4-D or 3-D)
Output: [B, num_classes]             (via forward; or full tuple via forward_full)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _INConv1d(nn.Module):
    """Conv1d + InstanceNorm1d(affine) + GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _DSConv1d(nn.Module):
    """Depthwise-separable Conv1d + InstanceNorm1d + GELU + residual skip."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size,
                            padding=kernel_size // 2,
                            groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.norm(self.pw(self.dw(x)))))


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
        s = self.pool(x).squeeze(-1)
        e = self.fc(s).unsqueeze(-1)
        return x * e


# ---------------------------------------------------------------------------
# Branch modules
# ---------------------------------------------------------------------------

class _MotionBranch(nn.Module):
    """DepthwiseSepConv stack + SE + GAP → motion feature vector."""

    def __init__(self, in_ch: int, hidden_dim: int,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList([
            _DSConv1d(hidden_dim, kernel_size=5, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.se = _SEBlock1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] → [B, hidden_dim]
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.se(x)
        return self.pool(x).squeeze(-1)


class _DomainBranch(nn.Module):
    """Lighter DepthwiseSepConv stack + GAP → domain feature vector."""

    def __init__(self, in_ch: int, hidden_dim: int,
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList([
            _DSConv1d(hidden_dim, kernel_size=5, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] → [B, hidden_dim]
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.pool(x).squeeze(-1)


# ---------------------------------------------------------------------------
# DisentangledCSI
# ---------------------------------------------------------------------------

class DisentangledCSI(nn.Module):
    """
    Disentangled 1D-CNN student model for cross-domain CSI-HAR.

    Parameters
    ----------
    feature_size     : int   — number of CSI subcarrier channels (default 232)
    enc_channels     : int   — shared encoder output channels (default 64)
    motion_channels  : int   — motion branch hidden dimension (default 128)
    domain_channels  : int   — domain branch hidden dimension (default 64)
    enc_layers       : int   — number of shared encoder blocks (default 2)
    motion_blocks    : int   — number of DSConv blocks in motion branch (default 3)
    domain_blocks    : int   — number of DSConv blocks in domain branch (default 2)
    dropout          : float — dropout rate in DSConv blocks (default 0.1)
    num_classes      : int   — number of activity classes
    domain_n_classes : list  — [n_device, n_env, n_user]; empty list disables domain heads
    win_len          : int   — window length in timesteps (default 500)

    Attributes for KD-trainer compatibility
    ----------------------------------------
    backbone_dim     : int  — motion branch output dimension (= motion_channels)
    forward_features(x) → (logits, motion_feat [B, backbone_dim])
    """

    def __init__(
        self,
        feature_size: int = 232,
        enc_channels: int = 64,
        motion_channels: int = 128,
        domain_channels: int = 64,
        enc_layers: int = 2,
        motion_blocks: int = 3,
        domain_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int = 6,
        domain_n_classes: list = None,
        win_len: int = 500,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.domain_n_classes = domain_n_classes or []
        self.win_len = win_len

        # ---- Shared encoder -------------------------------------------------
        enc_blocks: list = [_INConv1d(feature_size, enc_channels, kernel_size=7)]
        for _ in range(enc_layers - 1):
            enc_blocks.append(_DSConv1d(enc_channels, kernel_size=5, dropout=dropout))
        self.encoder = nn.Sequential(*enc_blocks)

        # ---- Motion branch --------------------------------------------------
        self.motion_branch = _MotionBranch(enc_channels, motion_channels,
                                           motion_blocks, dropout)
        self.backbone_dim = motion_channels
        self.motion_head = nn.Linear(motion_channels, num_classes)

        # ---- Domain branch --------------------------------------------------
        self.domain_branch = _DomainBranch(enc_channels, domain_channels,
                                           domain_blocks, dropout)
        self.domain_heads = nn.ModuleList([
            nn.Linear(domain_channels, n) for n in self.domain_n_classes
        ])

    # ------------------------------------------------------------------
    # Init-params (for few-shot / checkpoint cloning)
    # ------------------------------------------------------------------

    def get_init_params(self) -> dict:
        return {
            'feature_size': self.feature_size,
            'num_classes': self.num_classes,
            'domain_n_classes': self.domain_n_classes,
            'win_len': self.win_len,
        }

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    def _to_bft(self, x: torch.Tensor) -> torch.Tensor:
        """Convert any CSI-Bench input shape to [B, F, T]."""
        if x.dim() == 4:
            x = x.squeeze(1)          # [B, 1, T, F] → [B, T, F]
        if x.shape[-1] == self.feature_size:
            x = x.transpose(1, 2)     # [B, T, F] → [B, F, T]
        return x

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_full(self, x: torch.Tensor):
        """Full forward pass exposing all branch outputs.

        Returns
        -------
        motion_logits : [B, num_classes]
        motion_feat   : [B, motion_channels]  — backbone_dim embedding
        domain_logits : list of [B, n_domain_i]  — one per domain head
        domain_feat   : [B, domain_channels]
        """
        x = self._to_bft(x)                           # [B, F, T]
        enc = self.encoder(x)                          # [B, enc_channels, T]

        motion_feat = self.motion_branch(enc)          # [B, motion_channels]
        motion_logits = self.motion_head(motion_feat)  # [B, num_classes]

        domain_feat = self.domain_branch(enc)          # [B, domain_channels]
        domain_logits = [head(domain_feat) for head in self.domain_heads]

        return motion_logits, motion_feat, domain_logits, domain_feat

    def forward_features(self, x: torch.Tensor):
        """KD-trainer interface: (logits, feat [B, backbone_dim])."""
        motion_logits, motion_feat, _, _ = self.forward_full(x)
        return motion_logits, motion_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits
