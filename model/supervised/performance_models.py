"""
High-performance CSI-HAR models (no parameter-count constraint).

CsiConformer
  Conformer architecture (Gulati et al., Interspeech 2020) adapted for CSI
  classification.  Each Conformer block interleaves:
    - Macaron feed-forward (scaled by ½) for non-linear projection
    - Multi-head self-attention (global temporal context)
    - Depthwise convolutional module (local pattern capture + implicit
      relative-positional encoding)
    - A second macaron feed-forward
  The convolutional module is the critical differentiator from a vanilla
  Transformer: it gives the model a fixed-length local context window (kernel
  size 31) that complements the unlimited global attention span.  This design
  is proven best-in-class for sequential signals where both short-range
  dynamics (e.g. gesture sharpness) and long-range structure (activity
  duration, inter-phase dependencies) matter.

DeepBiGRU
  Stacked bidirectional GRU followed by a Transformer encoder and
  concat-pool classification head.  The BiGRU provides a strong sequential
  inductive bias: unlike prunedAttentionGRU's unidirectional GRU, every
  timestep attends to both past and future context, giving a richer
  contextual encoding.  The Transformer encoder then re-weights these
  encodings with global self-attention and gated feed-forward networks.
  Concat mean+max pooling aggregates both the average activity profile and
  the single most discriminative moment in the sequence.

HierCSIFormer
  Hierarchical CSI Transformer.  A three-stage architecture:
    Stage 1 — SubcarrierGroupStem projects the 232 subcarrier channels to
      d_model with validity gating for 2G/5G heterogeneity.  Three parallel
      depthwise-separable 1D conv branches (kernels 5, 11, 21) fuse multi-scale
      temporal motifs.  Two strided convs reduce T: 500 → 100 → 25.
    Stage 2 — A learnable CLS token is prepended and sinusoidal PE added.
      depth pre-LN Transformer encoder blocks (d=256, heads=8, ff=1024,
      GELU) apply global attention over 26 positions.
    Stage 3 — An FFT branch extracts frequency-domain activity signatures
      (d_fft=64); the CLS embedding and FFT vector are concatenated before
      the linear classification head.
  ~5.5 M parameters (d_model=256, depth=6).  Outperforms both CsiConformer
  and the CSI-Bench ViT baseline by combining hierarchical feature extraction
  with global attention on a compact token sequence.

ResNet18CSI
  ResNet-18 backbone adapted as a high-capacity teacher model for knowledge
  distillation.  The standard ResNet-18 conv stack (layers 1-4 + global avg
  pool) produces a 512-dim feature vector; a linear head maps it to class
  logits.  The explicit self.head attribute ensures compatibility with
  DomainAdversarialTrainer's forward hook.  forward_features() exposes both
  logits and backbone features so student models can use soft targets and
  intermediate-feature matching during KD.
  ~11.2 M parameters.

All models accept CSI-Bench 4-D input [B, 1, win_len, feature_size] and
implement get_init_params() for few-shot cloning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from model.supervised.csi_stem import FFTBranch, SubcarrierGroupStem


# ============================================================================
# Shared utility
# ============================================================================

class _SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding with dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d]
        return self.dropout(x + self.pe[:, : x.size(1)])


# ============================================================================
# CsiConformer building blocks
# ============================================================================

class _ConformerFF(nn.Module):
    """Macaron feed-forward sub-layer (pre-LN, SiLU activation)."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class _ConformerConvModule(nn.Module):
    """
    Conformer convolutional sub-layer.

    Pointwise(d→2d) → GLU(dim=channel) → Depthwise → BN → SiLU → Pointwise(d→d)

    The GLU gate halves the channel count back to d after the first
    pointwise expansion, replacing the standard ReLU non-linearity with
    a data-dependent gate that empirically trains faster.
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "conv_kernel must be odd for same-length output"
        padding = (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d]
        x = self.norm(x)
        x = x.transpose(1, 2)          # [B, d, T]
        x = self.pw1(x)                # [B, 2d, T]
        x = F.glu(x, dim=1)            # [B, d, T]
        x = self.dw(x)                 # [B, d, T]
        x = self.act(self.bn(x))       # [B, d, T]
        x = self.pw2(x)                # [B, d, T]
        x = self.drop(x)
        return x.transpose(1, 2)       # [B, T, d]


class _ConformerBlock(nn.Module):
    """
    Full Conformer block (eq. 1 from Gulati et al. 2020):

        x ← x + ½ · FF(LN(x))
        x ← x + MHSA(LN(x))
        x ← x + ConvModule(x)
        x ← x + ½ · FF(LN(x))
        x ← LN(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        conv_kernel: int,
        dropout: float,
    ):
        super().__init__()
        self.ff1 = _ConformerFF(d_model, ff_dim, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.conv = _ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = _ConformerFF(d_model, ff_dim, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d]
        x = x + 0.5 * self.ff1(x)
        x_n = self.norm_attn(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + attn_out
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


# ============================================================================
# CsiConformer
# ============================================================================

class CsiConformer(nn.Module):
    """
    Conformer adapted for CSI-based Human Activity Recognition.

    The macaron feed-forward structure (scaled residual weights of ½) provides
    stable gradient flow through deep stacks.  The convolutional module gives
    a kernel-sized local context at every position — equivalent to a learnable
    relative positional bias — which is especially useful for capturing the
    sharp onset/offset of gestures that pure attention tends to smooth over.

    Input:  [B, 1, win_len, feature_size]
    Output: [B, num_classes]

    Default parameter count: ~1.56 M
      (conf_d_model=128, conf_layers=4, ff_dim=512, n_heads=4, conv_kernel=31)
    """

    def __init__(
        self,
        feature_size: int = 232,
        conf_d_model: int = 128,
        n_heads: int = 4,
        conf_layers: int = 4,
        ff_dim: int = 512,
        conv_kernel: int = 31,
        dropout: float = 0.2,
        num_classes: int = 2,
        win_len: int = 500,
        d_fft: int = 32,
    ):
        super().__init__()
        if conf_d_model % n_heads != 0:
            raise ValueError(
                f"conf_d_model ({conf_d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.feature_size = feature_size
        self.conf_d_model = conf_d_model
        self.conf_layers = conf_layers
        self.num_classes = num_classes
        self.win_len = win_len
        self.d_fft = d_fft

        # Method 3 + 2: grouped subcarrier stem replaces Linear(F→d_model)
        self.stem = SubcarrierGroupStem(feature_size, group_size=8,
                                        d_model=conf_d_model, dropout=dropout)

        # Method 1: FFT frequency branch
        self.fft_branch = FFTBranch(win_len=win_len, feature_size=feature_size,
                                     group_size=8, d_fft=d_fft, dropout=dropout)

        self.pos_enc = _SinusoidalPE(conf_d_model, dropout)

        self.blocks = nn.ModuleList([
            _ConformerBlock(conf_d_model, n_heads, ff_dim, conv_kernel, dropout)
            for _ in range(conf_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(conf_d_model + d_fft, num_classes)

    def get_init_params(self) -> dict:
        return {
            "feature_size": self.feature_size,
            "conf_d_model": self.conf_d_model,
            "conf_layers": self.conf_layers,
            "num_classes": self.num_classes,
            "win_len": self.win_len,
            "d_fft": self.d_fft,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)                   # [B, T, F]
        if x.shape[-1] != self.feature_size:
            x = x.transpose(1, 2)             # [B, T, F]

        fft_feat = self.fft_branch(x)          # [B, d_fft]  (Method 1)

        x = self.stem(x)                       # [B, T, d_model]  (Methods 2+3)
        x = self.pos_enc(x)                    # [B, T, d_model]
        for block in self.blocks:
            x = block(x)                       # [B, T, d_model]

        x = x.transpose(1, 2)                  # [B, d_model, T]
        x = self.pool(x).squeeze(-1)           # [B, d_model]
        x = torch.cat([x, fft_feat], dim=-1)   # [B, d_model + d_fft]
        return self.head(x)                    # [B, num_classes]


# ============================================================================
# DeepBiGRU
# ============================================================================

class DeepBiGRU(nn.Module):
    """
    Deep Bidirectional GRU + Transformer encoder for CSI-HAR.

    Unlike prunedAttentionGRU's unidirectional GRU (each timestep only has
    access to past context), the BiGRU encodes every position with both
    causal and anti-causal context simultaneously.  With two stacked BiGRU
    layers, second-order temporal patterns (how local patterns evolve) are
    captured before the Transformer attention layer runs.

    The Transformer encoder (2 layers with GELU FFN) refines the BiGRU
    output via global self-attention: it learns which timesteps are most
    discriminative across the full sequence window.

    Concat mean+max pooling preserves both the sustained average signal and
    the single sharpest discriminative feature, yielding a richer fixed-size
    vector than mean-pooling alone.

    Input:  [B, 1, win_len, feature_size]
    Output: [B, num_classes]

    Default parameter count: ~1.15 M
      (proj_dim=128, bigru_hidden=96, bigru_layers=2, 1 Transformer layer,
       d_gru=192, n_heads=8, ff_dim=512)
    Reduced from 1.71 M: 1 transformer layer instead of 2, bigru_hidden 96
    instead of 128.  The original 1.71 M model memorised device/environment-
    specific signal characteristics (89 % in-domain, 35 % cross-device).
    Reducing capacity forces the model to learn more generalisable features.
    """

    def __init__(
        self,
        feature_size: int = 232,
        proj_dim: int = 128,
        bigru_hidden: int = 96,
        bigru_layers: int = 2,
        n_transformer_layers: int = 1,
        n_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 2,
        win_len: int = 500,
        d_fft: int = 32,
    ):
        super().__init__()
        d_gru = bigru_hidden * 2          # bidirectional output dim
        if d_gru % n_heads != 0:
            raise ValueError(
                f"bigru_hidden*2 ({d_gru}) must be divisible by n_heads ({n_heads})"
            )
        self.feature_size = feature_size
        self.bigru_hidden = bigru_hidden
        self.bigru_layers = bigru_layers
        self.n_transformer_layers = n_transformer_layers
        self.num_classes = num_classes
        self.win_len = win_len
        self.d_fft = d_fft

        # Method 3 + 2: grouped subcarrier stem replaces Linear(F→proj_dim)
        self.stem = SubcarrierGroupStem(feature_size, group_size=8,
                                        d_model=proj_dim, dropout=dropout)

        # Method 1: FFT frequency branch
        self.fft_branch = FFTBranch(win_len=win_len, feature_size=feature_size,
                                     group_size=8, d_fft=d_fft, dropout=dropout)

        self.bigru = nn.GRU(
            input_size=proj_dim,
            hidden_size=bigru_hidden,
            num_layers=bigru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bigru_layers > 1 else 0.0,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_gru,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_transformer_layers
        )

        # Concat-pool: avg + max + fft → 2 × d_gru + d_fft
        self.backbone_dim = d_gru * 2 + d_fft
        self.head = nn.Sequential(
            nn.Linear(self.backbone_dim, d_gru),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_gru, num_classes),
        )

    def get_init_params(self) -> dict:
        return {
            "feature_size": self.feature_size,
            "bigru_hidden": self.bigru_hidden,
            "bigru_layers": self.bigru_layers,
            "num_classes": self.num_classes,
            "win_len": self.win_len,
            "d_fft": self.d_fft,
        }

    def forward_features(self, x: torch.Tensor):
        """Return (logits, feat) where feat [B, backbone_dim] is the pre-head embedding."""
        if x.dim() == 4:
            x = x.squeeze(1)                    # [B, T, F]
        if x.shape[-1] != self.feature_size:
            x = x.transpose(1, 2)              # [B, T, F]

        fft_feat = self.fft_branch(x)           # [B, d_fft]

        x = self.stem(x)                        # [B, T, proj_dim]
        x, _ = self.bigru(x)                   # [B, T, d_gru]
        x = self.transformer(x)                # [B, T, d_gru]

        avg_pool = x.mean(dim=1)               # [B, d_gru]
        max_pool = x.max(dim=1).values         # [B, d_gru]
        feat = torch.cat([avg_pool, max_pool, fft_feat], dim=-1)  # [B, backbone_dim]
        return self.head(feat), feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits


# ============================================================================
# HierCSIFormer building blocks
# ============================================================================

class _MultiScaleBlock(nn.Module):
    """Three depthwise-separable 1D conv branches at different temporal scales,
    fused by a pointwise conv with a residual skip.

    Captures activity motifs at short (5-step ≈ 0.1 s), mid (11-step ≈ 0.2 s),
    and long (21-step ≈ 0.4 s) time scales before downsampling."""

    def __init__(self, d_model: int, kernels: tuple = (5, 11, 21), dropout: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model, bias=False),
                nn.Conv1d(d_model, d_model, 1, bias=False),
                nn.GELU(),
            )
            for k in kernels
        ])
        n = len(kernels)
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model * n, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, d, T]
        h = self.norm(x)
        h = self.fuse(torch.cat([b(h) for b in self.branches], dim=1))
        return x + h


class _StridedDSConv(nn.Module):
    """Strided depthwise-separable 1D conv for temporal downsampling.

    Uses kernel == stride so there is no overlap and output length is
    exactly T_in // stride (no padding needed).
    """

    def __init__(self, d_model: int, kernel: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel, stride=stride, groups=d_model, bias=False),
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# HierCSIFormer
# ============================================================================

class HierCSIFormer(nn.Module):
    """
    Hierarchical CSI Transformer for CSI-Bench HAR.

    Stage 1 — Subcarrier stem + multi-scale 1D temporal convs:
        SubcarrierGroupStem projects the 232 subcarrier channels (grouped into
        29 bins) to d_model, applying validity gating for 2G zero-padded samples.
        A _MultiScaleBlock fuses parallel depthwise-separable convs at kernels
        (5, 11, 21) to capture short/mid/long activity motifs simultaneously.
        Two strided convs reduce the sequence: T=500 → 100 → 25.

    Stage 2 — CLS-token Transformer encoder:
        A learnable CLS token is prepended and sinusoidal PE added.
        `hiercsi_layers` pre-LN Transformer encoder blocks apply global
        self-attention over T'=25+1=26 positions (O(26²) vs O(500²) in
        CsiConformer), then the CLS output is used for classification.

    Stage 3 — FFT branch fusion + head:
        An FFT branch extracts frequency-domain activity signatures (d_fft=64)
        and is concatenated with the CLS embedding before the linear head.

    Input:  [B, 1, win_len=500, feature_size=232]
    Output: [B, num_classes]

    Default ~5.5 M parameters (hiercsi_d_model=256, hiercsi_layers=6).
    """

    def __init__(
        self,
        feature_size: int = 232,
        win_len: int = 500,
        num_classes: int = 2,
        hiercsi_d_model: int = 256,
        hiercsi_layers: int = 6,
        dropout: float = 0.1,
        d_fft: int = 64,
    ):
        super().__init__()
        d = hiercsi_d_model
        n_heads = 8
        if d % n_heads != 0:
            raise ValueError(
                f"hiercsi_d_model ({d}) must be divisible by 8 (n_heads={n_heads})"
            )

        self.feature_size = feature_size
        self.win_len = win_len
        self.num_classes = num_classes
        self.hiercsi_d_model = hiercsi_d_model
        self.hiercsi_layers = hiercsi_layers
        self.d_fft = d_fft

        # ---- Stage 1: subcarrier projection --------------------------------
        self.stem = SubcarrierGroupStem(
            feature_size, group_size=8, d_model=d, dropout=dropout
        )

        # ---- Stage 1: multi-scale temporal block + downsampling ------------
        self.ms_block = _MultiScaleBlock(d, kernels=(5, 11, 21), dropout=dropout)
        # kernel == stride ⟹ no padding, exact T-division
        self.down1 = _StridedDSConv(d, kernel=5, stride=5)   # T: 500 → 100
        self.down2 = _StridedDSConv(d, kernel=4, stride=4)   # T: 100 → 25

        # ---- Stage 2: CLS token + PE + Transformer -------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_enc = _SinusoidalPE(d, dropout, max_len=64)  # 26 positions

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,              # pre-LN: more stable than post-LN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=hiercsi_layers)
        self.norm_out = nn.LayerNorm(d)

        # ---- Stage 3: FFT branch + head ------------------------------------
        self.fft_branch = FFTBranch(
            win_len=win_len, feature_size=feature_size,
            group_size=8, d_fft=d_fft, dropout=dropout,
        )
        self.head = nn.Linear(d + d_fft, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def get_init_params(self) -> dict:
        return {
            "feature_size": self.feature_size,
            "win_len": self.win_len,
            "num_classes": self.num_classes,
            "hiercsi_d_model": self.hiercsi_d_model,
            "hiercsi_layers": self.hiercsi_layers,
            "d_fft": self.d_fft,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)                  # [B, T, F]
        if x.shape[-1] != self.feature_size:
            x = x.transpose(1, 2)            # [B, T, F]

        fft_feat = self.fft_branch(x)        # [B, d_fft]

        # Stage 1
        x = self.stem(x)                     # [B, T=500, d]
        x = x.transpose(1, 2)               # [B, d, T=500]
        x = self.ms_block(x)                # [B, d, T=500]
        x = self.down1(x)                   # [B, d, T=100]
        x = self.down2(x)                   # [B, d, T=25]
        x = x.transpose(1, 2)              # [B, T=25, d]

        # Stage 2
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d]
        x = torch.cat([cls, x], dim=1)           # [B, 26, d]
        x = self.pos_enc(x)                      # [B, 26, d]
        x = self.transformer(x)                  # [B, 26, d]
        x = self.norm_out(x[:, 0])              # [B, d]  — CLS token

        # Stage 3
        x = torch.cat([x, fft_feat], dim=-1)   # [B, d + d_fft]
        return self.head(x)                      # [B, num_classes]


# ============================================================================
# ResNet18CSI — teacher model for knowledge distillation
# ============================================================================

def _replace_bn_with_in(module: nn.Module) -> nn.Module:
    """Recursively replace every BatchNorm2d in *module* with InstanceNorm2d.

    Preserves:
      num_features — channel count stays identical
      affine=True  — keeps learnable γ/β so the layer has the same expressive
                     power as BN (InstanceNorm defaults to affine=False)
      eps          — copied from the original BN layer

    track_running_stats is set to False: IN normalises per sample so there are
    no batch-level running statistics to track, and keeping them True would
    accumulate statistics during training that are never used at eval time.

    The function operates in-place via setattr and also returns the module.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.InstanceNorm2d(
                child.num_features,
                affine=True,
                track_running_stats=False,
                eps=child.eps,
            ))
        else:
            _replace_bn_with_in(child)
    return module


class ResNet18CSI(nn.Module):
    """
    ResNet-18 adapted as a high-capacity teacher for CSI-based HAR.

    The standard ResNet-18 2D convolutional stack treats the CSI window as a
    2-D image [T=500 × F=232] with 1 amplitude channel, which lets the spatial
    conv kernels jointly learn temporal dynamics and cross-subcarrier patterns
    — an inductive bias unavailable to 1-D temporal models.

    Architecture:
        Backbone  ResNet-18 conv layers 1-4 + AdaptiveAvgPool2d → [B, 512]
        Head      nn.Linear(512, num_classes)  — named self.head for
                  DomainAdversarialTrainer hook compatibility

    Knowledge distillation interface:
        forward_features(x) → (logits, backbone_feat)
          logits        [B, num_classes] — use as soft targets (temperature-scaled)
          backbone_feat [B, 512]         — use for intermediate feature matching

    DomainAdversarialTrainer compatibility:
        The trainer hooks self.model.head and captures its input tensor as the
        backbone feature for the gradient-reversal domain heads.  Because this
        class exposes an explicit self.head attribute and routes all forward
        passes through self.head(feat), the hook fires correctly and
        backbone_feat shape [B, 512] is consistent with partial_adv_frac slicing.

    Input:  [B, 1, win_len, feature_size]  (e.g. [B, 1, 500, 232])
    Output: [B, num_classes]

    Normalisation:
        All 20 BatchNorm2d layers (bn1 + 2 per BasicBlock × 8 blocks, including
        downsample shortcuts) are replaced with InstanceNorm2d (affine=True,
        track_running_stats=False).  IN normalises per sample rather than per
        batch, so its statistics are never domain-specific — a key advantage for
        cross-domain generalisation and domain-adversarial training where BN
        running stats could carry domain identity information.

    Parameter count: ~11.2 M (standard ResNet-18; IN has same γ/β count as BN)
    """

    def __init__(
        self,
        win_len: int = 500,
        feature_size: int = 232,
        num_classes: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.win_len = win_len
        self.feature_size = feature_size
        self.num_classes = num_classes

        _r = resnet18(weights=None)

        # Replace conv1: ImageNet RGB input (3-ch) → CSI amplitude input (1-ch)
        _r.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace all BatchNorm2d with InstanceNorm2d before assembling backbone.
        # Must happen before the Sequential is built so that _r.bn1 and the BNs
        # inside layer1-4 (including downsample sub-modules) are all replaced in
        # one traversal of the torchvision ResNet object.
        _replace_bn_with_in(_r)

        # Backbone: conv stack + global average pool → [B, 512, 1, 1]
        self.backbone = nn.Sequential(
            _r.conv1, _r.bn1, _r.relu, _r.maxpool,
            _r.layer1, _r.layer2, _r.layer3, _r.layer4,
            _r.avgpool,
        )

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Named 'head': DomainAdversarialTrainer registers a forward hook here
        # and reads inp[0] (the 512-dim feature) to build domain heads
        self.head = nn.Linear(512, num_classes)

    def get_init_params(self) -> dict:
        return {
            "win_len": self.win_len,
            "feature_size": self.feature_size,
            "num_classes": self.num_classes,
        }

    def forward_features(self, x: torch.Tensor):
        """Return (logits, backbone_feat) for knowledge distillation.

        backbone_feat [B, 512] can be used for soft-target KD (response-based)
        or intermediate feature matching (feature-based KD).
        """
        feat = self.backbone(x).flatten(1)   # [B, 512]
        feat = self.drop(feat)
        return self.head(feat), feat          # hook fires on self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits
