"""
PrunedAttentionGRU model adapted for CSI-Bench.

Architecture from: https://github.com/harikang/prunedAttentionGRU
  - Custom GRU with MaskedLinear gates (supports weight pruning)
  - Additive attention (Bahdanau-style) with MaskedLinear layers
  - Optional magnitude-based / random pruning via PruningModule

Input format: [B, 1, win_len, feature_size]  (CSI-Bench 4-D format)
Output format: [B, num_classes]
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from model.supervised.csi_stem import FFTBranch, SubcarrierGroupStem


# ---------------------------------------------------------------------------
# MaskedLinear
# ---------------------------------------------------------------------------

class MaskedLinear(nn.Module):
    """Linear layer whose weights are element-wise multiplied by a binary mask.

    The mask is a non-trainable Parameter so it is part of the state_dict and
    is saved/loaded with the model.  Weights outside the mask are zeroed out
    during the forward pass, not physically removed from memory.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(
            torch.ones(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.ones_(self.mask)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def prune(self, threshold: float, k: float) -> bool:
        """Zero-out weights whose absolute value is below threshold.

        Returns False (and does NOT apply the mask) if the resulting sparsity
        would drop below the minimum density k.
        """
        dev = self.weight.device
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(np.abs(tensor) < threshold, 0.0, mask)
        nz = np.count_nonzero(new_mask)
        if k <= nz / (self.in_features * self.out_features):
            self.weight.data = torch.from_numpy(tensor * new_mask).to(dev)
            self.mask.data = torch.from_numpy(new_mask).to(dev)
            return True
        return False


# ---------------------------------------------------------------------------
# MaskedAttention
# ---------------------------------------------------------------------------

class MaskedAttention(nn.Module):
    """Additive (Bahdanau-style) self-attention with prunable linear layers.

    Computes a context vector by weighting GRU hidden states.

    Args:
        hidden_dim:    Dimensionality of the GRU output.
        attention_dim: Internal dimensionality of the attention projection.
    """

    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.query = MaskedLinear(hidden_dim, attention_dim)
        self.key = MaskedLinear(hidden_dim, attention_dim)
        self.value = MaskedLinear(hidden_dim, attention_dim)
        self.context_vector = MaskedLinear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, seq_len, hidden_dim]

        Returns:
            context: [B, attention_dim]
        """
        q = self.query(hidden_states)          # [B, seq_len, attention_dim]
        k = self.key(hidden_states)            # [B, seq_len, attention_dim]
        v = self.value(hidden_states)          # [B, seq_len, attention_dim]

        scores = torch.tanh(q + k)             # [B, seq_len, attention_dim]
        scores = self.context_vector(scores).squeeze(-1)   # [B, seq_len]
        weights = F.softmax(scores, dim=-1)    # [B, seq_len]

        # Weighted sum over sequence → context vector
        context = (v * weights.unsqueeze(-1)).sum(dim=1)   # [B, attention_dim]
        return context

    def prune(self, threshold: float, k: float) -> bool:
        for linear in [self.query, self.key, self.value, self.context_vector]:
            tensor = linear.weight.data.cpu().numpy()
            mask = linear.mask.data.cpu().numpy()
            new_mask = np.where(np.abs(tensor) < threshold, 0.0, mask)
            nz = np.count_nonzero(new_mask)
            if k <= nz / (linear.in_features * linear.out_features):
                dev = linear.weight.device
                linear.weight.data = torch.from_numpy(tensor * new_mask).to(dev)
                linear.mask.data = torch.from_numpy(new_mask).to(dev)
            else:
                return False
        return True


# ---------------------------------------------------------------------------
# CustomGRU
# ---------------------------------------------------------------------------

class CustomGRU(nn.Module):
    """GRU whose update / reset / new-memory gates are MaskedLinear layers.

    This replaces nn.GRU so that individual weight elements can be pruned.
    The interface matches nn.GRU with batch_first support.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        combined = input_size + hidden_size
        self.update_gate = MaskedLinear(combined, hidden_size, bias=bias)
        self.reset_gate = MaskedLinear(combined, hidden_size, bias=bias)
        self.new_memory = MaskedLinear(combined, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor, hx: torch.Tensor = None):
        """
        Args:
            x:  [B, seq_len, input_size] if batch_first, else [seq_len, B, input_size]
            hx: optional initial hidden state [1, B, hidden_size]

        Returns:
            (output, h_n):
                output: [B, seq_len, hidden_size] (or seq-first)
                h_n:    [1, B, hidden_size]
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # → [seq_len, B, input_size]

        seq_len, batch_size, _ = x.size()

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            # Accept [1, B, hidden_size] like nn.GRU
            hx = hx.squeeze(0)

        output = []
        for t in range(seq_len):
            xt = x[t]                                          # [B, input_size]
            combined = torch.cat((xt, hx), dim=1)             # [B, input_size+hidden_size]

            z_t = torch.sigmoid(self.update_gate(combined))   # [B, hidden_size]
            r_t = torch.sigmoid(self.reset_gate(combined))    # [B, hidden_size]

            combined_reset = torch.cat((xt, r_t * hx), dim=1)
            n_t = torch.tanh(self.new_memory(combined_reset))  # [B, hidden_size]

            hx = (1.0 - z_t) * n_t + z_t * hx
            output.append(hx)

        output = torch.stack(output, dim=0)  # [seq_len, B, hidden_size]

        if self.batch_first:
            output = output.transpose(0, 1)  # → [B, seq_len, hidden_size]

        h_n = hx.unsqueeze(0)  # [1, B, hidden_size]
        return output, h_n

    def prune(self, threshold: float, k: float) -> bool:
        for linear in [self.update_gate, self.reset_gate, self.new_memory]:
            tensor = linear.weight.data.cpu().numpy()
            mask = linear.mask.data.cpu().numpy()
            new_mask = np.where(np.abs(tensor) < threshold, 0.0, mask)
            nz = np.count_nonzero(new_mask)
            if k <= nz / (linear.in_features * linear.out_features):
                dev = linear.weight.device
                linear.weight.data = torch.from_numpy(tensor * new_mask).to(dev)
                linear.mask.data = torch.from_numpy(new_mask).to(dev)
            else:
                return False
        return True


# ---------------------------------------------------------------------------
# PruningModule
# ---------------------------------------------------------------------------

class PruningModule(nn.Module):
    """Mixin that adds magnitude-based and random pruning to a model."""

    def prune_by_std(self, s: float = 0.5, k: float = 0.05):
        """Prune each MaskedLinear weight whose |value| < s × std(|weights|).

        Iterates only over leaf MaskedLinear layers to avoid double-pruning
        through composite modules (CustomGRU, MaskedAttention).
        """
        for name, module in self.named_modules():
            if isinstance(module, MaskedLinear):
                threshold = np.std(module.weight.data.abs().cpu().numpy()) * s
                print(f"[Prune] {name}: threshold={threshold:.6f}")
                while not module.prune(threshold, k):
                    threshold *= 0.99

    def prune_by_random(self, connectivity: float = 0.5):
        """Randomly prune MaskedLinear weights to the given connection density."""
        for name, module in self.named_modules():
            if isinstance(module, MaskedLinear):
                row, col = module.weight.shape
                mask = _random_mask((row, col), connectivity)
                init_w = nn.init.orthogonal_(module.weight.data)
                module.weight.data = init_w * torch.tensor(
                    mask, dtype=module.weight.dtype, device=module.weight.device
                )


def _random_mask(shape, connection):
    random.seed(1)
    s = np.random.uniform(size=shape)
    threshold = np.sort(s.flatten())[int(shape[0] * shape[1] * (1 - connection))]
    return (s >= threshold).astype(float)


# ---------------------------------------------------------------------------
# PrunedAttentionGRUClassifier  (CSI-Bench entry point)
# ---------------------------------------------------------------------------

class PrunedAttentionGRUClassifier(PruningModule):
    """Pruned-Attention-GRU model integrated into the CSI-Bench framework.

    The model stacks:
        CustomGRU  →  MaskedAttention  →  MaskedLinear (classifier head)

    Input accepted: [B, 1, win_len, feature_size]  (CSI-Bench 4-D format)
    Output:         [B, num_classes]

    Args:
        feature_size:   Number of CSI features per time step (default 232).
        hidden_dim:     GRU hidden state size (default 128).
        attention_dim:  Internal dimension of the attention projection (default 32).
        num_classes:    Number of output classes (determined by the task).
        win_len:        Sequence length (used only for get_init_params, default 500).
    """

    def __init__(
        self,
        feature_size: int = 232,
        hidden_dim: int = 128,
        attention_dim: int = 32,
        num_classes: int = 2,
        win_len: int = 500,
        d_fft: int = 32,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.win_len = win_len
        self.d_fft = d_fft

        # Method 3 + 2: grouped subcarrier stem; projects F→hidden_dim before GRU
        self.stem = SubcarrierGroupStem(feature_size, group_size=8,
                                        d_model=hidden_dim, dropout=0.1)

        # Method 1: FFT frequency branch
        self.fft_branch = FFTBranch(win_len=win_len, feature_size=feature_size,
                                     group_size=8, d_fft=d_fft, dropout=0.1)

        # GRU input_size is now hidden_dim (stem output) instead of feature_size
        self.gru = CustomGRU(hidden_dim, hidden_dim, bias=True, batch_first=True)
        self.attention = MaskedAttention(hidden_dim, attention_dim)
        self.fc = MaskedLinear(attention_dim + d_fft, num_classes)

    # ------------------------------------------------------------------
    # CSI-Bench compatibility
    # ------------------------------------------------------------------

    @property
    def head(self):
        """Alias for DomainAdversarialTrainer hook compatibility."""
        return self.fc

    def get_init_params(self) -> dict:
        """Return constructor kwargs — needed by CSI-Bench few-shot adapter."""
        return {
            "feature_size": self.feature_size,
            "hidden_dim": self.hidden_dim,
            "attention_dim": self.attention_dim,
            "num_classes": self.num_classes,
            "win_len": self.win_len,
            "d_fft": self.d_fft,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, win_len, feature_size]  or  [B, win_len, feature_size]

        Returns:
            logits: [B, num_classes]
        """
        if x.dim() == 4:
            x = x.squeeze(1)               # [B, T, F]
        if x.shape[-1] != self.feature_size:
            x = x.transpose(1, 2)          # [B, T, F]

        fft_feat = self.fft_branch(x)      # [B, d_fft]         (Method 1)
        x = self.stem(x)                   # [B, T, hidden_dim] (Methods 2+3)

        gru_out, _ = self.gru(x)           # [B, T, hidden_dim]
        context = self.attention(gru_out)  # [B, attention_dim]
        context = torch.cat([context, fft_feat], dim=-1)  # [B, attention_dim + d_fft]
        return self.fc(context)            # [B, num_classes]
