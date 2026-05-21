"""
DomainAdversarialTrainer — AugmentedTaskTrainer subclass with gradient reversal.

Implements domain-adversarial training (Ganin et al., JMLR 2016):
  backbone → HAR head  (+gradient → backbone learns discriminative HAR features)
  backbone → GRL → domain heads  (−gradient → backbone learns domain-invariant features)

Loss formulation:
  L_total = L_Focal + λ_domain(p) · L_adv

  L_Focal       = -α_t (1-p_t)^γ log(p_t)  — weighted focal loss (γ=2 default)
  L_adv         = mean CE over domain heads (device / environment / user)
  λ_domain(p)   = 2/(1+exp(−5p)) − 1       — sigmoid schedule, p=epoch/total_epochs
                  grows from ≈0 at epoch 0 to ≈1 at epoch 100

Active improvements:
  P1 — HAR accuracy gate: adversarial training suppressed until
       train_acc ≥ da_start_threshold, preventing GRL from disrupting
       early feature learning.
  P4 — Partial adversarial: GRL applied only to the last partial_adv_frac of
       backbone features; domain heads built on that reduced dimension.
  P5 — Layer freezing: when gate first opens, the first freeze_frac fraction
       of backbone parameters are frozen permanently.

Domain labels required at training time:
  Each training batch must be a 3-tuple (csi, act_label, domain_tensor) where
  domain_tensor has shape [B, n_domain_cols] and encodes (device, environment,
  user) indices.  BenchmarkCSIDataset produces this 3-tuple when initialised
  with domain_columns=['device', 'environment', 'user'].

Evaluation is inherited from TaskTrainer (expects 2-tuple batches).
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.supervised.augmented_trainer import AugmentedTaskTrainer


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GRLFunction(torch.autograd.Function):
    """Reverses gradients scaled by lambda_ during backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Passes features unchanged in the forward pass; negates gradients scaled
    by lambda_ during the backward pass."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# Domain classifier head
# ---------------------------------------------------------------------------

class _DomainHead(nn.Module):
    """Small MLP that predicts domain identity from backbone features."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DomainAdversarialTrainer
# ---------------------------------------------------------------------------

class DomainAdversarialTrainer(AugmentedTaskTrainer):
    """AugmentedTaskTrainer extended with gradient-reversal domain heads.

    Three domain heads are attached to the backbone's pre-classification
    representation (captured via a forward hook on model.head):
      - device classifier  (e.g. 8 training devices)
      - environment classifier  (e.g. 4-5 training environments)
      - user classifier  (e.g. 5 training users)

    Loss formulation:
        L = L_Focal + λ_domain(p) · mean(L_device, L_env, L_user)
        λ_domain(p) = 2/(1+exp(−5p)) − 1,  p = epoch / total_epochs

    The GRL ensures that backbone gradient from the domain terms is negated,
    pushing the backbone toward domain-invariant representations.

    Parameters
    ----------
    domain_n_classes  : list of ints — number of classes per domain column.
    domain_head_lr    : learning rate for domain head parameters.
    da_start_threshold: minimum training HAR accuracy before DA activates (P1).
    partial_adv_frac  : fraction of backbone features exposed to GRL (P4).
                        1.0 = full features (original behaviour).
                        0.5 = only the last 50 % of features receive adversarial
                        gradient; domain heads are built with this reduced dim.
    freeze_frac       : fraction of backbone parameter groups to freeze when the
                        DA gate first opens (P5).  0.0 = no freezing.
                        Parameters are frozen permanently for the rest of training.
    """

    def __init__(
        self,
        *args,
        domain_n_classes: list = None,
        domain_head_lr: float = 1e-3,
        da_start_threshold: float = 0.60,
        partial_adv_frac: float = 1.0,
        freeze_frac: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.domain_n_classes = domain_n_classes or []
        self.partial_adv_frac = max(0.01, min(1.0, partial_adv_frac))
        self.freeze_frac = max(0.0, min(1.0, freeze_frac))

        # ---- HAR-accuracy gate state (P1) ----------------------------------
        self.da_start_threshold = da_start_threshold
        self._prev_har_acc = 0.0
        self._da_gate_opened_epoch = None
        self._backbone_frozen = False   # P5: becomes True after first freeze

        # ---- Forward hook to capture backbone features ----------------------
        self._backbone_features: dict = {}

        def _hook(module, inp, output):
            self._backbone_features['x'] = inp[0]

        self._hook_handle = self.model.head.register_forward_hook(_hook)

        # ---- Detect backbone feature dimension via dummy forward pass -------
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 1, 500, 232, device=self.device)
            self.model(dummy)
        backbone_dim = self._backbone_features['x'].shape[-1]

        # P4: domain heads use only the last adv_dim features (partial adversarial)
        self.adv_dim = max(1, int(backbone_dim * self.partial_adv_frac))

        print(
            f"[DomainAdversarialTrainer] backbone_dim={backbone_dim}, "
            f"adv_dim={self.adv_dim} (partial_adv_frac={self.partial_adv_frac}), "
            f"domain_n_classes={self.domain_n_classes}, "
            f"lambda_domain=scheduled(0→1 via sigmoid), "
            f"freeze_frac={freeze_frac}"
        )
        self.model.train()

        if not self.domain_n_classes:
            print("[DomainAdversarialTrainer] WARNING: domain_n_classes is empty — "
                  "no domain heads will be created.  Domain adversarial loss disabled.")

        # ---- Lambda schedule -----------------------------------------------
        if isinstance(self.config, dict):
            self._total_epochs = self.config.get('epochs', 100)
        else:
            self._total_epochs = getattr(self.config, 'epochs', 100)

        # ---- Build GRL + domain heads (P4: heads use adv_dim, not full dim) -
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.domain_heads = nn.ModuleList([
            _DomainHead(self.adv_dim, n_cls) for n_cls in self.domain_n_classes
        ]).to(self.device)

        if self.domain_heads:
            self.domain_optimizer = torch.optim.AdamW(
                self.domain_heads.parameters(), lr=domain_head_lr
            )
        else:
            self.domain_optimizer = None

    # ------------------------------------------------------------------
    # P5: Layer freezing
    # ------------------------------------------------------------------

    def _freeze_lower_layers(self):
        """Freeze the first freeze_frac fraction of backbone parameter groups.

        Parameters are sorted in the order returned by model.named_parameters(),
        which follows the order of module registration (stem → blocks → head).
        Freezing is permanent for the rest of training.
        """
        all_params = [
            (name, p) for name, p in self.model.named_parameters()
            if p.requires_grad
        ]
        n_freeze = int(len(all_params) * self.freeze_frac)
        for name, p in all_params[:n_freeze]:
            p.requires_grad_(False)
        remaining = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"[DomainAdversarialTrainer] P5: froze {n_freeze}/{len(all_params)} "
            f"param groups ({remaining:,} trainable params remain)"
        )
        self._backbone_frozen = True

    # ------------------------------------------------------------------
    # Training epoch override
    # ------------------------------------------------------------------

    def train_epoch(self):
        """One epoch: L_Focal + λ_domain(p) · L_adv.

        Expects 3-tuple batches (csi, act_label, domain_tensor) from the
        training loader when domain_columns are configured.  Falls back to
        2-tuple batches (no domain loss) for compatibility.
        """
        self.model.train()
        if self.domain_heads:
            self.domain_heads.train()

        # ---- P1: HAR-accuracy-gated adversarial loss -----------------------
        epoch = getattr(self, 'current_epoch', 0)
        gate_open = self._prev_har_acc >= self.da_start_threshold

        # λ_domain(p) = 2/(1+exp(−10p)) − 1, p = epoch/total_epochs (global)
        p = epoch / max(1, self._total_epochs)
        lambda_domain = 2.0 / (1.0 + math.exp(-5.0 * p)) - 1.0

        if gate_open:
            if self._da_gate_opened_epoch is None:
                self._da_gate_opened_epoch = epoch
                print(
                    f"[DomainAdversarialTrainer] DA gate opened at epoch {epoch} "
                    f"(train_acc={self._prev_har_acc:.3f} >= "
                    f"threshold={self.da_start_threshold}, "
                    f"lambda_domain={lambda_domain:.4f})"
                )
                # P5: freeze lower layers the moment the gate first opens
                if self.freeze_frac > 0.0 and not self._backbone_frozen:
                    self._freeze_lower_layers()

            self.grl.lambda_ = 1.0   # full gradient reversal; scaling via lambda_domain
        else:
            self.grl.lambda_ = 0.0

        epoch_loss = epoch_acc = total_samples = total_time = 0.0

        for batch in self.train_loader:
            # Unpack batch — tolerate 2-tuple (no domain labels) and 3-tuple
            if len(batch) == 3:
                inputs, act_labels, domain_labels = batch
                domain_labels = domain_labels.to(self.device)
                has_domain = True
            else:
                inputs, act_labels = batch
                has_domain = False

            if inputs.size(0) == 0:
                continue

            batch_size = inputs.size(0)
            total_samples += batch_size
            inputs = inputs.to(self.device)

            if isinstance(act_labels, tuple):
                act_labels = act_labels[0]
            act_labels = act_labels.to(self.device)

            # ---- Online augmentation (no-op when use_augmentation=False) -----
            inputs = self._apply_input_augmentation(inputs)

            # ---- Forward pass ----------------------------------------------
            start = time.time()
            self.optimizer.zero_grad()
            if self.domain_optimizer is not None:
                self.domain_optimizer.zero_grad()

            outputs = self.model(inputs)
            backbone_feat = self._backbone_features.get('x')  # [B, backbone_dim]

            # ---- HAR loss --------------------------------------------------
            har_loss = self._loss(outputs, act_labels)
            loss = har_loss

            if gate_open and has_domain and backbone_feat is not None and self.domain_heads:
                # ---- P4 + GRL adversarial loss -----------------------------
                # P4: GRL applied only to the last adv_dim features
                adv_feat = backbone_feat[:, -self.adv_dim:]
                reversed_feat = self.grl(adv_feat)
                d_losses = [
                    F.cross_entropy(head(reversed_feat), domain_labels[:, i])
                    for i, head in enumerate(self.domain_heads)
                    if i < domain_labels.shape[1]
                ]
                if d_losses:
                    domain_loss = sum(d_losses) / len(d_losses)
                    loss = loss + lambda_domain * domain_loss

            # ---- Backward + step -------------------------------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.grad_clip_norm)
            if self.domain_heads:
                torch.nn.utils.clip_grad_norm_(self.domain_heads.parameters(),
                                               max_norm=self.grad_clip_norm)
            self.optimizer.step()
            if self.domain_optimizer is not None:
                self.domain_optimizer.step()

            total_time += time.time() - start
            epoch_loss += har_loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            epoch_acc += (preds == act_labels).sum().item()

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        self._prev_har_acc = epoch_acc / total_samples

        return (
            epoch_loss / total_samples,
            epoch_acc / total_samples,
            total_time / total_samples,
        )
