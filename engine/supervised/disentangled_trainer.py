"""
DisentangledTrainer — AugmentedTaskTrainer subclass for DisentangledCSI.

Multi-loss objective (per batch):

    L = (1 − α) · L_focal
        + α · T² · L_kd_soft
        + β · L_kd_feat
        + γ_dom · L_domain_ce
        + λ(p) · L_adv
        + γ_orth · L_ortho

  L_focal      : weighted focal CE on motion branch logits vs activity labels
  L_kd_soft    : KL divergence between student/teacher temperature-scaled logits
  L_kd_feat    : MSE between projected motion feature and teacher backbone feature
  L_domain_ce  : standard CE on domain branch logits vs domain labels
                 (trains domain branch to correctly predict domain identity)
  L_adv        : CE via GRL on motion features vs domain labels
                 (forces motion features to be domain-invariant)
  L_ortho      : mean absolute cosine similarity between motion and domain features
                 (enforces disentanglement)

Adversarial schedule:
  λ(p) = 2/(1 + exp(−5p)) − 1,  p = epoch / total_epochs
  Gated by da_start_threshold: λ stays 0 until train_acc ≥ threshold.

Projection head:
  nn.Linear(backbone_dim → 512) maps student motion feature to teacher space.
  Parameters are added to the existing optimizer; projector is training-only.

Teacher interface: forward_features(x) → (logits, feat [B, 512]).  Frozen.
Student interface: forward_full(x) → (motion_logits, motion_feat,
                                       domain_logits, domain_feat).
                   DisentangledCSI satisfies both.
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.supervised.augmented_trainer import AugmentedTaskTrainer
from engine.supervised.domain_adversarial_trainer import GradientReversalLayer


class DisentangledTrainer(AugmentedTaskTrainer):
    """AugmentedTaskTrainer extended with KD + disentanglement losses.

    Parameters
    ----------
    teacher           : nn.Module — frozen teacher (ResNet18CSI).
                        Must implement forward_features(x) → (logits, feat [B, 512]).
    kd_temperature    : float — distillation temperature T (default 4.0).
    kd_alpha          : float — weight of soft-target KD loss (default 0.5).
    kd_beta           : float — weight of feature MSE loss (default 0.1).
    lambda_domain_ce  : float — weight of domain branch CE loss (default 0.5).
    lambda_ortho      : float — weight of orthogonal regularisation (default 0.1).
    da_start_threshold: float — HAR train accuracy required before adversarial
                        loss activates (default 0.60).
    """

    TEACHER_FEAT_DIM = 512

    def __init__(
        self,
        *args,
        teacher: nn.Module,
        kd_temperature: float = 4.0,
        kd_alpha: float = 0.5,
        kd_beta: float = 0.1,
        lambda_domain_ce: float = 0.5,
        lambda_ortho: float = 0.1,
        da_start_threshold: float = 0.60,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ---- Validate student interface ------------------------------------
        if not hasattr(self.model, 'forward_full'):
            raise AttributeError(
                f"{type(self.model).__name__} must implement forward_full(x) "
                f"for DisentangledTrainer."
            )
        if not hasattr(self.model, 'backbone_dim'):
            raise AttributeError(
                f"{type(self.model).__name__} must define backbone_dim."
            )

        # ---- Freeze teacher ------------------------------------------------
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher.to(self.device)

        # ---- KD projector: motion_channels → 512 ---------------------------
        self.kd_projector = nn.Linear(
            self.model.backbone_dim, self.TEACHER_FEAT_DIM
        ).to(self.device)
        self.optimizer.add_param_group({
            'params': self.kd_projector.parameters(),
        })

        # ---- Adversarial discriminators on motion features (one per domain) -
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.adv_heads = nn.ModuleList([
            nn.Linear(self.model.backbone_dim, n_cls)
            for n_cls in self.model.domain_n_classes
        ]).to(self.device)
        if self.adv_heads:
            self.optimizer.add_param_group({
                'params': self.adv_heads.parameters(),
            })

        # ---- Loss weights ---------------------------------------------------
        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha
        self.kd_beta = kd_beta
        self.lambda_domain_ce = lambda_domain_ce
        self.lambda_ortho = lambda_ortho
        self.da_start_threshold = da_start_threshold

        # ---- Adversarial gate state ----------------------------------------
        self._prev_har_acc = 0.0
        self._da_gate_opened_epoch = None

        if isinstance(self.config, dict):
            self._total_epochs = self.config.get('epochs', 100)
        else:
            self._total_epochs = getattr(self.config, 'epochs', 100)

        print(
            f"[DisentangledTrainer] "
            f"backbone_dim={self.model.backbone_dim}, "
            f"T={kd_temperature}, α={kd_alpha}, β={kd_beta}, "
            f"λ_dom={lambda_domain_ce}, λ_orth={lambda_ortho}, "
            f"da_threshold={da_start_threshold}"
        )

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _kd_losses(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
    ):
        """Soft-target KL + feature MSE losses (unscaled)."""
        T = self.kd_temperature
        s_log_soft = F.log_softmax(s_logits / T, dim=-1)
        t_soft = F.softmax(t_logits / T, dim=-1).detach()
        kd_soft = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (T * T)

        s_proj = self.kd_projector(s_feat)
        feat_loss = F.mse_loss(s_proj, t_feat.detach())
        return kd_soft, feat_loss

    def _ortho_loss(
        self,
        motion_feat: torch.Tensor,
        domain_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Mean absolute cosine similarity between motion and domain features.

        Bounded in [0, 1]; minimising pushes the two representations orthogonal.
        """
        m = F.normalize(motion_feat, dim=-1)  # [B, motion_channels]
        d = F.normalize(domain_feat, dim=-1)  # [B, domain_channels]
        # Align dims if motion_channels ≠ domain_channels via a dot product
        # over the shared leading batch dimension only (per-sample cosine).
        # For mismatched dims, fall back to per-sample L2 cosine proxy:
        min_dim = min(m.size(-1), d.size(-1))
        return (m[:, :min_dim] * d[:, :min_dim]).sum(dim=-1).abs().mean()

    def _domain_adv_loss(
        self,
        motion_feat: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Adversarial CE on motion features via GRL → domain-invariant motion."""
        reversed_feat = self.grl(motion_feat)
        losses = [
            F.cross_entropy(head(reversed_feat), domain_labels[:, i])
            for i, head in enumerate(self.adv_heads)
            if i < domain_labels.shape[1]
        ]
        return sum(losses) / len(losses) if losses else motion_feat.new_tensor(0.0)

    def _domain_ce_loss(
        self,
        domain_logits: list,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Standard CE on domain branch logits — trains domain branch."""
        losses = [
            F.cross_entropy(domain_logits[i], domain_labels[:, i])
            for i in range(min(len(domain_logits), domain_labels.shape[1]))
        ]
        return sum(losses) / len(losses) if losses else domain_labels.new_tensor(0.0).float()

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self):
        """One epoch: focal + KD (soft + feat) + domain CE + adversarial + ortho."""
        self.model.train()
        self.kd_projector.train()
        if self.adv_heads:
            self.adv_heads.train()

        # ---- Adversarial gate + lambda schedule ----------------------------
        epoch = getattr(self, 'current_epoch', 0)
        gate_open = self._prev_har_acc >= self.da_start_threshold

        p = epoch / max(1, self._total_epochs)
        lambda_adv = 2.0 / (1.0 + math.exp(-5.0 * p)) - 1.0

        if gate_open:
            if self._da_gate_opened_epoch is None:
                self._da_gate_opened_epoch = epoch
                print(
                    f"[DisentangledTrainer] adversarial gate opened at epoch {epoch} "
                    f"(train_acc={self._prev_har_acc:.3f}, λ_adv={lambda_adv:.4f})"
                )
            self.grl.lambda_ = 1.0
        else:
            self.grl.lambda_ = 0.0

        epoch_loss = epoch_accuracy = total_samples = total_time = 0.0

        for batch in self.train_loader:
            if len(batch) == 3:
                inputs, labels, domain_labels = batch
                domain_labels = domain_labels.to(self.device)
                has_domain = True
            else:
                inputs, labels = batch
                has_domain = False

            if inputs.size(0) == 0:
                continue

            batch_size = inputs.size(0)
            total_samples += batch_size
            inputs = inputs.to(self.device)

            # ---- Normalise labels ------------------------------------------
            if isinstance(labels, tuple):
                labels = labels[0]
            if isinstance(labels, str):
                try:
                    labels = torch.tensor([int(labels)] * batch_size, dtype=torch.long)
                except ValueError:
                    labels = torch.zeros(batch_size, dtype=torch.long)
            elif not hasattr(labels, 'shape') or len(labels.shape) == 0:
                labels = torch.tensor([int(labels)] * batch_size, dtype=torch.long)
            labels = labels.to(self.device).long()

            # ---- Input augmentation (gated by use_augmentation) -----------
            inputs = self._apply_input_augmentation(inputs)

            # ---- Teacher forward (frozen) -----------------------------------
            with torch.no_grad():
                t_logits, t_feat = self.teacher.forward_features(inputs)

            # ---- Student forward -------------------------------------------
            start = time.time()
            self.optimizer.zero_grad()

            motion_logits, motion_feat, domain_logits, domain_feat = \
                self.model.forward_full(inputs)

            # ---- Focal loss -------------------------------------------------
            task_loss = self._loss(motion_logits, labels)

            # ---- KD losses --------------------------------------------------
            kd_soft, feat_loss = self._kd_losses(
                motion_logits, t_logits, motion_feat, t_feat
            )

            # ---- Orthogonal loss --------------------------------------------
            orth_loss = self._ortho_loss(motion_feat, domain_feat)

            # ---- Domain losses (only when domain labels available) ----------
            dom_ce_loss = motion_feat.new_tensor(0.0)
            adv_loss = motion_feat.new_tensor(0.0)
            if has_domain:
                dom_ce_loss = self._domain_ce_loss(domain_logits, domain_labels)
                if gate_open and self.adv_heads:
                    adv_loss = self._domain_adv_loss(motion_feat, domain_labels)

            # ---- Combine ---------------------------------------------------
            loss = (
                (1.0 - self.kd_alpha) * task_loss
                + self.kd_alpha * kd_soft
                + self.kd_beta * feat_loss
                + self.lambda_domain_ce * dom_ce_loss
                + lambda_adv * adv_loss
                + self.lambda_ortho * orth_loss
            )

            loss.backward()
            all_params = (
                list(self.model.parameters())
                + list(self.kd_projector.parameters())
                + list(self.adv_heads.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.grad_clip_norm)
            self.optimizer.step()

            total_time += time.time() - start
            epoch_loss += loss.item() * batch_size

            preds = torch.argmax(motion_logits, dim=1)
            epoch_accuracy += (preds == labels).sum().item()

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        self._prev_har_acc = epoch_accuracy / total_samples
        return (
            epoch_loss / total_samples,
            epoch_accuracy / total_samples,
            total_time / total_samples,
        )
