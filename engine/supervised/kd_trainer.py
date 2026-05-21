"""
KnowledgeDistillationTrainer — AugmentedTaskTrainer subclass that trains a
student model under the supervision of a frozen teacher (ResNet18CSI).

Loss (per batch):
    L = (1 - kd_alpha) * L_CE  +  kd_alpha * T² * L_KL  +  kd_beta * L_feat

    L_CE   : weighted focal cross-entropy on student logits
             (from AugmentedTaskTrainer._loss — class weights, focal gamma)
    L_KL   : KL divergence between temperature-scaled student and teacher
             soft-label distributions (Hinton et al., "Distilling the
             Knowledge in a Neural Network", NeurIPS 2015 Workshop)
    L_feat : MSE between a learned linear projection of the student backbone
             feature and the teacher backbone feature (feature-based matching)

Projection head:
    A single nn.Linear(student_backbone_dim → 512) is created for each student
    and its parameters are added to the existing optimizer's param groups so
    they share the same LR schedule and weight decay.  This projects the
    student's pre-head embedding to the teacher's 512-dim embedding space
    without adding capacity to the student at inference time (the projector is
    only used during training).

Teacher interface:
    The teacher must expose forward_features(x) → (logits, feat [B, 512]).
    ResNet18CSI satisfies this.  The teacher is moved to device and frozen
    (eval mode, no gradient).

Student interface:
    The student must expose:
      - forward_features(x) → (logits, feat [B, backbone_dim])
      - backbone_dim attribute (int)
    InceptionCSI, LightTCN, and DeepBiGRU satisfy both.
"""

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.supervised.augmented_trainer import AugmentedTaskTrainer


class KnowledgeDistillationTrainer(AugmentedTaskTrainer):
    """AugmentedTaskTrainer extended with soft-target + feature KD from a teacher.

    Parameters
    ----------
    teacher : nn.Module
        Frozen teacher model. Must implement forward_features(x) → (logits, feat).
    kd_temperature : float
        Softmax temperature T for soft-target distillation. Higher T produces
        softer distributions. Typical range: 2–8 (default 4.0).
    kd_alpha : float
        Weight of the soft-target KD loss. Task CE weight is (1 - kd_alpha).
        Set to 0.0 to disable soft-target distillation (default 0.5).
    kd_beta : float
        Weight of the feature matching MSE loss. Set to 0.0 to disable
        feature matching (default 0.1).
    """

    TEACHER_FEAT_DIM = 512  # ResNet18CSI backbone output dimension

    def __init__(
        self,
        *args,
        teacher: nn.Module,
        kd_temperature: float = 4.0,
        kd_alpha: float = 0.5,
        kd_beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not hasattr(self.model, 'backbone_dim'):
            raise AttributeError(
                f"{type(self.model).__name__} must define a 'backbone_dim' attribute "
                f"for knowledge distillation feature matching."
            )
        if not hasattr(self.model, 'forward_features'):
            raise AttributeError(
                f"{type(self.model).__name__} must implement forward_features(x) "
                f"returning (logits, feat) for knowledge distillation."
            )

        # Freeze and move teacher to device
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher.to(self.device)

        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha
        self.kd_beta = kd_beta

        # Linear projection: student backbone dim → teacher backbone dim (512).
        # Only used during training; not part of the student at inference time.
        self.kd_projector = nn.Linear(
            self.model.backbone_dim, self.TEACHER_FEAT_DIM
        ).to(self.device)

        # Add projector params to the existing optimizer so they share the same
        # LR schedule and weight decay as the student backbone.
        self.optimizer.add_param_group({
            'params': self.kd_projector.parameters(),
        })

        print(
            f"[KnowledgeDistillationTrainer] "
            f"student_dim={self.model.backbone_dim} → teacher_dim={self.TEACHER_FEAT_DIM}, "
            f"T={kd_temperature}, alpha={kd_alpha}, beta={kd_beta}"
        )

    # ------------------------------------------------------------------
    # KD loss components
    # ------------------------------------------------------------------

    def _kd_losses(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
    ) -> tuple:
        """Compute soft-target KL and feature MSE losses (unscaled).

        Returns (kd_soft_loss, feat_loss).  Caller applies kd_alpha / kd_beta.
        """
        T = self.kd_temperature

        # Soft-target KL divergence (Hinton 2015): T² restores gradient magnitude
        s_log_soft = F.log_softmax(s_logits / T, dim=-1)
        t_soft = F.softmax(t_logits / T, dim=-1).detach()
        kd_soft = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (T * T)

        # Feature matching MSE via learned projector
        s_proj = self.kd_projector(s_feat)
        feat_loss = F.mse_loss(s_proj, t_feat.detach())

        return kd_soft, feat_loss

    # ------------------------------------------------------------------
    # Training epoch (override)
    # ------------------------------------------------------------------

    def train_epoch(self):
        """One epoch: augmentation + combined CE / soft-target KD / feature MSE loss."""
        self.model.train()
        self.kd_projector.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0
        total_time = 0.0

        for batch in self.train_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch

            if inputs.size(0) == 0:
                continue

            batch_size = inputs.size(0)
            total_samples += batch_size

            inputs = inputs.to(self.device)

            # ---- normalise labels ----------------------------------------
            if isinstance(labels, tuple):
                labels = labels[0]
            if isinstance(labels, str):
                try:
                    labels = torch.tensor(
                        [int(labels)] * batch_size, dtype=torch.long
                    )
                except ValueError:
                    labels = torch.zeros(batch_size, dtype=torch.long)
            elif not hasattr(labels, 'shape') or len(labels.shape) == 0:
                labels = torch.tensor(
                    [int(labels)] * batch_size, dtype=torch.long
                )
            labels = labels.to(self.device).long()

            # ---- input augmentation (noise, shift, amplitude, dropout) ----
            inputs = self._apply_input_augmentation(inputs)
            apply_mixup = random.random() < self.mixup_prob

            # ---- teacher forward (frozen, no gradient) --------------------
            with torch.no_grad():
                t_logits, t_feat = self.teacher.forward_features(inputs)

            # ---- student forward + loss -----------------------------------
            start = time.time()
            self.optimizer.zero_grad()

            if apply_mixup and batch_size > 1:
                inputs_m, labels_a, labels_b, lam = self._mixup(inputs, labels)
                s_logits, s_feat = self.model.forward_features(inputs_m)

                task_loss = (
                    lam * self._loss(s_logits, labels_a)
                    + (1.0 - lam) * self._loss(s_logits, labels_b)
                )

                # Teacher sees the same mixed input for KD
                with torch.no_grad():
                    t_logits_m, t_feat_m = self.teacher.forward_features(inputs_m)

                kd_soft, feat_loss = self._kd_losses(
                    s_logits, t_logits_m, s_feat, t_feat_m
                )

                preds = torch.argmax(s_logits, dim=1)
                correct = (
                    lam * (preds == labels_a).float()
                    + (1.0 - lam) * (preds == labels_b).float()
                ).sum().item()
            else:
                s_logits, s_feat = self.model.forward_features(inputs)
                task_loss = self._loss(s_logits, labels)
                kd_soft, feat_loss = self._kd_losses(
                    s_logits, t_logits, s_feat, t_feat
                )

                preds = torch.argmax(s_logits, dim=1)
                correct = (preds == labels).sum().item()

            loss = (
                (1.0 - self.kd_alpha) * task_loss
                + self.kd_alpha * kd_soft
                + self.kd_beta * feat_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.kd_projector.parameters()),
                max_norm=self.grad_clip_norm,
            )
            self.optimizer.step()

            total_time += time.time() - start
            epoch_loss += loss.item() * batch_size
            epoch_accuracy += correct

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        return (
            epoch_loss / total_samples,
            epoch_accuracy / total_samples,
            total_time / total_samples,
        )
