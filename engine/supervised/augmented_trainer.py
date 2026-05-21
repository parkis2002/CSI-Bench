"""
AugmentedTaskTrainer — TaskTrainer subclass with online data augmentation.

Augmentations ported from: https://github.com/harikang/prunedAttentionGRU
  1. Multiplicative Gaussian noise  (applied per-batch, randomly)
  2. Temporal shift via torch.roll  (applied per-batch, randomly)
  3. Mixup                          (applied per-batch, randomly)

Shared improvements (applied to all models using this trainer):
  4. Amplitude scaling              (applied per-batch, randomly)
  5. Inverse-frequency class weights (α_t, computed once from training data)
  6. Weighted Focal Loss            (default γ=2): L = -α_t (1-p_t)^γ log(p_t)
     The modulating factor (1-p_t)^γ down-weights easy majority-class samples
     and amplifies gradient from hard minority samples.  Replaces CE + label
     smoothing.  Validation uses plain CE for unbiased loss monitoring.

Per-model optional improvements:
  7. Subcarrier dropout             (for DeepBiGRU domain generalisation)
  8. Configurable gradient clipping (for CsiConformer stability)

All augmentations are probabilistic and applied independently per batch.
The evaluation loop is inherited from TaskTrainer so all CSI-Bench metrics
(accuracy, F1, cross-domain splits) work without modification.
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.supervised.task_trainer import TaskTrainer


class AugmentedTaskTrainer(TaskTrainer):
    """TaskTrainer extended with online augmentation and weighted focal loss.

    Augmentation params
    -------------------
    noise_prob            : probability of multiplicative Gaussian noise per batch
    shift_prob            : probability of circular temporal shift per batch
    amplitude_prob        : probability of per-sample amplitude scaling per batch
    subcarrier_dropout_prob: fraction of subcarrier channels zeroed at random
                            per batch; simulates antenna/hardware differences
                            across devices (use for DeepBiGRU, default 0.0)
    mixup_prob            : probability of mixup within a batch

    Loss params
    -----------
    use_class_weights : compute inverse-frequency α_t weights from train_loader
                        for focal loss (default True)
    label_smoothing   : ε for CrossEntropyLoss label smoothing (used only when
                        focal_gamma == 0, i.e. plain CE fallback)
    focal_gamma       : γ for focal loss (default 2.0).
                        L(p_t) = -α_t (1-p_t)^γ log(p_t)
                        Set to 0.0 to fall back to CE + label smoothing.
                        Validation uses plain CE regardless, for unbiased
                        loss monitoring.

    Training params
    ---------------
    grad_clip_norm : max gradient norm for clip_grad_norm_ (default 1.0).
                     Set to 0.5 for CsiConformer — the conformer conv module
                     produces large gradient norms that destabilise training.

    BCE tasks are detected automatically; all loss improvements are skipped
    to preserve backwards compatibility.
    """

    def __init__(
        self,
        *args,
        mixup_prob: float = 0.3,
        noise_prob: float = 0.5,
        noise_scale: float = 0.05,
        shift_prob: float = 0.5,
        max_shift: int = 10,
        mixup_alpha: float = 1.0,
        amplitude_prob: float = 0.5,
        amplitude_lo: float = 0.7,
        amplitude_hi: float = 1.3,
        subcarrier_dropout_prob: float = 0.0,
        label_smoothing: float = 0.1,
        use_class_weights: bool = True,
        focal_gamma: float = 2.0,
        grad_clip_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mixup_prob = mixup_prob
        self.noise_prob = noise_prob
        self.noise_scale = noise_scale
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self.mixup_alpha = mixup_alpha
        self.amplitude_prob = amplitude_prob
        self.amplitude_lo = amplitude_lo
        self.amplitude_hi = amplitude_hi
        self.subcarrier_dropout_prob = subcarrier_dropout_prob
        self.focal_gamma = focal_gamma
        self.grad_clip_norm = grad_clip_norm
        self._class_weights = None

        self._is_bce = isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss))

        if not self._is_bce and self.num_classes is not None:
            if use_class_weights:
                self._class_weights = self._compute_class_weights()

            if focal_gamma > 0:
                # Validation uses plain CE for unbiased val_loss monitoring.
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.CrossEntropyLoss(
                    weight=self._class_weights,
                    label_smoothing=label_smoothing,
                )

    # ------------------------------------------------------------------
    # Class-weight computation
    # ------------------------------------------------------------------

    def _compute_class_weights(self) -> torch.Tensor:
        """Scan train_loader once to build inverse-frequency class weights α_t.

        w_c = total / (num_classes * count_c), normalised so min(w) = 1.
        Handles both 2-tuple (csi, labels) and 3-tuple (csi, labels, domain)
        batches so it works when called from DomainAdversarialTrainer.
        """
        counts = torch.zeros(self.num_classes, dtype=torch.float)
        for batch in self.train_loader:
            # Accept 2-tuple (csi, labels) or 3-tuple (csi, labels, domain)
            labels = batch[1] if isinstance(batch, (list, tuple)) else batch
            if isinstance(labels, tuple):
                labels = labels[0]
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            labels = labels.view(-1).long().cpu()
            for c in range(self.num_classes):
                counts[c] += (labels == c).sum().item()

        total = counts.sum()
        weights = total / (self.num_classes * counts.clamp(min=1.0))
        # Normalise so the majority class (smallest weight) = 1.0, then cap at
        # 10× to prevent a near-empty phantom class from dominating training.
        weights = weights / weights.min()
        weights = weights.clamp(max=10.0)
        print(f"[AugmentedTaskTrainer] class weights: {weights.tolist()}")
        return weights.to(self.device)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted focal loss: L(p_t) = -α_t (1-p_t)^γ log(p_t).

        p_t is derived from unweighted CE so the modulating factor correctly
        reflects the model's raw confidence (not scaled by α_t).
        Falls back to self.criterion (plain CE or CE+smoothing) when γ=0.
        """
        if self._is_bce:
            return self.criterion(logits, labels)
        if self.focal_gamma > 0:
            ce = F.cross_entropy(logits, labels, reduction='none')   # -log(p_t), [B]
            p_t = torch.exp(-ce)                                     # p_t, [B]
            alpha_t = (self._class_weights[labels] if self._class_weights is not None
                       else torch.ones_like(p_t))
            return (alpha_t * (1.0 - p_t) ** self.focal_gamma * ce).mean()
        return self.criterion(logits, labels)

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _apply_input_augmentation(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply all per-batch input augmentations except mixup.

        Applies noise, temporal shift, amplitude scaling, and subcarrier
        dropout in that order.  Each augmentation fires independently with
        its own probability.  Used by both train_epoch and KD subclasses.
        """
        if random.random() < self.noise_prob:
            inputs = self._gaussian_noise(inputs)
        if random.random() < self.shift_prob:
            inputs = self._temporal_shift(inputs)
        if random.random() < self.amplitude_prob:
            inputs = self._amplitude_scale(inputs)
        if self.subcarrier_dropout_prob > 0:
            inputs = self._subcarrier_dropout(inputs)
        return inputs

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Multiplicative Gaussian noise: x_aug = x + noise_scale * x * N(0, 1).

        The original formula (x + x*N(0,1)) added noise with std=|x|, which is
        large enough to flip signs (~16% probability per element) and destroys
        sequential temporal patterns.  noise_scale (default 0.05) keeps the
        perturbation meaningful but well below the signal magnitude.
        """
        return x + self.noise_scale * x * torch.randn_like(x)

    def _temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Circular shift along time axis by a random number of steps.

        Time axis: dim 2 for 4-D [B, C, T, F], dim 1 for 3-D [B, T, F].
        """
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        time_dim = 2 if x.dim() == 4 else 1
        return torch.roll(x, shifts=shift, dims=time_dim)

    def _amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample random amplitude scaling drawn from Uniform(amplitude_lo, amplitude_hi).

        Different devices and environments produce CSI at different signal
        strengths due to hardware gain and path-loss differences.  Random
        amplitude scaling discourages the model from using absolute amplitude
        as a discriminative feature, improving cross-device/env generalisation.
        """
        shape = (x.size(0),) + (1,) * (x.dim() - 1)  # broadcast over non-batch dims
        scale = x.new_empty(shape).uniform_(self.amplitude_lo, self.amplitude_hi)
        return x * scale

    def _subcarrier_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a random fraction of subcarrier (feature) channels.

        CSI has 232 subcarrier channels.  Different devices and antennas have
        different noise floors and hardware-specific artefacts per subcarrier.
        Randomly dropping channels forces the model to build representations
        that are robust to missing or corrupt subcarriers, improving
        cross-device generalisation for large models like DeepBiGRU.
        """
        if self.subcarrier_dropout_prob <= 0.0:
            return x
        # Feature dim: -1 (last dim) for both [B, 1, T, F] and [B, T, F]
        n_feat = x.size(-1)
        n_drop = int(n_feat * self.subcarrier_dropout_prob)
        if n_drop == 0:
            return x
        drop_idx = torch.randperm(n_feat, device=x.device)[:n_drop]
        x = x.clone()
        x[..., drop_idx] = 0.0
        return x

    def _mixup(self, x: torch.Tensor, labels: torch.Tensor):
        """Apply mixup between pairs of samples in the batch.

        Returns:
            x_mixed, labels_a, labels_b, lam
        """
        rand_idx = torch.randperm(x.size(0), device=x.device)
        labels_a = labels
        labels_b = labels[rand_idx]
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        x_mixed = lam * x + (1.0 - lam) * x[rand_idx]
        return x_mixed, labels_a, labels_b, lam

    # ------------------------------------------------------------------
    # Training epoch (override)
    # ------------------------------------------------------------------

    def train_epoch(self):
        """One epoch with noise, shift, amplitude scaling, subcarrier dropout,
        and mixup augmentation.

        Returns:
            (epoch_loss, epoch_accuracy, time_per_sample)
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0
        total_time = 0.0

        for inputs, labels in self.train_loader:
            if inputs.size(0) == 0:
                continue

            batch_size = inputs.size(0)
            total_samples += batch_size

            inputs = inputs.to(self.device)

            # ---- normalise labels to a long tensor -----------------------
            if isinstance(labels, tuple):
                labels = labels[0]
            if isinstance(labels, str):
                try:
                    labels = torch.tensor(
                        [int(labels)] * batch_size, dtype=torch.long
                    )
                except ValueError:
                    labels = torch.zeros(batch_size, dtype=torch.long)
            elif not hasattr(labels, "shape") or len(labels.shape) == 0:
                labels = torch.tensor(
                    [int(labels)] * batch_size, dtype=torch.long
                )

            labels = labels.to(self.device)

            # ---- online augmentation (noise, shift, amplitude, subcarrier) ---
            inputs = self._apply_input_augmentation(inputs)
            apply_mixup = random.random() < self.mixup_prob

            # ---- forward pass -------------------------------------------
            start = time.time()
            self.optimizer.zero_grad()

            if apply_mixup and batch_size > 1:
                inputs_m, labels_a, labels_b, lam = self._mixup(inputs, labels)
                outputs = self.model(inputs_m)

                if self._is_bce:
                    loss = (
                        lam * self.criterion(outputs, F.one_hot(labels_a, self.num_classes).float())
                        + (1.0 - lam) * self.criterion(outputs, F.one_hot(labels_b, self.num_classes).float())
                    )
                else:
                    loss = lam * self._loss(outputs, labels_a) + (1.0 - lam) * self._loss(outputs, labels_b)

                preds = torch.argmax(outputs, dim=1)
                correct = (
                    lam * (preds == labels_a).float()
                    + (1.0 - lam) * (preds == labels_b).float()
                ).sum().item()
            else:
                outputs = self.model(inputs)
                if self._is_bce:
                    labels_oh = F.one_hot(labels, self.num_classes).float()
                    loss = self.criterion(outputs, labels_oh)
                else:
                    loss = self._loss(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                correct = (preds == labels).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
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
