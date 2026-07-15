"""
Loss functions for the enhanced reward model with C51, absolute progress,
and optional Huber fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation import AblationConfig


def _c51_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focal_gamma: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """C51 cross-entropy with optional focal weighting and optional
    per-sample reweighting.

    focal_gamma = 0 → standard soft-label CE.
    focal_gamma > 0 → (1 - p_t)^gamma * CE per-sample, sharpening rare-bin
      learning.
    sample_weights: (B,) per-batch-sample weight. When provided, per-token
      losses are scaled by the sample's weight before reduction.
    """
    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, n_bins)
    per_token = -(targets * log_probs).sum(dim=-1)     # (B, T)
    if focal_gamma > 0:
        probs = log_probs.exp()
        target_idx = targets.argmax(dim=-1, keepdim=True)
        p_t = probs.gather(-1, target_idx).squeeze(-1)
        per_token = (1.0 - p_t).clamp(min=0.0) ** focal_gamma * per_token
    if sample_weights is not None:
        # Broadcast (B,) weights over T. Normalize so batch mean stays 1 →
        # preserves effective learning rate.
        w = sample_weights / (sample_weights.mean() + 1e-8)
        per_token = per_token * w.view(-1, 1)
    return per_token.mean()


def soft_bins(values: torch.Tensor, bin_min: float, bin_max: float,
              n_bins: int, label_smoothing_sigma: float = 0.0) -> torch.Tensor:
    """C51-style soft bin targets: distribute probability mass between adjacent bins.

    Args:
        values: (*) arbitrary-shape tensor of continuous target values
        bin_min, bin_max: support range
        n_bins: number of discrete bins
        label_smoothing_sigma: if > 0, convolve the two-hot targets with a
            Gaussian of this sigma (in bins) for calibration regularization.
    Returns:
        (*, n_bins) soft target distributions
    """
    values = values.clamp(bin_min, bin_max)
    bin_width = (bin_max - bin_min) / (n_bins - 1)
    bin_idx = (values - bin_min) / bin_width
    lower = bin_idx.long().clamp(0, n_bins - 2)
    upper = lower + 1
    upper_w = bin_idx - lower.float()
    lower_w = 1.0 - upper_w
    soft = torch.zeros(*values.shape, n_bins, device=values.device)
    soft.scatter_(-1, lower.unsqueeze(-1), lower_w.unsqueeze(-1))
    soft.scatter_(-1, upper.unsqueeze(-1), upper_w.unsqueeze(-1))

    if label_smoothing_sigma > 0:
        # Gaussian kernel in bin-units; 1D conv along the last axis.
        radius = max(1, int(round(3 * label_smoothing_sigma)))
        k = torch.arange(-radius, radius + 1, device=values.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (k / label_smoothing_sigma) ** 2)
        kernel = kernel / kernel.sum()
        # (*, n_bins) → (B, 1, n_bins) for conv1d
        flat = soft.reshape(-1, 1, n_bins)
        smoothed = F.conv1d(flat, kernel.view(1, 1, -1), padding=radius)
        # renormalize (convolution with zero-padding slightly loses mass at edges)
        smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft = smoothed.reshape(soft.shape)
    return soft


class RMLoss:
    """
    Combined loss for the enhanced model. Computes:
      1. C51 cross-entropy on relative progress (or fallback to Curriculum Huber)
      2. C51 cross-entropy on absolute progress

    All components are gated by AblationConfig flags.
    """

    def __init__(self, config: AblationConfig, delta_min: float = 0.01,
                 delta_max: float = 0.30,
                 focal_gamma: float = 0.0, focal_on: str = "none",
                 tri_state_weight: float = 0.0, tri_state_eps: float = 0.02,
                 completion_weight: float = 0.0,
                 completion_focal_gamma: float = 2.0,
                 completion_pos_weight: float = 2.0,
                 smoothness_weight: float = 0.0,
                 c51_label_smoothing_sigma: float = 0.0):
        """
        focal_gamma: >0 enables focal-weighted C51 CE (ARM λ_succ style).
        focal_on: which C51 head(s) get focal weighting. One of:
            "none", "rel", "abs", "both"
        tri_state_weight: >0 enables auxiliary 3-way sign head on Δlabel.
        tri_state_eps: dead-band; |Δlabel| < eps → sign class = 0 (stagnant).
        completion_weight: >0 enables binary 'is final frame' aux loss.
        completion_focal_gamma: γ for focal BCE on completion head.
        completion_pos_weight: alpha-like weight on the positive (is_final) class.
        """
        self.config = config
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.focal_gamma = focal_gamma
        self.focal_on = focal_on
        self.tri_state_weight = tri_state_weight
        self.tri_state_eps = tri_state_eps
        self.completion_weight = completion_weight
        self.completion_focal_gamma = completion_focal_gamma
        self.completion_pos_weight = completion_pos_weight
        self.smoothness_weight = smoothness_weight
        self.c51_label_smoothing_sigma = c51_label_smoothing_sigma

    def __call__(
        self,
        # Model outputs
        progress_preds: torch.Tensor,     # (B, T) scalar predictions
        backbone_out: torch.Tensor,       # (B, T, d_model)
        rel_logits: torch.Tensor | None,  # (B, T, n_rel_bins) — None if baseline
        model: nn.Module,
        # Targets
        labels: torch.Tensor,             # (B, T) relative progress labels
        abs_labels: torch.Tensor | None,  # (B, T) absolute progress labels
        completion_targets: torch.Tensor | None = None,  # (B, T) binary 0/1
        # Training state
        current_lr: float = 1e-4,
        base_lr: float = 1e-4,
        # Per-sample loss reweighting (online-Q "true streaming" curriculum)
        sample_weights: torch.Tensor | None = None,   # (B,) per-sample weight
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Returns:
            (total_loss, loss_components_dict) where dict has per-component values for logging.
        """
        components = {}

        # ── Loss 1: Relative progress ───────────────────────────────────────
        if self.config.use_c51 and rel_logits is not None:
            rel_targets = soft_bins(
                labels, model.rel_bin_min, model.rel_bin_max, model.n_rel_bins,
                label_smoothing_sigma=self.c51_label_smoothing_sigma,
            )
            use_focal_rel = self.focal_gamma > 0 and self.focal_on in ("rel", "both")
            loss_progress = _c51_ce(
                rel_logits, rel_targets,
                focal_gamma=self.focal_gamma if use_focal_rel else 0.0,
                sample_weights=sample_weights,
            )
        else:
            # Fallback: Curriculum Huber (original WARP-RM)
            lr_frac = current_lr / base_lr
            delta = self.delta_min * (self.delta_max / self.delta_min) ** lr_frac
            if sample_weights is None:
                loss_progress = F.huber_loss(progress_preds, labels, delta=delta)
            else:
                # Per-sample weighting on Huber as well.
                per_token = F.huber_loss(progress_preds, labels, delta=delta, reduction="none")
                w = sample_weights / (sample_weights.mean() + 1e-8)
                loss_progress = (per_token * w.view(-1, 1)).mean()
        components["progress"] = loss_progress.item()

        total = loss_progress

        # ── Loss 2: Absolute progress ──────────────────────────────────────
        loss_abs = torch.tensor(0.0, device=labels.device)
        if self.config.use_abs_progress and abs_labels is not None:
            abs_logits = model.abs_progress_head(backbone_out)
            abs_targets = soft_bins(
                abs_labels, 0.0, 1.0, model.n_abs_bins,
                label_smoothing_sigma=self.c51_label_smoothing_sigma,
            )
            use_focal_abs = self.focal_gamma > 0 and self.focal_on in ("abs", "both")
            loss_abs = _c51_ce(abs_logits, abs_targets, focal_gamma=self.focal_gamma if use_focal_abs else 0.0)
            total = total + self.config.abs_progress_weight * loss_abs
        components["abs"] = loss_abs.item()

        # ── Loss 3: Tri-state sign aux (ARM core) ──────────────────────────
        # Per-step Δlabel → {−1, 0, +1} with dead-band ε_stagnant.
        # Targets placed at position t (not t+1) so the causal transformer
        # predicts the *forward* delta from position t's feature.
        loss_tri = torch.tensor(0.0, device=labels.device)
        if self.tri_state_weight > 0 and hasattr(model, "tri_state_head"):
            tri_logits = model.tri_state_head(backbone_out)  # (B, T, 3)
            # Δlabel[:, t] = label[:, t+1] - label[:, t]; last token has no target
            d = labels[:, 1:] - labels[:, :-1]  # (B, T-1)
            tri_targets = torch.zeros_like(d, dtype=torch.long)  # class 1 = stagnant
            tri_targets[d > self.tri_state_eps] = 2   # class 2 = forward
            tri_targets[d < -self.tri_state_eps] = 0  # class 0 = backward
            tri_logits_trim = tri_logits[:, :-1, :]   # align with targets
            loss_tri = F.cross_entropy(
                tri_logits_trim.reshape(-1, 3),
                tri_targets.reshape(-1),
            )
            total = total + self.tri_state_weight * loss_tri
        components["tri_state"] = loss_tri.item()

        # ── Loss 4: Completion aux (ARM λ_succ terminal anchor) ─────────────
        loss_completion = torch.tensor(0.0, device=labels.device)
        if (self.completion_weight > 0 and hasattr(model, "completion_head")
                and completion_targets is not None):
            logits = model.completion_head(backbone_out).squeeze(-1)  # (B, T)
            targets_f = completion_targets.float()
            # Focal BCE with positive-class weight (α).
            bce = F.binary_cross_entropy_with_logits(
                logits, targets_f,
                reduction="none",
                pos_weight=torch.tensor(self.completion_pos_weight, device=logits.device),
            )
            if self.completion_focal_gamma > 0:
                probs = torch.sigmoid(logits)
                p_t = torch.where(targets_f > 0.5, probs, 1.0 - probs)
                bce = (1.0 - p_t).clamp(min=0.0) ** self.completion_focal_gamma * bce
            loss_completion = bce.mean()
            total = total + self.completion_weight * loss_completion
        components["completion"] = loss_completion.item()

        # ── Loss 5: Temporal smoothness regularizer ────────────────────────
        # Penalize |ŷ_t - ŷ_{t-1}| only when the *label* is also smooth there:
        # if the GT itself jumps (e.g. mid-flip boundary or rewind→forward
        # transition), the model should jump too. Huber kernel on the residual
        # of (pred_delta - label_delta) rather than on raw pred deltas.
        loss_smooth = torch.tensor(0.0, device=labels.device)
        if self.smoothness_weight > 0:
            pred_d = progress_preds[:, 1:] - progress_preds[:, :-1]
            label_d = labels[:, 1:] - labels[:, :-1]
            loss_smooth = F.huber_loss(pred_d, label_d, delta=0.05)
            total = total + self.smoothness_weight * loss_smooth
        components["smoothness"] = loss_smooth.item()

        return total, components
