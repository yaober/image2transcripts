# =========================================================
#  train_clip_model.py  (NVRTC‑free version)
# =========================================================
"""
Key changes
===========
1. **GradScaler / autocast** switch to new `torch.amp` API (no deprecation warnings).
2. **GPU ➜ CPU fallback** in `zinb_nll` avoids the lgamma kernel that triggers the
   `libnvrtc‑builtins.so` dependency; only this tiny step runs on CPU, everything
   else remains on GPU.
3. Helper `_zinb_nll_cpu` keeps the analytical form identical to the original
   implementation, so metrics and convergence are unchanged.
"""
import math
import os
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # ✅ new API

# -----------  ZINB negative log‑likelihood  --------------

def _zinb_nll_cpu(x: torch.Tensor,
                  mu: torch.Tensor,
                  theta: torch.Tensor,
                  pi: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """Pure‑CPU implementation to dodge NVRTC lgamma JIT on GPU."""
    theta = torch.clamp(theta, min=eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)

    # NB component
    nb_case = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x     * (torch.log(mu    + eps) - log_theta_mu_eps)
    )

    # zero‑inflation mixture
    zero_mask = (x < 1e-8)
    nll = -torch.where(
        zero_mask,
        torch.log(pi + (1.0 - pi) * torch.exp(nb_case) + eps),
        torch.log(1.0 - pi + eps) + nb_case,
    )
    return nll.mean()


def zinb_nll(x: torch.Tensor,
             mu: torch.Tensor,
             theta: torch.Tensor,
             pi: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    """GPU‑safe wrapper: moves tensors to CPU for lgamma, then back."""
    dev = x.device
    nll_cpu = _zinb_nll_cpu(x.float().cpu(),
                            mu.float().cpu(),
                            theta.float().cpu(),
                            pi.float().cpu(),
                            eps)
    return nll_cpu.to(dev)

# -----------  full multi‑task loss  ----------------------

def full_loss(i_emb: torch.Tensor,
              g_emb: torch.Tensor,
              zinb_params: tuple,
              gene_input: torch.Tensor,
              t_img: torch.Tensor,
              t_gen: torch.Tensor,
              w_align: float = 0.1,
              w_zinb: float = 0.5):
    """
    Args:
        i_emb      : [B, D] image embedding
        g_emb      : [B, D] gene embedding
        zinb_params: (mu, theta, pi) from gene decoder
        gene_input : [B, 2G] input features (raw + logFC)
        t_img, t_gen: scalar contrastive temperature (learned)
        w_align    : weight for alignment loss
        w_zinb     : weight for ZINB loss

    Returns:
        total_loss, contrastive_loss, alignment_loss, zinb_loss
    """
    B = i_emb.size(0)
    G = gene_input.size(1) // 2
    x_raw = gene_input[:, :G]

    # 1. Contrastive (image ⇄ gene)
    T_img = F.softplus(t_img)
    T_gen = F.softplus(t_gen)

    logits_ig = i_emb @ g_emb.T / T_img
    logits_gi = g_emb @ i_emb.T / T_gen
    labels = torch.arange(B, device=i_emb.device)

    c_loss = (
        F.cross_entropy(logits_ig, labels) +
        F.cross_entropy(logits_gi, labels)
    ) / 2.0

    # 2. Alignment (same-sample L2)
    a_loss = F.mse_loss(i_emb, g_emb)

    # 3. ZINB reconstruction loss
    mu, theta, pi = zinb_params
    z_loss = zinb_nll(x_raw, mu, theta, pi)

    # 4. Total loss
    total = c_loss + w_align * a_loss + w_zinb * z_loss
    return total, c_loss.item(), a_loss.item(), z_loss.item()
