import torch
import torch.nn.functional as F


def zinb_loss(x, mean, theta, pi_logits, eps=1e-8):
    # Compute the ZINB term in fp32 for numerical stability under AMP.
    x = x.float()
    mean = torch.clamp(mean.float(), min=eps, max=1e6)
    theta = torch.clamp(theta.float(), min=eps, max=1e6)
    pi_logits = pi_logits.float()

    # NB log-likelihood
    log_theta = torch.log(theta + eps)
    log_mean = torch.log(mean + eps)
    t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
    t2 = theta * (log_theta - torch.log(theta + mean + eps))
    t3 = x * (log_mean - torch.log(theta + mean + eps))
    log_nb = t1 + t2 + t3

    # Zero-inflated component via numerically stable logsumexp
    log_pi = F.logsigmoid(pi_logits)
    log_1_pi = F.logsigmoid(-pi_logits)

    # x == 0: log(pi + (1 - pi) * NB(0))
    log_nb_zero = theta * (log_theta - torch.log(theta + mean + eps))
    log_zero = torch.logsumexp(
        torch.stack([log_pi, log_1_pi + log_nb_zero], dim=0), dim=0,
    )
    # x > 0: log(1 - pi) + log(NB(x))
    log_nonzero = log_1_pi + log_nb

    mask = (x < 1e-8).float()
    nll = -(mask * log_zero + (1 - mask) * log_nonzero)

    return nll.mean()


def mse_log1p_loss(x_true, pred_mu, eps=1e-8, mu_max: float = 1e6):
    """MSE on log1p-transformed counts: fair baseline for a ZINB head.

    Takes the model's Softplus ``pred_mu`` as a non-negative point estimate
    and computes ``MSE(log1p(mu), log1p(x))``.  Matching the log-scale of
    counts keeps the loss well-conditioned for sparse single-cell data while
    remaining a purely L2 objective.

    ``pred_mu`` is clamped into ``[eps, mu_max]`` before ``log1p`` to avoid
    Inf / NaN under AMP: Softplus can overflow to +Inf with fp16 activations
    late in training, which then cascades through ``log1p`` into non-finite
    losses and validation skips.  Clamping mirrors the safety range used by
    ``zinb_loss``.
    """
    x_true = x_true.float().clamp_min(0.0)
    pred_mu = torch.clamp(pred_mu.float(), min=eps, max=mu_max)
    return F.mse_loss(torch.log1p(pred_mu), torch.log1p(x_true))


def full_loss(i_emb, g_emb, pred_mu, pred_theta, pred_pi, gene_input, t_img, t_gen,
              w_contrast=0.1, w_align=0.1, w_zinb=1.0, loss_mode="zinb"):
    """Weighted sum of contrastive + alignment + reconstruction losses.

    ``loss_mode='zinb'`` uses the proper ZINB log-likelihood (the default and
    the setting used for the main paper results).  ``loss_mode='mse'`` swaps
    the reconstruction term for an MSE on ``log1p`` counts, which is used
    as a non-probabilistic baseline; ``w_zinb`` is reused as the weight of
    the MSE term in that case.
    """
    i_emb = i_emb.float()
    g_emb = g_emb.float()
    pred_mu = pred_mu.float()
    pred_theta = pred_theta.float()
    pred_pi = pred_pi.float()
    gene_input = gene_input.float()

    B = i_emb.size(0)
    G = gene_input.size(1) // 2
    x_true = gene_input[:, :G]

    # 1. Contrastive (CLIP-style symmetric cross-entropy)
    T_img = torch.clamp(t_img.float(), min=0.01, max=5.0)
    T_gen = torch.clamp(t_gen.float(), min=0.01, max=5.0)

    logits_ig = i_emb @ g_emb.T / T_img
    logits_gi = g_emb @ i_emb.T / T_gen
    labels = torch.arange(B, device=i_emb.device)

    c_loss = (F.cross_entropy(logits_ig, labels) + F.cross_entropy(logits_gi, labels)) / 2

    # 2. Smooth-L1 alignment (more robust to outlier embeddings than MSE)
    a_loss = F.smooth_l1_loss(i_emb, g_emb)

    # 3. Reconstruction (ZINB by default, MSE-on-log1p for the baseline).
    if loss_mode == "zinb":
        z_loss = zinb_loss(x_true, pred_mu, pred_theta, pred_pi)
    elif loss_mode == "mse":
        z_loss = mse_log1p_loss(x_true, pred_mu)
        # MSE only uses pred_mu; attach pred_theta and pred_pi to the graph
        # with zero weight so the ZINB head parameters still receive a
        # (zero) gradient.  Without this DDP reports "parameters that were
        # not used in producing loss" even when find_unused_parameters=True
        # is set, because the learned contrastive temperatures then
        # interact badly with the extra autograd hooks.
        z_loss = z_loss + 0.0 * (pred_theta.sum() + pred_pi.sum())
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    total = w_contrast * c_loss + w_align * a_loss + w_zinb * z_loss

    return total, c_loss.item(), a_loss.item(), z_loss.item()
