import torch
import torch.nn.functional as F


def zinb_loss(x, mean, theta, pi_logits, eps=1e-8):
    mean = torch.clamp(mean, min=eps, max=1e6)
    theta = torch.clamp(theta, min=eps, max=1e6)

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


def full_loss(i_emb, g_emb, pred_mu, pred_theta, pred_pi, gene_input, t_img, t_gen,
              w_contrast=0.1, w_align=0.1, w_zinb=1.0):

    B = i_emb.size(0)
    G = gene_input.size(1) // 2
    x_true = gene_input[:, :G]

    # 1. Contrastive (CLIP-style symmetric cross-entropy)
    T_img = torch.clamp(t_img, min=0.01, max=5.0)
    T_gen = torch.clamp(t_gen, min=0.01, max=5.0)

    logits_ig = i_emb @ g_emb.T / T_img
    logits_gi = g_emb @ i_emb.T / T_gen
    labels = torch.arange(B, device=i_emb.device)

    c_loss = (F.cross_entropy(logits_ig, labels) + F.cross_entropy(logits_gi, labels)) / 2

    # 2. Smooth-L1 alignment (more robust to outlier embeddings than MSE)
    a_loss = F.smooth_l1_loss(i_emb, g_emb)

    # 3. ZINB reconstruction
    z_loss = zinb_loss(x_true, pred_mu, pred_theta, pred_pi)

    total = w_contrast * c_loss + w_align * a_loss + w_zinb * z_loss

    if torch.isnan(total):
        print("NaN detected in loss!")
        return torch.tensor(0.0, device=total.device, requires_grad=True), 0, 0, 0

    return total, c_loss.item(), a_loss.item(), z_loss.item()