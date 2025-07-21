# =========================================================
#  train.py   (dual-ZINB full loss with stability fix)
# =========================================================
import torch
import torch.nn.functional as F

# -----------  ZINB negative log-likelihood (stable) --------------
def _zinb_nll_cpu(x, mu, theta, pi, eps=1e-8):
    x = torch.clamp(x, min=1e-4, max=100.0)  # Avoid zeros and NaNs
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)
    pi = torch.clamp(pi, min=eps, max=1.0 - eps)

    log_theta_mu = torch.log(theta + mu + eps)
    nb = (
        torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )
    zero_mask = (x < 1e-3)
    nll = -torch.where(
        zero_mask,
        torch.log(pi + (1.0 - pi) * torch.exp(nb) + eps),
        torch.log(1.0 - pi + eps) + nb,
    )
    return nll.mean()

def zinb_nll(x, mu, theta, pi):
    dev = x.device
    return _zinb_nll_cpu(x.float().cpu(), mu.float().cpu(), theta.float().cpu(), pi.float().cpu()).to(dev)

# -----------  full multi-task loss (w/ sanity check) ----------------------
def full_loss(i_emb, g_emb, pred_i, pred_g, gene_input, t_img, t_gen,
              w_contrast=0.2, w_align=0.05, w_mse_gene=1.0, w_mse_img=0.1, w_embed=0.1):
    B = i_emb.size(0)
    G = gene_input.size(1) // 2
    x_raw = gene_input[:, :G]

    # Contrastive loss
    T_img = torch.clamp(F.softplus(t_img), min=1e-3, max=10.0)
    T_gen = torch.clamp(F.softplus(t_gen), min=1e-3, max=10.0)
    logits_ig = torch.clamp(i_emb @ g_emb.T / T_img, min=-100, max=100)
    logits_gi = torch.clamp(g_emb @ i_emb.T / T_gen, min=-100, max=100)
    labels = torch.arange(B, device=i_emb.device)
    c_loss = (F.cross_entropy(logits_ig, labels) + F.cross_entropy(logits_gi, labels)) / 2

    # Embedding distillation loss
    e_loss = F.mse_loss(i_emb, g_emb.detach())

    # Alignment loss
    a_loss = F.mse_loss(i_emb, g_emb)

    # MSE prediction loss
    mse_img = F.mse_loss(pred_i, x_raw)
    mse_gene = F.mse_loss(pred_g, x_raw)

    total = (w_contrast * c_loss + w_align * a_loss + w_embed * e_loss +
             w_mse_img * mse_img + w_mse_gene * mse_gene)

    if torch.isnan(total):
        print("NaN in loss computation")
        raise ValueError("NaN in total loss")

    return total, c_loss.item(), a_loss.item(), (w_mse_img * mse_img + w_mse_gene * mse_gene).item()
