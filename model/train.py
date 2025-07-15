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
def full_loss(i_emb, g_emb, zinb_img, zinb_gene, gene_input, t_img, t_gen,
              w_contrast=0.2, w_align=0.05, w_zinb_gene=1.0, w_zinb_img=0.1, w_embed=0.1):
    B = i_emb.size(0)
    G = gene_input.size(1) // 2
    x_raw = gene_input[:, :G]

    # ===== Contrastive loss =====
    T_img = torch.clamp(F.softplus(t_img), min=1e-3, max=10.0)
    T_gen = torch.clamp(F.softplus(t_gen), min=1e-3, max=10.0)
    logits_ig = torch.clamp(i_emb @ g_emb.T / T_img, min=-100, max=100)
    logits_gi = torch.clamp(g_emb @ i_emb.T / T_gen, min=-100, max=100)
    labels = torch.arange(B, device=i_emb.device)
    c_loss = (F.cross_entropy(logits_ig, labels) + F.cross_entropy(logits_gi, labels)) / 2

    # ===== Embedding distillation loss =====
    e_loss = F.mse_loss(i_emb, g_emb.detach())

    # ===== Alignment loss =====
    a_loss = F.mse_loss(i_emb, g_emb)

    # ===== ZINB losses =====
    mu_i, th_i, pi_i = zinb_img
    mu_g, th_g, pi_g = zinb_gene

    try:
        z_loss_i = zinb_nll(x_raw, mu_i, th_i, pi_i)
        z_loss_g = zinb_nll(x_raw, mu_g, th_g, pi_g)
    except Exception as e:
        print("ZINB loss failed:", e)
        print("mu_i nan:", torch.isnan(mu_i).sum().item(), " | th_i nan:", torch.isnan(th_i).sum().item())
        print("pi_i nan:", torch.isnan(pi_i).sum().item())
        raise

    z_loss = w_zinb_img * z_loss_i + w_zinb_gene * z_loss_g
    total = w_contrast * c_loss + w_align * a_loss + w_embed * e_loss + z_loss

    # Sanity check
    if torch.isnan(total):
        print("NaN detected in total loss. Debugging info:")
        print("- i_emb nan:", torch.isnan(i_emb).any().item())
        print("- g_emb nan:", torch.isnan(g_emb).any().item())
        print("- x_raw min/max:", x_raw.min().item(), x_raw.max().item())
        raise ValueError("NaN in loss computation")

    return total, c_loss.item(), a_loss.item(), z_loss.item()
