# =========================================================
#  model.py   (place in the same folder as before)
# =========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16


# ---------- Gene Transformer Encoder w/ gene-id embeddings ----------
class GeneTransformerEncoder(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768, heads: int = 4):
        super().__init__()
        # Project [raw, logFC] to embed space
        self.value_proj = nn.Linear(2, embed_dim, bias=False)

        # Learnable gene ID embedding
        self.gene_id_emb = nn.Embedding(gene_dim, embed_dim)

        # Learnable positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, gene_dim, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        """
        x : [B, 2*G] (concatenated raw counts & logFC)
        returns: [B, embed_dim] (mean pooled transformer output)
        """
        B, D = x.shape
        G = D // 2
        x = x.view(B, G, 2)  # [B, G, 2]

        # Get gene ID embedding: shape [B, G, embed_dim]
        gene_ids = torch.arange(G, device=x.device).unsqueeze(0).expand(B, -1)
        gene_id_embed = self.gene_id_emb(gene_ids)

        # Add everything together
        tok = (
            self.value_proj(x) +
            gene_id_embed +
            self.pos_emb[:, :G, :]
        )  # [B, G, embed_dim]

        tok = self.transformer(tok)  # [B, G, embed_dim]
        return tok.mean(dim=1)       # mean pool across genes



# ---------- CLIP-like dual-tower model w/ ZINB decoder ----------
class Image2Transcripts(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768, vit_weights="IMAGENET1K_V1"):
        super().__init__()
        # ViT backbone
        self.image_encoder = vit_b_16(weights=vit_weights)
        self.image_encoder.heads = nn.Identity()   # cls token -> 768 dim
        self.image_proj   = nn.Linear(768, embed_dim, bias=False)

        self.gene_encoder = GeneTransformerEncoder(gene_dim, embed_dim)
        self.gene_proj   = nn.Linear(embed_dim, embed_dim, bias=False)

        # ZINB head: outputs (μ, θ, π) for every gene
        self.zinb_head = nn.Linear(embed_dim, gene_dim * 3)

        # contrastive temperature (learned, separate modal)
        self.t_img = nn.Parameter(torch.tensor(0.07))
        self.t_gen = nn.Parameter(torch.tensor(0.07))

        self.gene_dim = gene_dim

    # ----- helpers for ZINB -----
    @staticmethod
    def _softplus(x):
        return torch.nn.functional.softplus(x) + 1e-4          # avoid 0

    def _split_zinb(self, z):
        """
        z : [B, 3G]  →  μ(positive), θ(positive), π(in 0-1)
        """
        B = z.size(0)
        z = z.view(B, 3, self.gene_dim)
        mu     = self._softplus(z[:, 0])
        theta  = self._softplus(z[:, 1])
        pi_logit = z[:, 2]
        pi     = torch.sigmoid(pi_logit)
        return mu, theta, pi

    # ----- forward -----
    def forward(self, image, gene_input):
        # towers
        i_emb = F.normalize(self.image_proj(self.image_encoder(image)), dim=-1)
        g_emb_raw = self.gene_encoder(gene_input)
        g_emb = F.normalize(self.gene_proj(g_emb_raw), dim=-1)

        # ZINB params from **gene** tower (not image tower)
        mu, theta, pi = self._split_zinb(self.zinb_head(g_emb_raw))
        return i_emb, g_emb, (mu, theta, pi)