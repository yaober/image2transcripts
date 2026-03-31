import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import load_checkpoint


class GeneTransformerEncoder(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768, heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.value_proj = nn.Linear(2, embed_dim, bias=False)
        self.gene_id_emb = nn.Embedding(gene_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, gene_dim, embed_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=2048, batch_first=True,
            dropout=dropout, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, gene_mask=None):
        B, D = x.shape
        G = D // 2
        x = x.view(B, G, 2)
        gene_ids = torch.arange(G, device=x.device).unsqueeze(0).expand(B, -1)
        tok = self.value_proj(x) + self.gene_id_emb(gene_ids) + self.pos_emb[:, :G, :]
        tok = self.input_dropout(tok)

        src_key_padding_mask = None
        if gene_mask is not None:
            src_key_padding_mask = gene_mask

        tok = self.transformer(tok, src_key_padding_mask=src_key_padding_mask)
        pooled = self.pool_norm(tok.mean(dim=1))
        return pooled, tok


class ZINBHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Softplus())
        self.theta_head = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Softplus())
        self.pi_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.theta_head(h), self.pi_head(h)


class Image2Transcripts(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768,
                 vit_model="vit_base_patch16_224",
                 ckpt_path=None, dropout: float = 0.1):
        super().__init__()

        self.image_encoder = timm.create_model(vit_model, pretrained=(ckpt_path is None))

        if ckpt_path:
            print(f"Loading pretrained weights from {ckpt_path}")
            load_checkpoint(self.image_encoder, ckpt_path, strict=False)

        if hasattr(self.image_encoder, "reset_classifier"):
            self.image_encoder.reset_classifier(0)
        else:
            self.image_encoder.head = nn.Identity()

        feat_dim = getattr(self.image_encoder, "num_features",
                           getattr(self.image_encoder, "embed_dim"))

        self.image_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

        self.gene_encoder = GeneTransformerEncoder(
            gene_dim, embed_dim, dropout=dropout,
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

        self.zinb_head = ZINBHead(feat_dim, gene_dim, dropout=dropout)

        self.t_img = nn.Parameter(torch.tensor(0.07))
        self.t_gen = nn.Parameter(torch.tensor(0.07))
        self.gene_dim = gene_dim

    def forward(self, image, gene_input, gene_mask=None):
        B = image.size(0)
        vit = self.image_encoder

        x = vit.patch_embed(image)
        cls_token = vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = vit.pos_embed[:, :x.size(1), :]
        x = vit.pos_drop(x + pos_embed)
        for blk in vit.blocks:
            x = blk(x)
        x = vit.norm(x)

        i_raw = x[:, 0]
        i_emb = F.normalize(self.image_proj(i_raw), dim=-1)

        g_raw_pooled, _ = self.gene_encoder(gene_input, gene_mask=gene_mask)
        g_emb = F.normalize(self.gene_proj(g_raw_pooled), dim=-1)

        pred_mu, pred_theta, pred_pi = self.zinb_head(i_raw)

        return i_emb, g_emb, pred_mu, pred_theta, pred_pi