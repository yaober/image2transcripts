import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ---------- ConvGate: learnable attention mask ----------
class ConvGate(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, patch_tokens):
        # Expecting patch_tokens: [B, N, C]
        assert patch_tokens.dim() == 3, f"Expected [B, N, C], got {patch_tokens.shape}"
        gate = torch.sigmoid(self.net(patch_tokens))  # [B, N, 1]
        return gate

# ---------- Cross-Attention module ----------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, query_token, context_tokens):
        q = self.query(query_token).unsqueeze(1)  # [B, 1, C]
        k = self.key(context_tokens)              # [B, N, C]
        v = self.value(context_tokens)            # [B, N, C]
        attn = torch.softmax(q @ k.transpose(1, 2) * self.scale, dim=-1)  # [B, 1, N]
        out = attn @ v  # [B, 1, C]
        return out.squeeze(1)  # [B, C]

# ---------- Gene Transformer Encoder ----------
class GeneTransformerEncoder(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768, heads: int = 4):
        super().__init__()
        self.value_proj = nn.Linear(2, embed_dim, bias=False)
        self.gene_id_emb = nn.Embedding(gene_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, gene_dim, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

    def forward(self, x):
        B, D = x.shape
        G = D // 2
        x = x.view(B, G, 2)
        gene_ids = torch.arange(G, device=x.device).unsqueeze(0).expand(B, -1)
        tok = self.value_proj(x) + self.gene_id_emb(gene_ids) + self.pos_emb[:, :G, :]
        tok = self.transformer(tok)
        return tok.mean(dim=1), tok

# ---------- Dual-tower with cross-attn + conv gate ViT ----------
class Image2Transcripts(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768, vit_model="vit_base_patch16_224"):
        super().__init__()
        self.image_encoder = timm.create_model(vit_model, pretrained=True)
        self.image_encoder.head = nn.Identity()
        self.image_proj = nn.Linear(self.image_encoder.embed_dim, embed_dim, bias=False)

        self.gene_encoder = GeneTransformerEncoder(gene_dim, embed_dim)
        self.gene_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.zinb_head = nn.Linear(embed_dim, gene_dim * 3, bias=False)

        self.gate = ConvGate(self.image_encoder.embed_dim)
        self.cross_attn = CrossAttentionFusion(embed_dim)

        self.t_img = nn.Parameter(torch.tensor(0.07))
        self.t_gen = nn.Parameter(torch.tensor(0.07))
        self.gene_dim = gene_dim

    @staticmethod
    def _softplus(x):
        return F.softplus(x) + 1e-4

    def _split_zinb(self, z):
        B = z.size(0)
        z = z.view(B, 3, self.gene_dim)
        mu = self._softplus(z[:, 0])
        theta = self._softplus(z[:, 1])
        pi = torch.sigmoid(z[:, 2])
        return mu, theta, pi

    def forward(self, image, gene_input):
        B = image.size(0)
        vit = self.image_encoder

        x = vit.patch_embed(image)  # [B, N, C]
        gate = self.gate(x)         # [B, N, 1]
        x = x * gate

        cls_token = vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit.pos_embed[:, :x.size(1), :]
        x = vit.pos_drop(x)
        for blk in vit.blocks:
            x = blk(x)
        x = vit.norm(x)

        # ----- image tower -----
        i_cls = x[:, 0]  # CLS token
        i_feat = self.cross_attn(i_cls, x[:, 1:])  # fused with patches
        i_emb = self.image_proj(i_feat)
        i_emb = F.normalize(i_emb, dim=-1)
        i_emb = torch.nan_to_num(i_emb)

        # ----- gene tower -----
        g_raw_pooled, _ = self.gene_encoder(gene_input)
        g_emb = self.gene_proj(g_raw_pooled)
        g_emb = F.normalize(g_emb, dim=-1)
        g_emb = torch.nan_to_num(g_emb)

        # ----- ZINB decoding -----
        mu_i, th_i, pi_i = self._split_zinb(self.zinb_head(i_feat))
        mu_g, th_g, pi_g = self._split_zinb(self.zinb_head(g_raw_pooled))

        return i_emb, g_emb, (mu_i, th_i, pi_i), (mu_g, th_g, pi_g)
