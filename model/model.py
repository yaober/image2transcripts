import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GeneTransformerEncoder(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768, heads: int = 4):
        super().__init__()
        self.value_proj = nn.Linear(2, embed_dim, bias=False)
        self.gene_id_emb = nn.Embedding(gene_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, gene_dim, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads,
                                               dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

    def forward(self, x):
        B, D = x.shape
        G = D // 2
        x = x.view(B, G, 2)
        gene_ids = torch.arange(G, device=x.device).unsqueeze(0).expand(B, -1)
        tok = self.value_proj(x) + self.gene_id_emb(gene_ids) + self.pos_emb[:, :G, :]
        tok = self.transformer(tok)
        return tok.mean(dim=1), tok

class Image2Transcripts(nn.Module):
    def __init__(self, gene_dim: int, embed_dim: int = 768, vit_model="vit_base_patch16_224"):
        super().__init__()
        self.image_encoder = timm.create_model(vit_model, pretrained=True)
        self.image_encoder.head = nn.Identity()
        self.image_proj = nn.Linear(self.image_encoder.embed_dim, embed_dim, bias=False)

        self.gene_encoder = GeneTransformerEncoder(gene_dim, embed_dim)
        self.gene_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.reg_head = nn.Linear(embed_dim, gene_dim, bias=True)  # simple regression

        self.t_img = nn.Parameter(torch.tensor(0.07))
        self.t_gen = nn.Parameter(torch.tensor(0.07))
        self.gene_dim = gene_dim

    def forward(self, image, gene_input):
        B = image.size(0)
        vit = self.image_encoder

        # Patch embedding and Transformer
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
        i_emb = torch.nan_to_num(i_emb)

        g_raw_pooled, _ = self.gene_encoder(gene_input)
        g_emb = F.normalize(self.gene_proj(g_raw_pooled), dim=-1)
        g_emb = torch.nan_to_num(g_emb)

        pred_i = self.reg_head(i_raw)
        pred_g = self.reg_head(g_raw_pooled)

        return i_emb, g_emb, pred_i, pred_g
