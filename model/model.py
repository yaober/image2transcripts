from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import load_checkpoint


# ``scripts/foundation_models.py`` provides the gated pathology-foundation
# model loaders (Phikon / Phikon2 / UNI / UNI2-h / Prov-GigaPath) plus an
# ``ensure_hf_token`` helper.  Import is lazy / guarded so users who only
# need the default ``vit_base_patch16_224`` backbone do not have to install
# ``transformers`` or obtain an HF token.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


_FOUNDATION_BACKBONES = {
    "phikon", "phikon2", "uni", "uni2-h", "prov-gigapath",
}


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


# ---------------------------------------------------------------------------
# Backbone construction helpers
# ---------------------------------------------------------------------------


def _build_image_backbone(backbone_name: str, pretrained: bool,
                           ckpt_path: str | None):
    """Load the image encoder and return (module, feature_dim, family_tag).

    Supports the original ``vit_base_patch16_224`` (backward-compat: behaviour
    is byte-identical to the pre-refactor code path) plus the three gated
    pathology foundation models wired in ``scripts/foundation_models.py``.
    """
    if backbone_name in _FOUNDATION_BACKBONES:
        from foundation_models import (  # type: ignore  # lazy
            ensure_hf_token, expected_embed_dim, load_extractor,
        )

        ensure_hf_token()
        model = load_extractor(backbone_name)

        feat_dim = expected_embed_dim(backbone_name)
        if feat_dim is None:
            # Fallback: try common attributes or dry-run.
            feat_dim = getattr(model, "num_features",
                                getattr(model, "embed_dim", None))
            if feat_dim is None:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224)
                    out = model(dummy)
                    if hasattr(out, "last_hidden_state"):
                        feat_dim = int(out.last_hidden_state.shape[-1])
                    else:
                        feat_dim = int(out.shape[-1])

        family = "phikon" if backbone_name.startswith("phikon") else "timm_pooled"
        return model, int(feat_dim), family

    # Default: timm ViT-B/16 path (identical to original behaviour).
    use_pretrained = pretrained and (ckpt_path is None)
    model = timm.create_model(backbone_name, pretrained=use_pretrained)

    if ckpt_path:
        print(f"Loading pretrained weights from {ckpt_path}")
        load_checkpoint(model, ckpt_path, strict=False)

    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
    else:
        model.head = nn.Identity()

    feat_dim = getattr(model, "num_features",
                       getattr(model, "embed_dim"))
    return model, int(feat_dim), "vit_b16_manual"


# ---------------------------------------------------------------------------
# Image2Transcripts
# ---------------------------------------------------------------------------


class Image2Transcripts(nn.Module):
    """Multimodal ViT + gene transformer + ZINB reconstruction head.

    ``backbone_name`` selects the image encoder.  The default
    ``vit_base_patch16_224`` preserves the original architecture and forward
    numerics, so previously saved checkpoints load unchanged.  The five
    foundation-model choices (``phikon``, ``phikon2``, ``uni``, ``uni2-h``,
    ``prov-gigapath``) are enabled when ``scripts/foundation_models.py`` is
    importable and ``HF_TOKEN`` is set in the environment.

    ``freeze_backbone`` sets ``requires_grad=False`` on every parameter of
    the image encoder.  Combined with a foundation-model backbone this gives
    the "foundation features + ZINB/contrastive head" comparison that is
    fair to our own full-training recipe (same objective, same auxiliary
    heads, only the image encoder differs).
    """

    def __init__(self, gene_dim: int, embed_dim: int = 768,
                 vit_model="vit_base_patch16_224",
                 backbone_name: str | None = None,
                 ckpt_path=None, dropout: float = 0.1,
                 pretrained: bool = True,
                 fixed_temperature: bool = False,
                 freeze_backbone: bool = False):
        super().__init__()

        # ``vit_model`` is kept as an alias for backward compatibility;
        # new call sites should prefer ``backbone_name``.
        chosen_backbone = backbone_name or vit_model
        self.backbone_name = chosen_backbone

        self.image_encoder, feat_dim, self._image_family = _build_image_backbone(
            chosen_backbone, pretrained=pretrained, ckpt_path=ckpt_path,
        )
        self.feat_dim = feat_dim

        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            # Keep BN / norm statistics fixed during training too.
            self.image_encoder.eval()
        self.freeze_backbone = freeze_backbone

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

        # Contrastive temperatures are learned by default; --fixed_temperature
        # turns them into buffers so the AAAI fixed-vs-learned ablation can
        # verify they're load-bearing.
        if fixed_temperature:
            self.register_buffer("t_img", torch.tensor(0.07))
            self.register_buffer("t_gen", torch.tensor(0.07))
        else:
            self.t_img = nn.Parameter(torch.tensor(0.07))
            self.t_gen = nn.Parameter(torch.tensor(0.07))
        self.gene_dim = gene_dim

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode regardless of caller."""
        super().train(mode)
        if getattr(self, "freeze_backbone", False) and hasattr(self, "image_encoder"):
            self.image_encoder.eval()
        return self

    def _forward_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Return the pooled (B, feat_dim) image representation."""
        if self._image_family == "phikon":
            out = self.image_encoder(image)
            return out.last_hidden_state[:, 0, :]

        if self._image_family == "timm_pooled":
            # UNI / UNI2-h / Prov-GigaPath were loaded with num_classes=0, so
            # ``forward`` returns the pooled pre-classifier feature directly.
            return self.image_encoder(image)

        # Original bit-for-bit ViT-B/16 path — do not change.
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
        return x[:, 0]

    def forward(self, image, gene_input, gene_mask=None):
        # A frozen backbone is evaluated under no_grad to save activation
        # memory — without this, UNI2-h at batch 64 OOMs on a V100.
        if self.freeze_backbone:
            with torch.no_grad():
                i_raw = self._forward_image_features(image)
            i_raw = i_raw.detach()
        else:
            i_raw = self._forward_image_features(image)

        i_emb = F.normalize(self.image_proj(i_raw), dim=-1)

        g_raw_pooled, _ = self.gene_encoder(gene_input, gene_mask=gene_mask)
        g_emb = F.normalize(self.gene_proj(g_raw_pooled), dim=-1)

        pred_mu, pred_theta, pred_pi = self.zinb_head(i_raw)

        return i_emb, g_emb, pred_mu, pred_theta, pred_pi
