"""HEST-Bench custom encoders for the AAAI benchmark comparison.

Each encoder exposes the TRIDENT-style interface required by
``hest.bench.benchmark``:

- ``forward(x)`` returns a (B, D) embedding,
- ``eval_transforms`` is a callable applied to a PIL image / numpy array,
- ``precision`` is the autocast dtype used during inference.

The two encoders here are:

- ``Image2TranscriptEncoder`` — the ViT-B/16 image branch from a trained
  Image2Transcripts run (default checkpoint:
  ``runs/fixedsplit_v3/full_seed42/best_model.pt``). Returns the pre-projection
  768-d ``i_raw`` feature.
- ``GHISTEncoder`` — the ResNet-style CNN backbone from GHIST. We pull the
  encoder out of ``benchmark/GHIST/model/backbone.py`` and discard the
  cell-level decoder so the wrapper produces a single (B, D) image embedding.
  GHIST's full pipeline depends on nuclei segmentation + cell-type priors
  that the HEST benchmark cannot supply, so this is encoder-only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
SCRIPTS_DIR = REPO_ROOT / "scripts"
GHIST_DIR = REPO_ROOT / "benchmark" / "GHIST"

for p in (REPO_ROOT, MODEL_DIR, SCRIPTS_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_ghist_backbone():
    """Load ``benchmark/GHIST/model/backbone.py::Backbone`` without polluting
    ``sys.modules['model']`` (which is owned by this repo's Image2Transcripts
    code).
    """
    import importlib.util

    layers_path = GHIST_DIR / "model" / "layers.py"
    init_path = GHIST_DIR / "model" / "intialisation.py"
    backbone_path = GHIST_DIR / "model" / "backbone.py"

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # GHIST's backbone.py uses ``from .layers import unetConv2`` and
    # ``from .intialisation import init_weights`` — relative imports that
    # require a parent package. Build a synthetic ``ghist_pkg`` package.
    if "ghist_pkg" not in sys.modules:
        ghist_pkg = type(sys)("ghist_pkg")
        ghist_pkg.__path__ = [str(GHIST_DIR / "model")]
        sys.modules["ghist_pkg"] = ghist_pkg
        _load("ghist_pkg.layers", layers_path)
        _load("ghist_pkg.intialisation", init_path)
        _load("ghist_pkg.backbone", backbone_path)
    return sys.modules["ghist_pkg.backbone"].Backbone


def _imagenet_eval_transforms(size: int = 224):
    # HEST's H5PatchDataset already feeds PIL Images, so do not prepend
    # ``ToPILImage``.
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class Image2TranscriptEncoder(nn.Module):
    """Wrap the trained Image2Transcripts ViT-B/16 image branch as a HEST encoder.

    Returns the pre-projection 768-d feature ``i_raw``. ``i_emb`` (the
    contrastively-aligned projection) would also be valid; the raw feature is
    higher-dimensional and not L2-normalised, which the linear probe prefers.
    """

    def __init__(
        self,
        ckpt_path: str | os.PathLike = REPO_ROOT
        / "runs/fixedsplit_v3/full_seed42/best_model.pt",
        gene_dim: int = 372,
        embed_dim: int = 768,
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()
        from model import Image2Transcripts  # type: ignore  # model/model.py

        self.precision = precision
        self.eval_transforms = _imagenet_eval_transforms(224)

        full = Image2Transcripts(
            gene_dim=gene_dim,
            embed_dim=embed_dim,
            vit_model="vit_base_patch16_224",
            pretrained=False,
            fixed_temperature=False,
        )
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        # Strip a possible DDP "module." prefix.
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = full.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Image2TranscriptEncoder] missing keys: {missing[:6]}{'…' if len(missing) > 6 else ''}")
        if unexpected:
            print(f"[Image2TranscriptEncoder] unexpected keys: {unexpected[:6]}{'…' if len(unexpected) > 6 else ''}")

        # Keep only the components we need for embedding extraction.
        self.image_encoder = full.image_encoder
        self._image_family = full._image_family
        self.embed_dim = full.feat_dim  # 768 for ViT-B/16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate Image2Transcripts._forward_image_features for the manual
        # ViT-B/16 path.
        vit = self.image_encoder
        B = x.size(0)
        z = vit.patch_embed(x)
        cls_token = vit.cls_token.expand(B, -1, -1)
        z = torch.cat((cls_token, z), dim=1)
        pos_embed = vit.pos_embed[:, : z.size(1), :]
        z = vit.pos_drop(z + pos_embed)
        for blk in vit.blocks:
            z = blk(z)
        z = vit.norm(z)
        return z[:, 0]  # (B, 768)


class GHISTEncoder(nn.Module):
    """Pull GHIST's UNet3+ encoder bottleneck (``conv5``) and use it as a HEST encoder.

    GHIST's full pipeline expects per-cell crops + nuclei masks + cell-type
    priors, which the HEST benchmark does not provide. We therefore use only
    the backbone's encoder side. The deepest feature map (``conv5`` output,
    1024 channels) is global-average-pooled to a single (B, 1024) embedding.

    GHIST ships no pretrained backbone weights for the HEST-bench tasks; the
    repo only includes ``data_demo`` configs. Without a checkpoint, this
    wrapper measures the *architectural prior* of GHIST's CNN under random
    initialisation, which we footnote explicitly in the results table.
    """

    def __init__(
        self,
        ckpt_path: str | os.PathLike | None = None,
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()
        # GHIST's top-level package is also named ``model``, which collides
        # with this repo's ``model`` package. Load GHIST's backbone by file
        # path through ``importlib`` to avoid the collision.
        Backbone = _load_ghist_backbone()

        self.precision = precision
        self.eval_transforms = _imagenet_eval_transforms(224)

        backbone = Backbone()
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            print(f"[GHISTEncoder] loaded {ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[GHISTEncoder] no checkpoint supplied — using random init backbone (footnote)")

        self.conv1 = backbone.conv1
        self.maxpool1 = backbone.maxpool1
        self.conv2 = backbone.conv2
        self.maxpool2 = backbone.maxpool2
        self.conv3 = backbone.conv3
        self.maxpool3 = backbone.maxpool3
        self.conv4 = backbone.conv4
        self.maxpool4 = backbone.maxpool4
        self.conv5 = backbone.conv5
        self.embed_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.conv2(self.maxpool1(h1))
        h3 = self.conv3(self.maxpool2(h2))
        h4 = self.conv4(self.maxpool3(h3))
        hd5 = self.conv5(self.maxpool4(h4))
        return torch.nn.functional.adaptive_avg_pool2d(hd5, 1).flatten(1)
