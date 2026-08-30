"""Shared utilities for pathology-foundation-model embedding extraction.

Factored out of ``feature_extraction.py`` so the same model-loading and
in-memory-batched-inference code can be reused by the Xenium cell-level
feature extractor (``scripts/extract_foundation_features_xenium.py``) and
the TCGA WSI extractor (``feature_extraction.py``).

Three pathology foundation models are supported out of the box:

    phikon2         owkin/phikon-v2                (AutoModel, CLS token)
    prov-gigapath   prov-gigapath/prov-gigapath    (timm, pooled output)
    uni2-h          MahmoodLab/UNI2-h              (timm, custom SwiGLU ViT)

An ``HF_TOKEN`` is required to pull any of the three; the module respects
a pre-set ``HF_TOKEN`` env var and will not silently overwrite one that is
already set by the caller.
"""

from __future__ import annotations

import contextlib
import os
from typing import Dict, List

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, ViTModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_INPUT_SIZE: int = 224
ALL_MODEL_NAMES: List[str] = ["phikon2", "prov-gigapath", "uni2-h"]

NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406],
                               dtype=torch.float32).view(3, 1, 1)
NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225],
                              dtype=torch.float32).view(3, 1, 1)


# Approximate embedding widths; used only for pre-allocating output buffers.
_KNOWN_EMBED_DIMS: Dict[str, int] = {
    "phikon": 768,
    "phikon2": 1024,
    "prov-gigapath": 1536,
    "uni": 1024,
    "uni2-h": 1536,
}


def expected_embed_dim(model_name: str) -> int | None:
    return _KNOWN_EMBED_DIMS.get(model_name)


# ---------------------------------------------------------------------------
# HuggingFace token handling
# ---------------------------------------------------------------------------


# The three supported foundation models are all gated; the caller must
# already have an authenticated HF session.  We read ``HF_TOKEN`` from the
# env and never overwrite one that is already set — callers provide their
# own token without editing the source.  There is no hardcoded fallback:
# if ``HF_TOKEN`` is absent, ensure_hf_token() raises a clear error.
_DEFAULT_HF_TOKEN_FALLBACK: str | None = None


def ensure_hf_token(fallback: str | None = _DEFAULT_HF_TOKEN_FALLBACK) -> None:
    """Ensure ``HF_TOKEN`` is set in the environment.

    If already set, leave it alone.  Otherwise fall back to ``fallback``.
    Raise if neither is available.
    """
    if os.environ.get("HF_TOKEN"):
        return
    if fallback:
        os.environ["HF_TOKEN"] = fallback
        return
    raise RuntimeError(
        "HF_TOKEN is not set; pathology foundation models require an "
        "authenticated HuggingFace session.  Run "
        "`export HF_TOKEN=<your_hf_token>` before invoking this script."
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_extractor(model_name: str) -> torch.nn.Module:
    """Load a pretrained pathology foundation model in eval mode.

    The four non-default options (``phikon``, ``uni``) are kept for
    backward compatibility with existing ``feature_extraction.py`` configs.
    """
    if model_name == "phikon":
        extractor = ViTModel.from_pretrained(
            "owkin/phikon", add_pooling_layer=False,
        )
    elif model_name == "phikon2":
        extractor = AutoModel.from_pretrained("owkin/phikon-v2")
    elif model_name == "prov-gigapath":
        extractor = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True,
        )
    elif model_name == "uni":
        extractor = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, patch_size=16, num_classes=0,
            img_size=224, dynamic_img_size=True,
        )
    elif model_name == "uni2-h":
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        extractor = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs,
        )
    else:
        raise ValueError(f"Unknown foundation model: {model_name}")
    return extractor


def load_all_models(model_names: List[str], device: torch.device,
                     ) -> Dict[str, torch.nn.Module]:
    """Load and prepare (eval, .to(device)) each named foundation model."""
    ensure_hf_token()
    models: Dict[str, torch.nn.Module] = {}
    for name in model_names:
        m = load_extractor(name)
        m.eval()
        m.to(device)
        for p in m.parameters():
            p.requires_grad = False
        models[name] = m
    return models


# ---------------------------------------------------------------------------
# Dataset / preprocessing
# ---------------------------------------------------------------------------


class PatchDataset(Dataset):
    """In-memory RGB patch dataset with ImageNet normalisation.

    Accepts a list of ``(H, W, 3)`` uint8 numpy arrays and returns
    normalised float32 ``(3, H, W)`` tensors ready for any of the supported
    foundation models.
    """

    def __init__(self, patches: List[np.ndarray]):
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = np.ascontiguousarray(self.patches[idx])
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        tensor.mul_(1.0 / 255.0)
        tensor.sub_(NORMALIZE_MEAN).div_(NORMALIZE_STD)
        return tensor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _extract_features(model_name: str, raw_output) -> torch.Tensor:
    """Normalise each model's forward output into a ``(B, D)`` feature tensor."""
    if "phikon" in model_name:
        return raw_output.last_hidden_state[:, 0, :]
    # UNI / GigaPath / vanilla timm models with reset_classifier return the
    # pooled feature directly.
    return raw_output


def run_inference_on_patches(
    models: Dict[str, torch.nn.Module],
    patches: List[np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str = "    batches",
) -> Dict[str, np.ndarray]:
    """Run shared batched inference for several foundation models in parallel.

    Exactly mirrors the behaviour of the original
    ``feature_extraction.run_inference_on_patches`` so the TCGA pipeline is
    unchanged after the refactor.
    """
    if not patches:
        return {}

    dataset = PatchDataset(patches)
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    features: Dict[str, List[np.ndarray]] = {name: [] for name in models}
    use_autocast = device.type == "cuda"

    iterator = tqdm(loader, desc=progress_desc, leave=False) if show_progress else loader

    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device, non_blocking=True)
            ctx = (torch.autocast(device_type="cuda")
                   if use_autocast else contextlib.nullcontext())
            with ctx:
                for name, model in models.items():
                    out = model(batch)
                    feats = _extract_features(name, out)
                    features[name].append(feats.detach().cpu().float().numpy())
                    del out, feats

    return {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in features.items()
    }
