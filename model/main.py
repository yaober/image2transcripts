import os
import math
import json
import random
import argparse
from datetime import timedelta
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision.io import read_image
import torchvision.transforms.v2 as T2
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import scanpy as sc

from model import Image2Transcripts
from train import full_loss

# -------------------------- Dataset --------------------------
class XeniumCellDataset(Dataset):
    def __init__(self, gene_dir: str, img_dir: str, transform=None,
                 gene_mask_ratio: float = 0.0, k_neighbors: int = 5):
        self.gene_dir, self.img_dir = Path(gene_dir), Path(img_dir)
        self.transform = transform or T2.ToDtype(torch.float32, scale=True)
        self.gene_mask_ratio = gene_mask_ratio
        self.k_neighbors = k_neighbors
        self.data_pairs = []
        self.slide_ids_per_cell = []
        self.shared_genes = None
        self.cell_exprs = self.logfc_all = self.cell_coords = None
        self._prepare()

    def _prepare(self):
        expr_files = sorted([p for p in self.gene_dir.iterdir() if p.suffix == ".h5ad"])
        tmp, shared = [], None
        for fp in expr_files:
            ad = sc.read_h5ad(fp)
            shared = set(ad.var_names) if shared is None else (shared & set(ad.var_names))
            tmp.append((fp.stem, ad))

        self.shared_genes = sorted(shared)
        print(f"Shared genes: {len(self.shared_genes)}")

        expr_list, coord_list = [], []
        offset = 0

        for slide_id, ad in tmp:
            img_folder = self.img_dir / slide_id
            if not img_folder.is_dir():
                continue

            ad = ad[:, self.shared_genes]
            X = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
            coords = ad.obs[["x_centroid", "y_centroid"]].values

            valid_indices = []
            for i, cell_id in enumerate(ad.obs_names):
                img_fp = img_folder / f"{cell_id}.png"
                if img_fp.exists():
                    self.data_pairs.append((str(img_fp), offset + len(valid_indices)))
                    self.slide_ids_per_cell.append(slide_id)
                    valid_indices.append(i)

            if valid_indices:
                expr_list.append(X[valid_indices])
                coord_list.append(coords[valid_indices])
                offset += len(valid_indices)

        self.cell_exprs = np.concatenate(expr_list, axis=0).astype(np.float32)
        self.cell_coords = np.concatenate(coord_list, axis=0).astype(np.float32)

        print(f"Total cells: {self.cell_exprs.shape[0]}")

        if self.k_neighbors == 0:
            print("k_neighbors=0: disabling neighbour logFC (zeroed)")
            self.logfc_all = np.zeros_like(self.cell_exprs)
        else:
            print(f"Pre-computing neighbour logFC (k={self.k_neighbors}) ...")
            from sklearn.neighbors import NearestNeighbors
            nn_model = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(self.cell_coords)
            _, knn_idx = nn_model.kneighbors(self.cell_coords)
            neigh_mean = self.cell_exprs[knn_idx[:, 1:]].mean(axis=1)
            self.logfc_all = np.log2((self.cell_exprs + 1.0) / (neigh_mean + 1.0)).astype(np.float32)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_fp, expr_idx = self.data_pairs[idx]
        img = self.transform(read_image(img_fp))

        counts = self.cell_exprs[expr_idx].copy()
        logfc = self.logfc_all[expr_idx].copy()

        gene_mask = None
        if self.gene_mask_ratio > 0:
            G = len(counts)
            n_mask = int(G * self.gene_mask_ratio)
            mask_idx = np.random.choice(G, n_mask, replace=False)
            counts[mask_idx] = 0.0
            logfc[mask_idx] = 0.0
            gene_mask = np.zeros(G, dtype=np.bool_)
            gene_mask[mask_idx] = True

        feat = np.concatenate([counts, logfc]).astype(np.float32)

        if gene_mask is not None:
            return img, torch.from_numpy(feat), torch.from_numpy(gene_mask)
        return img, torch.from_numpy(feat), torch.zeros(len(counts), dtype=torch.bool)


# -------------------------- Utils --------------------------
def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12365"
    # 30-min collective watchdog (default is 10 min). End-of-epoch val on
    # large frozen ViT-L/H foundation backbones can drift one rank far
    # enough behind the others to trip the default timeout on the val_stats
    # all_reduce — see job 10476797 (prov-gigapath).
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

def cosine_sim(x, y):
    return F.cosine_similarity(x, y, dim=-1).mean().item()

def pearson_corr(pred, target):
    vx = pred - torch.mean(pred, dim=1, keepdim=True)
    vy = target - torch.mean(target, dim=1, keepdim=True)
    cost = torch.sum(vx * vy, dim=1)
    norm = torch.sqrt(torch.sum(vx ** 2, dim=1) * torch.sum(vy ** 2, dim=1))
    return torch.mean(cost / (norm + 1e-8)).item()


def set_seed(seed, rank=0):
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def seed_worker(worker_id, base_seed, rank):
    worker_seed = base_seed + rank * 1000 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_random_slide_split(unique_slides, seed):
    slides = list(unique_slides)
    rng = np.random.RandomState(seed)
    rng.shuffle(slides)
    split_idx = int(0.8 * len(slides))
    return {
        "name": f"random_seed_{seed}",
        "source": "generated",
        "train": sorted(slides[:split_idx]),
        "val": sorted(slides[split_idx:]),
        "test": [],
    }


def load_slide_split(split_file, unique_slides):
    split_fp = Path(split_file).expanduser().resolve()
    with open(split_fp) as f:
        split_spec = json.load(f)

    if not isinstance(split_spec, dict):
        raise ValueError(f"Split file must contain a JSON object: {split_fp}")

    available = set(unique_slides)
    split_names = ("train", "val", "test")
    normalized = {"name": split_spec.get("name", split_fp.stem), "source": str(split_fp)}
    seen = {}

    for split_name in split_names:
        slides = split_spec.get(split_name, [])
        if slides is None:
            slides = []
        if not isinstance(slides, list) or not all(isinstance(s, str) for s in slides):
            raise ValueError(f"Split '{split_name}' must be a list of slide IDs in {split_fp}")
        if len(slides) != len(set(slides)):
            raise ValueError(f"Split '{split_name}' has duplicate slide IDs in {split_fp}")
        normalized[split_name] = sorted(slides)
        for slide in slides:
            seen.setdefault(slide, []).append(split_name)

    overlaps = {slide: splits for slide, splits in seen.items() if len(splits) > 1}
    if overlaps:
        raise ValueError(f"Slides assigned to multiple splits in {split_fp}: {overlaps}")

    unknown = sorted(set(seen) - available)
    if unknown:
        raise ValueError(f"Split file contains unknown slides not present in dataset: {unknown}")

    missing = sorted(available - set(seen))
    if missing:
        raise ValueError(f"Split file is missing dataset slides: {missing}")

    if not normalized["train"] or not normalized["val"]:
        raise ValueError(f"Split file must contain at least one train slide and one val slide: {split_fp}")

    return normalized


def write_split_manifest(out_dir, split_spec, train_idx, val_idx, test_idx, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split_name": split_spec.get("name"),
        "split_source": split_spec.get("source"),
        "seed": seed,
        "train_slides": split_spec["train"],
        "val_slides": split_spec["val"],
        "test_slides": split_spec.get("test", []),
        "train_cells": len(train_idx),
        "val_cells": len(val_idx),
        "test_cells": len(test_idx),
    }
    with open(out_dir / "slide_split_used.json", "w") as f:
        json.dump(manifest, f, indent=2)


def build_optimizer(model, lr_backbone, lr_head, weight_decay):
    """Differential learning rates: lower for pretrained ViT, higher for new heads."""
    backbone_params, head_params = [], []
    m = unwrap(model)
    for name, param in m.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_encoder"):
            backbone_params.append(param)
        else:
            head_params.append(param)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------- Train Loop --------------------------
def train(model, tr_ld, vl_ld, device, args, rank, sampler=None):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"Training started. Logs at {out_dir}")

    opt = build_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    total_steps = args.epochs * len(tr_ld)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = cosine_warmup_scheduler(opt, warmup_steps, total_steps)
    scaler = GradScaler()

    best_loss = float("inf")
    patience_counter = 0
    log_fp = out_dir / "train_log.csv"

    if rank == 0:
        with open(log_fp, "w") as f:
            f.write("epoch,train_l,val_l,cos_val,pearson_val,lr\n")

    for ep in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(ep)

        # --- Train ---
        model.train()
        train_loss_sum = 0
        train_batches = 0
        skipped_train_batches = 0
        pbar = tqdm(tr_ld, disable=(rank != 0), desc=f"Ep {ep}")

        for step, (img, g, gmask) in enumerate(pbar, start=1):
            img, g, gmask = img.to(device), g.to(device), gmask.to(device)
            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                i_emb, g_emb, mu, theta, pi = model(img, g, gene_mask=gmask)
                m = unwrap(model)
                loss, _, _, _ = full_loss(
                    i_emb, g_emb, mu, theta, pi, g,
                    m.t_img, m.t_gen,
                    w_contrast=args.w_contrast,
                    w_align=args.w_align,
                    w_zinb=args.w_zinb,
                    loss_mode=args.loss_mode,
                )

            skip_code = 0
            loss_value = loss.detach()
            if not torch.isfinite(loss_value):
                skip_code = 1
            elif args.max_loss > 0 and loss_value.abs().item() > args.max_loss:
                skip_code = 2

            skip_tensor = torch.tensor([skip_code], device=device, dtype=torch.int32)
            dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
            skip_code = int(skip_tensor.item())

            if skip_code:
                skipped_train_batches += 1
                if rank == 0:
                    reason = "non-finite" if skip_code == 1 else f"larger than max_loss={args.max_loss:g}"
                    print(f"Skipping {reason} train loss at epoch {ep}, batch {step}")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            # AMP's built-in Inf-detection in scaler.step reliably catches
            # overflow but NOT silent NaN gradients that slip through
            # ``clip_grad_norm_`` (e.g., when the contrastive softmax at a
            # very low temperature produces an all-zero denominator).  Left
            # alone, those NaN grads are applied to model parameters,
            # corrupting weights and eventually turning every validation
            # forward into NaN — the "all val batches skipped" failure.
            # Check for non-finite gradients collectively before stepping.
            local_bad = torch.zeros(1, device=device, dtype=torch.int32)
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    local_bad[0] = 1
                    break
            dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
            if int(local_bad.item()) > 0:
                skipped_train_batches += 1
                if rank == 0:
                    print(f"Skipping step: non-finite gradient at epoch {ep}, batch {step}")
                opt.zero_grad(set_to_none=True)
                # Refresh the scaler so future batches start from a fresh
                # loss scale; without this the same scale can keep producing
                # the same overflow pattern.
                scaler.update()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            new_scale = scaler.get_scale()

            if new_scale < old_scale:
                skipped_train_batches += 1
                if rank == 0:
                    print(f"Skipping optimizer step after AMP overflow at epoch {ep}, batch {step}")
                opt.zero_grad(set_to_none=True)
                continue

            scheduler.step()
            train_loss_sum += loss.item()
            train_batches += 1

        train_stats = torch.tensor(
            [train_loss_sum, train_batches, skipped_train_batches],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        global_train_batches = int(train_stats[1].item())
        global_skipped_train_batches = int(train_stats[2].item())

        if global_train_batches == 0:
            raise RuntimeError(f"All training batches were skipped at epoch {ep}")

        avg_loss = (train_stats[0] / train_stats[1]).item()

        # --- Val ---
        # Validation runs in fp32 (no autocast) so a healthy model is never
        # tripped by fp16 overflow in the eval forward.  The extra compute is
        # a few percent of total training time; the robustness matters more —
        # otherwise an isolated fp16-overflowing val batch would cascade
        # into the "all val batches skipped" abort even when training is
        # still making progress.
        model.eval()
        val_loss, val_cos, val_pearson = 0, 0, 0
        val_batches = 0
        skipped_val_batches = 0
        with torch.no_grad():
            for img, g, gmask in vl_ld:
                img, g = img.to(device), g.to(device)
                i_emb, g_emb, mu, theta, pi = model(img, g)
                m = unwrap(model)
                loss, _, _, _ = full_loss(
                    i_emb, g_emb, mu, theta, pi, g,
                    m.t_img, m.t_gen,
                    w_contrast=args.w_contrast,
                    w_align=args.w_align,
                    w_zinb=args.w_zinb,
                    loss_mode=args.loss_mode,
                )
                if not torch.isfinite(loss):
                    skipped_val_batches += 1
                    continue
                if args.max_loss > 0 and loss.detach().abs().item() > args.max_loss:
                    skipped_val_batches += 1
                    continue
                val_loss += loss.item()
                val_cos += cosine_sim(i_emb, g_emb)
                val_pearson += pearson_corr(mu, g[:, :g.shape[1] // 2])
                val_batches += 1

        val_stats = torch.tensor(
            [val_loss, val_cos, val_pearson, val_batches, skipped_val_batches],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        global_val_batches = int(val_stats[3].item())
        global_skipped_val_batches = int(val_stats[4].item())

        if global_val_batches == 0:
            raise RuntimeError(f"All validation batches were skipped at epoch {ep}")

        v_l = (val_stats[0] / val_stats[3]).item()
        v_c = (val_stats[1] / val_stats[3]).item()
        v_p = (val_stats[2] / val_stats[3]).item()

        current_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            if global_skipped_train_batches:
                print(f"  Skipped {global_skipped_train_batches} train batches in epoch {ep}")
            if global_skipped_val_batches:
                print(f"  Skipped {global_skipped_val_batches} val batches in epoch {ep}")
            print(f"  L_tr: {avg_loss:.4f} | L_val: {v_l:.4f} | "
                  f"Cos: {v_c:.3f} | PCC: {v_p:.3f} | LR: {current_lr:.2e}")
            with open(log_fp, "a") as f:
                f.write(f"{ep},{avg_loss},{v_l},{v_c},{v_p},{current_lr}\n")

            if v_l < best_loss:
                best_loss = v_l
                patience_counter = 0
                torch.save(unwrap(model).state_dict(), out_dir / "best_model.pt")
            else:
                patience_counter += 1

        # Broadcast early stopping decision from rank 0
        stop_flag = torch.tensor([0], device=device)
        if rank == 0 and patience_counter >= args.patience:
            stop_flag[0] = 1
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            if rank == 0:
                print(f"Early stopping at epoch {ep} (patience={args.patience})")
            break

    if rank == 0:
        torch.save(unwrap(model).state_dict(), out_dir / "last_model.pt")


# -------------------------- Main --------------------------
def main_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    set_seed(args.seed, rank)

    train_tfm = T2.Compose([
        T2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        T2.Resize((224, 224), antialias=True),
        T2.RandomHorizontalFlip(),
        T2.RandomVerticalFlip(),
        T2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
        T2.RandomGrayscale(p=0.05),
        T2.ToDtype(torch.float32, scale=True),
    ])

    ds = XeniumCellDataset(
        args.gene_dir, args.img_dir, transform=train_tfm,
        gene_mask_ratio=args.gene_mask_ratio,
        k_neighbors=args.k_neighbors,
    )

    slide_ids = np.array(ds.slide_ids_per_cell)
    unique_slides = sorted(np.unique(slide_ids).tolist())
    if args.split_file:
        split_spec = load_slide_split(args.split_file, unique_slides)
    else:
        split_spec = build_random_slide_split(unique_slides, args.seed)

    # Data-scaling curve: deterministically truncate the train slide list to
    # the first ``max_train_slides`` entries of a seed-specific permutation.
    # Identical seeds give nested subsets across different N, which is the
    # property a scaling plot needs.
    if args.max_train_slides > 0 and len(split_spec["train"]) > args.max_train_slides:
        all_train = sorted(split_spec["train"])
        perm = np.random.RandomState(args.seed).permutation(len(all_train))
        picked = sorted(all_train[int(i)] for i in perm[:args.max_train_slides])
        if rank == 0:
            print(f"[max_train_slides] truncating train from {len(all_train)} -> "
                  f"{len(picked)} slides: {picked}")
        split_spec["train"] = picked

    train_slides = set(split_spec["train"])
    val_slides = set(split_spec["val"])
    test_slides = set(split_spec.get("test", []))

    train_idx = [i for i, s in enumerate(slide_ids) if s in train_slides]
    val_idx = [i for i, s in enumerate(slide_ids) if s in val_slides]
    test_idx = [i for i, s in enumerate(slide_ids) if s in test_slides]

    if not train_idx or not val_idx:
        raise ValueError("Resolved split produced an empty train or val subset")

    # Permutation (shuffled-labels) negative control: re-map each training
    # cell's image to a different training cell's transcript vector.  We
    # mutate the dataset's ``data_pairs`` in place only for training cells so
    # the validation/test loaders keep their correct pairings (the val loader
    # uses a fresh ``XeniumCellDataset`` below, which is unaffected).
    if args.shuffle_labels:
        rng = np.random.RandomState(args.seed + 12345)
        train_expr_ids = [ds.data_pairs[i][1] for i in train_idx]
        rng.shuffle(train_expr_ids)
        for pos, orig_i in enumerate(train_idx):
            img_fp, _ = ds.data_pairs[orig_i]
            ds.data_pairs[orig_i] = (img_fp, train_expr_ids[pos])
        if rank == 0:
            print(f"[shuffle_labels] permuted transcript indices for "
                  f"{len(train_idx)} training cells (seed {args.seed + 12345})")

    if rank == 0:
        print(f"Seed: {args.seed}")
        print(f"Split: {split_spec['name']} ({split_spec['source']})")
        print(f"Train Slides: {len(train_slides)} ({len(train_idx)} cells)")
        print(f"Val Slides:   {len(val_slides)} ({len(val_idx)} cells)")
        if test_slides:
            print(f"Test Slides:  {len(test_slides)} ({len(test_idx)} cells)")
        write_split_manifest(args.out_dir, split_spec, train_idx, val_idx, test_idx, args.seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # Validation dataset with no gene masking and minimal augmentation
    val_tfm = T2.Compose([
        T2.Resize((224, 224), antialias=True),
        T2.ToDtype(torch.float32, scale=True),
    ])
    ds_val_clean = XeniumCellDataset(
        args.gene_dir, args.img_dir, transform=val_tfm,
        gene_mask_ratio=0.0,
        k_neighbors=args.k_neighbors,
    )
    val_ds_clean = Subset(ds_val_clean, val_idx)

    worker_init = partial(seed_worker, base_seed=args.seed, rank=rank)
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    tr_ld = DataLoader(train_ds, batch_size=args.batch, sampler=train_sampler,
                       num_workers=4, pin_memory=True, drop_last=True,
                       worker_init_fn=worker_init)
    vl_ld = DataLoader(val_ds_clean, batch_size=args.batch, shuffle=False,
                       num_workers=4, pin_memory=True,
                       worker_init_fn=worker_init)

    device = torch.device(f"cuda:{rank}")

    ckpt = args.ckpt_path if args.ckpt_path and os.path.exists(args.ckpt_path) else None
    model = Image2Transcripts(
        gene_dim=len(ds.shared_genes), ckpt_path=ckpt,
        backbone_name=args.image_backbone,
        fixed_temperature=args.fixed_temperature,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # A frozen backbone has requires_grad=False on every encoder param, so
    # DDP's reducer doesn't register them at all — no ``find_unused_parameters``
    # needed.  Adding it here used to trigger the t_gen "marked ready twice"
    # bug (DDP scans scalar params under find_unused_parameters and
    # double-fires the autograd hook).
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train(model, tr_ld, vl_ld, device, args, rank, sampler=train_sampler)
    cleanup_ddp()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", default="output_zinb")
    ap.add_argument("--ckpt_path", default=None, help="Path to local ViT checkpoint")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())

    # Learning rates
    ap.add_argument("--lr_backbone", type=float, default=1e-5,
                    help="LR for pretrained ViT backbone")
    ap.add_argument("--lr_head", type=float, default=5e-4,
                    help="LR for gene encoder, projection heads, ZINB head")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.05,
                    help="Fraction of total steps for linear warmup")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_loss", type=float, default=10000.0,
                    help="Skip train/val batches whose absolute loss exceeds this value; set <=0 to disable")

    # Loss weights
    ap.add_argument("--w_contrast", type=float, default=0.1)
    ap.add_argument("--w_align", type=float, default=0.1)
    ap.add_argument("--w_zinb", type=float, default=1.0,
                    help="Weight of the reconstruction term (ZINB or MSE, per --loss_mode)")
    ap.add_argument("--loss_mode", choices=["zinb", "mse"], default="zinb",
                    help="Reconstruction objective: 'zinb' (default) or 'mse' on log1p counts (baseline)")

    # AAAI ablation switches
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="Permutation test: re-pair each training cell's image with "
                         "another training cell's transcript vector before training")
    ap.add_argument("--max_train_slides", type=int, default=0,
                    help="Data-scaling sweep: deterministically keep only the first N "
                         "of a seed-permuted train-slide list (0 = keep all)")
    ap.add_argument("--fixed_temperature", action="store_true",
                    help="Freeze the contrastive temperatures to 0.07 (buffer, not parameter) "
                         "so the learned-vs-fixed ablation can measure their contribution")

    # Image encoder swap (foundation-model benchmark)
    ap.add_argument("--image_backbone", default="vit_base_patch16_224",
                    help="Image encoder choice.  Default 'vit_base_patch16_224' (ImageNet). "
                         "Foundation models: 'phikon2', 'prov-gigapath', 'uni2-h' "
                         "(plus legacy 'phikon', 'uni').  Gated FMs require HF_TOKEN.")
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze the image encoder (zero gradient).  Combined with a "
                         "foundation-model backbone this gives the fair 'foundation "
                         "features + ZINB/contrastive head' comparison.")

    # Regularization
    ap.add_argument("--k_neighbors", type=int, default=5,
                    help="Number of spatial neighbours for the logFC gene feature. "
                         "0 = disable the neighbour channel entirely (logFC zeroed). "
                         "Sweep k ∈ {0,3,5,10,20} for the reviewer-k sensitivity table.")
    ap.add_argument("--gene_mask_ratio", type=float, default=0.15,
                    help="Fraction of genes randomly masked during training")
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (epochs without val loss improvement)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for slide split, sampling, and training")
    ap.add_argument("--split_file", default=None,
                    help="Optional JSON file with explicit train/val/test slide IDs")

    args = ap.parse_args()
    mp.spawn(main_ddp, args=(args.gpus, args), nprocs=args.gpus, join=True)
