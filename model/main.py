import os
import math
import argparse
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
                 gene_mask_ratio: float = 0.0):
        self.gene_dir, self.img_dir = Path(gene_dir), Path(img_dir)
        self.transform = transform or T2.ToDtype(torch.float32, scale=True)
        self.gene_mask_ratio = gene_mask_ratio
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

        print("Pre-computing neighbor logFC ...")
        from sklearn.neighbors import NearestNeighbors
        nn_model = NearestNeighbors(n_neighbors=6).fit(self.cell_coords)
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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
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
        avg_loss = 0
        pbar = tqdm(tr_ld, disable=(rank != 0), desc=f"Ep {ep}")

        for img, g, gmask in pbar:
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
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            avg_loss += loss.item()

        avg_loss /= len(tr_ld)

        # --- Val ---
        model.eval()
        val_loss, val_cos, val_pearson = 0, 0, 0
        with torch.no_grad(), autocast(device_type="cuda"):
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
                )
                val_loss += loss.item()
                val_cos += cosine_sim(i_emb, g_emb)
                val_pearson += pearson_corr(mu, g[:, :g.shape[1] // 2])

        val_loss /= len(vl_ld)
        val_cos /= len(vl_ld)
        val_pearson /= len(vl_ld)

        metrics = torch.tensor([val_loss, val_cos, val_pearson], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        v_l, v_c, v_p = metrics.tolist()

        current_lr = scheduler.get_last_lr()[0]

        if rank == 0:
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
    )

    slide_ids = np.array(ds.slide_ids_per_cell)
    unique_slides = np.unique(slide_ids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_slides)

    split_idx = int(0.8 * len(unique_slides))
    train_slides = set(unique_slides[:split_idx])
    val_slides = set(unique_slides[split_idx:])

    train_idx = [i for i, s in enumerate(slide_ids) if s in train_slides]
    val_idx = [i for i, s in enumerate(slide_ids) if s in val_slides]

    if rank == 0:
        print(f"Train Slides: {len(train_slides)} ({len(train_idx)} cells)")
        print(f"Val Slides:   {len(val_slides)} ({len(val_idx)} cells)")

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
    )
    val_ds_clean = Subset(ds_val_clean, val_idx)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    tr_ld = DataLoader(train_ds, batch_size=args.batch, sampler=train_sampler,
                       num_workers=4, pin_memory=True, drop_last=True)
    vl_ld = DataLoader(val_ds_clean, batch_size=args.batch, shuffle=False,
                       num_workers=4, pin_memory=True)

    device = torch.device(f"cuda:{rank}")

    ckpt = args.ckpt_path if args.ckpt_path and os.path.exists(args.ckpt_path) else None
    model = Image2Transcripts(gene_dim=len(ds.shared_genes), ckpt_path=ckpt).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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

    # Loss weights
    ap.add_argument("--w_contrast", type=float, default=0.1)
    ap.add_argument("--w_align", type=float, default=0.1)
    ap.add_argument("--w_zinb", type=float, default=1.0)

    # Regularization
    ap.add_argument("--gene_mask_ratio", type=float, default=0.15,
                    help="Fraction of genes randomly masked during training")
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (epochs without val loss improvement)")

    args = ap.parse_args()
    mp.spawn(main_ddp, args=(args.gpus, args), nprocs=args.gpus, join=True)