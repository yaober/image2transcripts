import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset, random_split
from torchvision.io import read_image
import torchvision.transforms.v2 as T2
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch.amp import GradScaler, autocast
import scanpy as sc
import torch.nn.functional as F

from tqdm import tqdm
from scipy.stats import pearsonr

from model import Image2Transcripts
from train import full_loss

# -------------------------- Dataset --------------------------
class XeniumCellDataset(Dataset):
    def __init__(self, gene_dir: str, img_dir: str, transform=None):
        self.gene_dir, self.img_dir = Path(gene_dir), Path(img_dir)
        self.transform = transform or T2.ToDtype(torch.float32, scale=True)
        self.data_pairs, self.slide_ids = [], []
        self.shared_genes = None
        self.cell_exprs = self.logfc_all = None
        self._prepare()

    def _prepare(self):
        expr_files = sorted([p for p in self.gene_dir.iterdir() if p.suffix == ".h5ad"])
        if not expr_files:
            raise RuntimeError(f"No .h5ad files under {self.gene_dir}")

        tmp, shared = [], None
        for fp in expr_files:
            ad = sc.read_h5ad(fp)
            shared = set(ad.var_names) if shared is None else shared & set(ad.var_names)
            tmp.append((fp.stem, ad))
        self.shared_genes = sorted(shared)
        print(f"✅ Shared genes: {len(self.shared_genes)}")

        expr_list, coord_list = [], []
        for slide_id, ad in tmp:
            img_folder = self.img_dir / slide_id
            if not img_folder.is_dir(): continue
            ad = ad[:, self.shared_genes]
            X = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
            coords = ad.obs[["x_centroid", "y_centroid"]].values
            for i, cell_id in enumerate(ad.obs_names):
                img_fp = img_folder / f"{cell_id}.png"
                if img_fp.exists():
                    self.data_pairs.append((str(img_fp), len(expr_list) + i))
                    self.slide_ids.append(slide_id)
            expr_list.append(X); coord_list.append(coords)

        if not expr_list:
            raise RuntimeError("❌ No matching image–gene pairs found.")
        self.cell_exprs  = np.concatenate(expr_list, axis=0).astype(np.float32)
        self.cell_coords = np.concatenate(coord_list, axis=0).astype(np.float32)

        print("🔗 Pre-computing neighbor logFC ...")
        nn_model = NearestNeighbors(n_neighbors=6).fit(self.cell_coords)
        _, knn_idx = nn_model.kneighbors(self.cell_coords)
        neigh_mean = self.cell_exprs[knn_idx[:, 1:]].mean(axis=1)
        self.logfc_all = np.log2((self.cell_exprs + 1e-3) / (neigh_mean + 1e-3)).astype(np.float32)

    def __len__(self): return len(self.data_pairs)
    def __getitem__(self, idx):
        img_fp, expr_idx = self.data_pairs[idx]
        img = self.transform(read_image(img_fp))
        feat = np.concatenate([self.cell_exprs[expr_idx], self.logfc_all[expr_idx]]).astype(np.float32)
        #print(f"[debug] image shape = {img.shape}") 
        return img, torch.from_numpy(feat)

# -------------------------- DDP utils --------------------------
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ------------------------ Training loop ------------------------
def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def cosine_sim(x, y):
    return F.cosine_similarity(x, y, dim=-1).mean().item()

def pearson_corr(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return np.mean([
        pearsonr(pred[i], target[i])[0]
        for i in range(pred.shape[0])
        if np.std(pred[i]) > 0 and np.std(target[i]) > 0
    ])

def train(model, tr_ld, vl_ld, device, epochs, out_dir, rank, lr=3e-4,
          early_stop_patience=10, save_every=5):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    scaler = GradScaler(init_scale=1024)

    best = float("inf")
    bad_epochs = 0 
    log_fp = out_dir / f"train_log_rank{rank}.csv"
    with open(log_fp, "w") as f:
        f.write("epoch,train,val,train_c,val_c,train_a,val_a,train_z,val_z,cos,pearson\n")

    for ep in range(1, epochs + 1):
        model.train(); met = np.zeros(4); sim, corr = [], []

        pbar = tqdm(tr_ld, desc=f"[GPU{rank}] Epoch {ep:02d}", ncols=100, disable=(rank != 0))
        for img, g in pbar:
            img, g = img.to(device, non_blocking=True), g.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                out = model(img, g)
                m = unwrap(model)
                loss, c, a, z = full_loss(*out, g, m.t_img, m.t_gen)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()

            met += np.array([loss.item(), c, a, z])
            sim.append(cosine_sim(out[0], out[1]))
            corr.append(pearson_corr(out[2][0], g[:, :g.shape[1] // 2]))

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "contrast": f"{c:.3f}",
                "align": f"{a:.3f}",
                "zinb": f"{z:.3f}",
            })

        scheduler.step()

        model.eval(); met_v = np.zeros(4); sim_v, corr_v = [], []
        with torch.no_grad(), autocast(device_type='cuda'):
            for img, g in vl_ld:
                img, g = img.to(device, non_blocking=True), g.to(device, non_blocking=True)
                out = model(img, g)
                m = unwrap(model)
                loss, c, a, z = full_loss(*out, g, m.t_img, m.t_gen)

                met_v += np.array([loss.item(), c, a, z])
                sim_v.append(cosine_sim(out[0], out[1]))
                corr_v.append(pearson_corr(out[2][0], g[:, :g.shape[1] // 2]))

        met /= len(tr_ld); met_v /= len(vl_ld)
        sim_m, sim_mv = np.mean(sim), np.mean(sim_v)
        corr_m, corr_mv = np.mean(corr), np.mean(corr_v)

        with open(log_fp, "a") as f:
            f.write(f"{ep},{met[0]:.4f},{met_v[0]:.4f},"
                    f"{met[1]:.4f},{met_v[1]:.4f},"
                    f"{met[2]:.4f},{met_v[2]:.4f},"
                    f"{met[3]:.4f},{met_v[3]:.4f},"
                    f"{sim_mv:.4f},{corr_mv:.4f}\n")

        if rank == 0:
            print(f"E{ep:03d}  L {met[0]:.3f}/{met_v[0]:.3f}  "
                  f"C {met[1]:.3f}/{met_v[1]:.3f}  A {met[2]:.3f}/{met_v[2]:.3f}  "
                  f"Z {met[3]:.3f}/{met_v[3]:.3f}  COS {sim_mv:.3f}  R {corr_mv:.3f}")

            # 🧠 Save best model
            if met_v[0] < best:
                best = met_v[0]
                bad_epochs = 0
                torch.save(unwrap(model).state_dict(), out_dir / "best_model.pt")
                print("   ✔️  saved best model")
            else:
                bad_epochs += 1
                print(f"   ❌ no improvement for {bad_epochs} epoch(s)")

            # 💾 Save intermediate checkpoint
            if ep % save_every == 0:
                torch.save(unwrap(model).state_dict(), out_dir / f"epoch_{ep:03d}.pt")

            # ⛔️ Early stopping
            if bad_epochs >= early_stop_patience:
                print(f"🛑 Early stopping triggered at epoch {ep}")
                break


# --------------------------- Main DDP ---------------------------
def main_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    tfm = T2.Compose([
        T2.RandomAffine(degrees=5, scale=(0.9, 1.1)),
        T2.Resize((224, 224), antialias=True),
        T2.RandomHorizontalFlip(),
        T2.ColorJitter(hue=.05, saturation=.05),
        T2.ToDtype(torch.float32, scale=True),
    ])

    ds = XeniumCellDataset(args.gene_dir, args.img_dir, transform=tfm)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    tr_ld = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=4, pin_memory=True)

    # use first 20% of dataset for validation (shared across ranks)
    val_len = int(0.2 * len(ds))
    vl_ds = Subset(ds, list(range(val_len)))
    vl_ld = DataLoader(vl_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(f"cuda:{rank}")
    model = Image2Transcripts(gene_dim=len(ds.shared_genes)).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train(model, tr_ld, vl_ld, device, args.epochs, args.out_dir, rank)
    cleanup_ddp()

# --------------------------- CLI Entry ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", default="output_zinb")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    args = ap.parse_args()

    mp.spawn(main_ddp, args=(args.gpus, args), nprocs=args.gpus, join=True)