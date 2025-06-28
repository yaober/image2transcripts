# =========================================================
#  main.py  ·  ViT ⇄ Gene CLIP  (fast loader + tqdm)
# =========================================================
"""
Launch:
    python main.py --gene_dir data/expr --img_dir data/imgs --out_dir output_zinb
"""

import os, argparse, torch, numpy as np, scanpy as sc
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.io import read_image
import torchvision.transforms.v2 as T2            # ➜ torchvision ≥ 0.18
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch.cuda.amp import GradScaler, autocast

from model import Image2Transcripts               # 仍然沿用你的 model.py
from train import full_loss                       # 仍然沿用你的 train.py


# -------------------------- Dataset --------------------------
class XeniumCellDataset(Dataset):
    """
    Return (aug_image_tensor, concat[expr, logFC]) with:
      • TorchVision C++ decoder (read_image)  → 快于 PIL
      • logFC 5-NN 预先批量计算            → __getitem__ 无 KNN 查询
    """
    def __init__(self, gene_dir: str, img_dir: str, transform=None):
        self.gene_dir, self.img_dir = Path(gene_dir), Path(img_dir)
        self.transform = transform or T2.ToDtype(torch.float32, scale=True)

        self.data_pairs, self.slide_ids = [], []     # (img_fp, expr_idx)
        self.shared_genes = None
        self.cell_exprs = self.logfc_all = None      # [N, G] float32
        self._prepare()

    def _prepare(self):
        # ---------- 1. gather h5ad ----------
        expr_files = sorted([p for p in self.gene_dir.iterdir() if p.suffix == ".h5ad"])
        if not expr_files:
            raise RuntimeError(f"No .h5ad files under {self.gene_dir}")

        tmp, shared = [], None
        for fp in tqdm(expr_files, desc="🔍 Reading h5ad"):
            ad = sc.read_h5ad(fp)
            shared = set(ad.var_names) if shared is None else shared & set(ad.var_names)
            tmp.append((fp.stem, ad))
        self.shared_genes = sorted(shared)
        print(f"✅ Shared genes: {len(self.shared_genes)}")

        # ---------- 2. match images & build expr matrix ----------
        expr_list, coord_list = [], []
        for slide_id, ad in tqdm(tmp, desc="🖼️  Matching images"):
            img_folder = self.img_dir / slide_id
            if not img_folder.is_dir():
                continue
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

        # ---------- 3. pre-compute 5-NN logFC ----------
        print("🔗 Pre-computing neighbor logFC ...")
        nn_model = NearestNeighbors(n_neighbors=6).fit(self.cell_coords)
        _, knn_idx = nn_model.kneighbors(self.cell_coords)
        neigh_mean = self.cell_exprs[knn_idx[:, 1:]].mean(axis=1)
        self.logfc_all = np.log2((self.cell_exprs + 1e-3) /
                                 (neigh_mean + 1e-3)).astype(np.float32)

    # —— torch Dataset API ——
    def __len__(self): return len(self.data_pairs)
    def __getitem__(self, idx):
        img_fp, expr_idx = self.data_pairs[idx]
        img = self.transform(read_image(img_fp))       # C×H×W float32 0-1
        feat = np.concatenate([self.cell_exprs[expr_idx],
                               self.logfc_all[expr_idx]]).astype(np.float32)
        return img, torch.from_numpy(feat)


# ------------------------ Training loop ------------------------
def train(model, tr_ld, vl_ld, device, epochs, out_dir, lr=3e-4):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=epochs)
    scaler = GradScaler(); best = float("inf")

    log_fp = out_dir / "train_log.csv"
    with open(log_fp, "w") as f:
        f.write("epoch,train,val,train_c,val_c,train_a,val_a,train_z,val_z\n")

    for ep in range(1, epochs + 1):
        # -------- TRAIN --------
        model.train(); met = np.zeros(4)
        for img, g in tqdm(tr_ld, desc=f"🚂 Train E{ep:03d}", leave=False):
            img, g = img.to(device, non_blocking=True), g.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss, c, a, z = full_loss(*model(img, g), g, model.t_img, model.t_gen)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            met += np.array([loss.item(), c, a, z])
        scheduler.step()
        if ep == 1:
            model.eval()
            with torch.no_grad(), autocast():
                for img_debug, g_debug in tr_ld:
                    img_debug, g_debug = img_debug.to(device), g_debug.to(device)
                    i_emb_dbg, g_emb_dbg, _ = model(img_debug[:8], g_debug[:8])
                    print("📊 image emb std:", i_emb_dbg.std().item(), "mean:", i_emb_dbg.mean().item())
                    print("📊 gene  emb std:", g_emb_dbg.std().item(), "mean:", g_emb_dbg.mean().item())
                    break

        # -------- VAL --------
        model.eval(); met_v = np.zeros(4)
        with torch.no_grad(), autocast():
            for img, g in tqdm(vl_ld, desc=f"🧪 Val   E{ep:03d}", leave=False):
                img, g = img.to(device, non_blocking=True), g.to(device, non_blocking=True)
                loss, c, a, z = full_loss(*model(img, g), g, model.t_img, model.t_gen)
                met_v += np.array([loss.item(), c, a, z])

        # -------- LOG --------
        met /= len(tr_ld); met_v /= len(vl_ld)
        with open(log_fp, "a") as f:
            f.write(f"{ep},{met[0]:.4f},{met_v[0]:.4f},{met[1]:.4f},{met_v[1]:.4f},"
                    f"{met[2]:.4f},{met_v[2]:.4f},{met[3]:.4f},{met_v[3]:.4f}\n")

        print(f"E{ep:03d}  L {met[0]:.3f}/{met_v[0]:.3f}  "
              f"C {met[1]:.3f}/{met_v[1]:.3f}  "
              f"A {met[2]:.3f}/{met_v[2]:.3f}  "
              f"Z {met[3]:.3f}/{met_v[3]:.3f}")

        if met_v[0] < best:
            best = met_v[0]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print("   ✔️  saved best")


# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_dir", required=True)
    ap.add_argument("--img_dir",  required=True)
    ap.add_argument("--out_dir",  default="output_zinb")
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--epochs",   type=int, default=30)
    args = ap.parse_args()

    # --- transforms (tensor-native) ---
    tfm = T2.Compose([
        T2.Resize((224, 224), antialias=True),
        T2.RandomHorizontalFlip(),
        T2.ColorJitter(hue=.05, saturation=.05),
        T2.ToDtype(torch.float32, scale=True),
    ])

    # --- dataset & split ---
    ds = XeniumCellDataset(args.gene_dir, args.img_dir, transform=tfm)
    uniq_slides = np.unique(ds.slide_ids)
    if len(uniq_slides) > 1:
        n_splits = min(5, len(uniq_slides))
        gkf = GroupKFold(n_splits=n_splits)
        tr_idx, vl_idx = next(gkf.split(np.arange(len(ds)), groups=ds.slide_ids))
        tr_ds, vl_ds = Subset(ds, tr_idx), Subset(ds, vl_idx)
        print(f"🔀 GroupKFold ({n_splits}-fold)  train={len(tr_ds)}  val={len(vl_ds)}")
    else:
        tr_len = int(0.8 * len(ds)); vl_len = len(ds) - tr_len
        tr_ds, vl_ds = random_split(ds, [tr_len, vl_len],
                                    generator=torch.Generator().manual_seed(42))
        print(f"🔀 Random split  train={tr_len}  val={vl_len}")

    # --- DataLoaders ---
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=8, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4)
    vl_ld = DataLoader(vl_ds, batch_size=args.batch, shuffle=False,
                       num_workers=8, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4)

    # --- model & train ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Image2Transcripts(gene_dim=len(ds.shared_genes)).to(device)
    train(model, tr_ld, vl_ld, device, args.epochs, args.out_dir)
