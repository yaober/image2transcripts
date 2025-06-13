import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16
from PIL import Image
import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# -------- Dataset Definition --------
class XeniumCellDataset(Dataset):
    def __init__(self, gene_expr_dir, image_dir, transform=None):
        self.gene_expr_dir = gene_expr_dir
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.data_pairs = []
        self.shared_genes = None
        self._prepare_data()

    def _prepare_data(self):
        expr_files = [f for f in os.listdir(self.gene_expr_dir) if f.endswith('.h5ad')]
        shared_genes = None
        temp_storage = []

        for fname in expr_files:
            adata = sc.read_h5ad(os.path.join(self.gene_expr_dir, fname))
            genes = set(adata.var_names)
            shared_genes = genes if shared_genes is None else shared_genes & genes
            temp_storage.append((fname, adata))

        self.shared_genes = sorted(list(shared_genes))
        for fname, adata in temp_storage:
            img_folder = os.path.join(self.image_dir, fname.replace(".h5ad", ""))
            if not os.path.isdir(img_folder):
                continue
            adata = adata[:, self.shared_genes]
            for cell_id in adata.obs.index:
                img_path = os.path.join(img_folder, f"{cell_id}.png")
                if os.path.exists(img_path):
                    expr_vector = adata[cell_id].X.toarray().flatten() if hasattr(adata[cell_id].X, "toarray") else adata[cell_id].X.flatten()
                    self.data_pairs.append((img_path, expr_vector))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, expr_vector = self.data_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        expr_tensor = torch.tensor(expr_vector, dtype=torch.float32)
        return image, expr_tensor

# -------- Model Definition --------
class Image2Transcripts(nn.Module):
    def __init__(self, num_genes, embed_dim=768):
        super().__init__()
        self.image_encoder = vit_b_16(pretrained=True)
        self.image_encoder.heads = nn.Identity()
        self.gene_encoder = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, image, gene):
        image_embed = self.image_encoder(image)
        gene_embed = self.gene_encoder(gene)
        return image_embed, gene_embed

def clip_loss(image_features, gene_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    gene_features = F.normalize(gene_features, dim=-1)
    logits_image = image_features @ gene_features.T / temperature
    logits_gene = gene_features @ image_features.T / temperature
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_image, labels)
    loss_g = F.cross_entropy(logits_gene, labels)
    return (loss_i + loss_g) / 2

# -------- Training Loop --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--expr_dir", type=str, default="data/gene_expression")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    full_dataset = XeniumCellDataset(args.expr_dir, args.image_dir)
    num_genes = len(full_dataset.shared_genes)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Image2Transcripts(num_genes=num_genes)
    model = nn.DataParallel(model)  # ✅ Use all available GPUs
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    temperature = 0.07

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        for batch_images, batch_genes in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            batch_images = batch_images.to(device, non_blocking=True)
            batch_genes = batch_genes.to(device, non_blocking=True)
            image_embed, gene_embed = model(batch_images, batch_genes)
            loss = clip_loss(image_embed, gene_embed, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_images, batch_genes in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                batch_images = batch_images.to(device, non_blocking=True)
                batch_genes = batch_genes.to(device, non_blocking=True)
                image_embed, gene_embed = model(batch_images, batch_genes)
                loss = clip_loss(image_embed, gene_embed, temperature)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save model
        torch.save(model.module.state_dict(), f"model_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
