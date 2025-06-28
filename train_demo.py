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

# ------------------------
# Setup
# ------------------------
os.makedirs("output_94_batch", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------
# Dataset
# ------------------------
class XeniumCellDataset(Dataset):
    def __init__(self, gene_expr_dir, image_dir, transform=None):
        self.gene_expr_dir = gene_expr_dir
        self.image_dir = image_dir
        self.transform = transform or transforms.ToTensor()
        self.data_pairs = []
        self.shared_genes = None
        self._prepare_data()

    def _prepare_data(self):
        expr_files = [f for f in os.listdir(self.gene_expr_dir) if f.endswith('.h5ad')]
        shared_genes = None
        temp_storage = []

        for fname in tqdm(expr_files, desc="Finding shared genes"):
            adata = sc.read_h5ad(os.path.join(self.gene_expr_dir, fname))
            genes = set(adata.var_names)
            shared_genes = genes if shared_genes is None else shared_genes & genes
            temp_storage.append((fname, adata))

        self.shared_genes = sorted(list(shared_genes))
        print(f"Shared genes found: {len(self.shared_genes)}")

        for fname, adata in tqdm(temp_storage, desc="Collecting image-gene pairs"):
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

# ------------------------
# Model
# ------------------------
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

# ------------------------
# Loss
# ------------------------
def clip_loss(image_features, gene_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    gene_features = F.normalize(gene_features, dim=-1)
    logits_image = image_features @ gene_features.T / temperature
    logits_gene = gene_features @ image_features.T / temperature
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_image, labels)
    loss_g = F.cross_entropy(logits_gene, labels)
    return (loss_i + loss_g) / 2

# ------------------------
# Data Preparation
# ------------------------
dataset = XeniumCellDataset("data/gene_expression", "data/images_94", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

# ------------------------
# Training Setup
# ------------------------
model = Image2Transcripts(num_genes=len(dataset.shared_genes)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
epochs = 30
temperature = 0.07
best_val_loss = float('inf')

log_path = os.path.join("output_94_batch", "train_log.csv")
with open(log_path, "w") as f:
    f.write("epoch,train_loss,val_loss\n")

# ------------------------
# Training Loop
# ------------------------
for epoch in range(1, epochs + 1):
    model.train()
    total_train_loss = 0
    for batch_images, batch_genes in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        batch_images = batch_images.to(device)
        batch_genes = batch_genes.to(device)
        image_embed, gene_embed = model(batch_images, batch_genes)
        loss = clip_loss(image_embed, gene_embed, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_images, batch_genes in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            batch_images = batch_images.to(device)
            batch_genes = batch_genes.to(device)
            image_embed, gene_embed = model(batch_images, batch_genes)
            loss = clip_loss(image_embed, gene_embed, temperature)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    with open(log_path, "a") as f:
        f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join("output_94_batch", "best_model.pt"))
        print("✅ Best model updated!")
