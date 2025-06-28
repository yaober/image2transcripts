# -------------------- model.py --------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import scanpy as sc
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# ---------- Gene Transformer Encoder ----------
class GeneTransformerEncoder(nn.Module):
    def __init__(self, gene_dim, embed_dim=768, heads=4):
        super().__init__()
        self.proj = nn.Linear(2, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, gene_dim, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads, 1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        B, D = x.shape
        G = D // 2
        x = x.view(B, G, 2)
        x = self.proj(x) + self.pos_embedding[:, :G, :]
        x = self.transformer(x)
        return x.mean(dim=1)

# ---------- Full CLIP-Like Model ----------
class Image2Transcripts(nn.Module):
    def __init__(self, gene_dim, embed_dim=768):
        super().__init__()
        self.image_encoder = vit_b_16(pretrained=True)
        self.image_encoder.heads = nn.Identity()
        self.gene_encoder = GeneTransformerEncoder(gene_dim=gene_dim, embed_dim=embed_dim)
        self.reconstruction_head = nn.Linear(embed_dim, gene_dim * 2)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, image, gene_input):
        image_embed = self.image_encoder(image)
        gene_embed = self.gene_encoder(gene_input)
        recon = self.reconstruction_head(gene_embed)
        return image_embed, gene_embed, recon

# -------------------- train_clip_model.py --------------------
def full_loss(image_embed, gene_embed, recon, gene_input, temperature):
    image_embed = F.normalize(image_embed, dim=-1)
    gene_embed = F.normalize(gene_embed, dim=-1)
    logits_i = image_embed @ gene_embed.T / temperature
    logits_g = gene_embed @ image_embed.T / temperature
    labels = torch.arange(image_embed.size(0), device=image_embed.device)
    contrastive_loss = (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_g, labels)) / 2
    align_loss = F.mse_loss(image_embed, gene_embed)
    recon_loss = F.mse_loss(recon, gene_input)
    total = contrastive_loss + 0.1 * align_loss + 0.5 * recon_loss
    return total, contrastive_loss.item(), align_loss.item(), recon_loss.item()

def retrieval_accuracy(image_embed, gene_embed, top_k=1):
    image_embed = F.normalize(image_embed, dim=-1)
    gene_embed = F.normalize(gene_embed, dim=-1)
    sim = image_embed @ gene_embed.T
    topk = sim.topk(k=top_k, dim=1).indices
    labels = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    correct = (topk == labels).any(dim=1).float()
    return correct.mean().item()

def train_clip_model(model, train_loader, val_loader, device, gene_dim, output_dir="output_de", lr=1e-4, epochs=30):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    log_path = os.path.join(output_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss, total_train_acc = 0, 0
        for images, gene_inputs in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            images, gene_inputs = images.to(device), gene_inputs.to(device)
            image_embed, gene_embed, recon = model(images, gene_inputs)
            loss, _, _, _ = full_loss(image_embed, gene_embed, recon, gene_inputs, model.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_acc += retrieval_accuracy(image_embed, gene_embed)
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)

        model.eval()
        total_val_loss, total_val_acc = 0, 0
        with torch.no_grad():
            for images, gene_inputs in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images, gene_inputs = images.to(device), gene_inputs.to(device)
                image_embed, gene_embed, recon = model(images, gene_inputs)
                loss, _, _, _ = full_loss(image_embed, gene_embed, recon, gene_inputs, model.temperature)
                total_val_loss += loss.item()
                total_val_acc += retrieval_accuracy(image_embed, gene_embed)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_train_acc:.4f},{avg_val_acc:.4f}\n")
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print("✅ Best model saved!")

# -------------------- main.py --------------------
if __name__ == "__main__":
    class XeniumCellDataset(Dataset):
        def __init__(self, gene_expr_dir, image_dir, transform=None):
            self.gene_expr_dir = gene_expr_dir
            self.image_dir = image_dir
            self.transform = transform or transforms.ToTensor()
            self.data_pairs = []
            self.shared_genes = None
            self.cell_coords = []
            self.cell_exprs = []
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
            print(f"✅ Shared genes: {len(self.shared_genes)}")
            for fname, adata in temp_storage:
                img_folder = os.path.join(self.image_dir, fname.replace(".h5ad", ""))
                if not os.path.isdir(img_folder):
                    continue
                adata = adata[:, self.shared_genes]
                expr_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                coords = adata.obs[["x_centroid", "y_centroid"]].values
                self.cell_exprs.append(expr_matrix)
                self.cell_coords.append(coords)
                for i, cell_id in enumerate(adata.obs.index):
                    img_path = os.path.join(img_folder, f"{cell_id}.png")
                    if os.path.exists(img_path):
                        self.data_pairs.append((img_path, expr_matrix[i], coords[i]))
            self.cell_exprs = np.concatenate(self.cell_exprs, axis=0)
            self.cell_coords = np.concatenate(self.cell_coords, axis=0)
            self.nn_model = NearestNeighbors(n_neighbors=6).fit(self.cell_coords)

        def __len__(self):
            return len(self.data_pairs)

        def __getitem__(self, idx):
            img_path, expr, coord = self.data_pairs[idx]
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            expr = np.asarray(expr)
            dists, indices = self.nn_model.kneighbors([coord])
            neighbor_exprs = self.cell_exprs[indices[0][1:]]
            neighbor_mean = np.mean(neighbor_exprs, axis=0)
            logfc = np.log2((expr + 1e-3) / (neighbor_mean + 1e-3))
            final_input = np.concatenate([expr, logfc])
            expr_tensor = torch.tensor(final_input, dtype=torch.float32)
            return image, expr_tensor

    # ----- Config & Train -----
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = XeniumCellDataset("data/gene_expression", "data/images", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Image2Transcripts(gene_dim=len(dataset.shared_genes)).to(device)

    train_clip_model(model, train_loader, val_loader, device, gene_dim=len(dataset.shared_genes))