# %%
import os
import torch
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# %%
import h5py
import torch
from torch.utils.data import Dataset

class CLIPDataset(Dataset):
    def __init__(self, hdf5_file):
        self.file = h5py.File(hdf5_file, 'r')
        self.cell_embeddings = self.file['cell_embeddings']
        self.image_embeddings = self.file['image_embeddings']

    def __len__(self):
        return len(self.cell_embeddings)

    def __getitem__(self, idx):
        cell_emb = torch.tensor(self.cell_embeddings[idx], dtype=torch.float32)
        image_emb = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)
        
        cell_emb = cell_emb.squeeze()
        image_emb = image_emb.squeeze()
        
        assert cell_emb.shape == image_emb.shape, f"Shape mismatch: Cell {cell_emb.shape}, Image {image_emb.shape}"
        
        return image_emb, cell_emb 

    def __del__(self):
        self.file.close()

# %%
dataset = CLIPDataset('data/embeddings/clip_embeddings_test.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# %%

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer_norm1(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class CLIPModel(nn.Module):
    def __init__(self, image_dim, gene_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(CLIPModel, self).__init__()
        self.image_mlp = MLP(image_dim, hidden_dim, output_dim, dropout_rate)
        self.gene_mlp = MLP(gene_dim, hidden_dim, output_dim, dropout_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, gene_features):
        image_embeddings = self.image_mlp(image_features)
        gene_embeddings = self.gene_mlp(gene_features)

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        gene_embeddings = F.normalize(gene_embeddings, dim=-1)

        # Scaled pairwise cosine similarities
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ gene_embeddings.t()
        logits_per_gene = logits_per_image.t()

        return logits_per_image, logits_per_gene

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_per_image, logits_per_gene):
        batch_size = logits_per_image.size(0)

        targets = torch.arange(batch_size).long().to(logits_per_image.device)

        loss_img = F.cross_entropy(logits_per_image / self.temperature, targets)
        loss_gene = F.cross_entropy(logits_per_gene / self.temperature, targets)

        return (loss_img + loss_gene) / 2


# %%
def train_clip(model, dataloader, optimizer, device, temperature=0.07):
    model.to(device)
    model.train()
    criterion = InfoNCELoss(temperature)
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, total=num_batches, desc="Training")

    for image_batch, gene_batch in progress_bar:
        image_batch = image_batch.to(device)
        gene_batch = gene_batch.to(device)

        optimizer.zero_grad()
        logits_per_image, logits_per_gene = model(image_batch, gene_batch)
        loss = criterion(logits_per_image, logits_per_gene)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    average_loss = total_loss / num_batches
    progress_bar.set_postfix({"Avg Loss": f"{average_loss:.4f}"})
    progress_bar.close()

    return average_loss

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%

image_dim = 512 
gene_dim = 512 
hidden_dim = 32
output_dim = 128

model = CLIPModel(image_dim, gene_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



# %%
from torchsummary import summary
summary(model, [(image_dim,), (gene_dim,)])

# %%

num_epochs = 100
for epoch in range(num_epochs):
    loss = train_clip(model, dataloader, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# %%



