# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import random_split
from tqdm import tqdm
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")


# %%
class GSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, transformer_dim=None, sparsity_penalty=1e-5, num_heads=2, num_layers=1):
        super(GSAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transformer_dim = transformer_dim if transformer_dim is not None else hidden_dim
        self.sparsity_penalty = sparsity_penalty

        # Encoder with Transformer
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.final_encoder = nn.Sequential(
            nn.Linear(hidden_dim, transformer_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) 
        )

    def forward(self, x):
        original_size = x.size()  
        
        if x.shape[1] < self.input_dim:
            padding_size = self.input_dim - x.shape[1]
            x = F.pad(x, (0, padding_size), 'constant', 0)
        
        # Initial linear layer
        x = self.initial_linear(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Final encoding
        encoded = self.final_encoder(x)
        
        # Decoding
        decoded = self.decoder(encoded)
    
        decoded = decoded[:, :original_size[1]]
        
        return encoded, decoded

    def sparsity_loss(self, encoded):
        sparsity_loss = self.sparsity_penalty * torch.mean(torch.abs(encoded))
        return sparsity_loss


# %%
class GeneExpressionDataset(Dataset):
    def __init__(self, adata):
        self.data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.cell_ids = adata.obs.index.tolist()

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        return self.data[idx], self.cell_ids[idx]


# %%

input_dim = 18085
hidden_dim = 64  
transformer_dim = 32

# %%
expression_data = np.random.rand(100, 18085)
input_tensor = torch.tensor(expression_data, dtype=torch.float32)

# %%
sparse_autoencoder = GSAE(input_dim, hidden_dim, transformer_dim)

# %%
encoded, decoded = sparse_autoencoder(input_tensor)

# %%
sparsity_loss = sparse_autoencoder.sparsity_loss(encoded)
sparsity_loss

# %%
reconstruction_loss = F.mse_loss(decoded, input_tensor)
reconstruction_loss

# %%
folder_path = 'data/Xenium/breast_cancer/outs/'

# %%
adata_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5ad')]
adatas = [sc.read_h5ad(path) for path in adata_paths]
print(f"Loaded {len(adatas)} datasets")

# %%
gene_sets = [set(adata.var_names) for adata in adatas]
common_genes = sorted(list(set.intersection(*gene_sets)))

# Filter adatas to keep only common genes
filtered_adatas = [adata[:, common_genes] for adata in adatas]

# Combine all datasets
combined_adata = sc.concat(filtered_adatas, join='outer')

# Create dataset
dataset = GeneExpressionDataset(combined_adata)

# %%
train_ratio = 0.8
val_ratio = 0.2
train_size = int(len(dataset) * train_ratio)
val_size = len(dataset) - train_size


# %%

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# %%

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# %%

# Initialize model
input_dim = len(common_genes)
hidden_dim = 128
transformer_dim = 64

model = GSAE(input_dim=input_dim, hidden_dim=hidden_dim, transformer_dim=transformer_dim)


# %%

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%
# Training loop
num_epochs = 1000
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    total_loss = 0.0
    for batch, _ in tqdm(train_loader, desc="Training Batches", leave=False):  # We don't need cell_ids for training
        batch = batch.to(device)
        optimizer.zero_grad()
        encoded, decoded = model(batch)
        reconstruction_loss = criterion(decoded, batch)
        sparsity_loss = model.module.sparsity_loss(encoded) if isinstance(model, nn.DataParallel) else model.sparsity_loss(encoded)
        loss = reconstruction_loss + sparsity_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch, _ in tqdm(val_loader, desc="Validation Batches", leave=False):  # We don't need cell_ids for validation
            batch = batch.to(device)
            encoded, decoded = model(batch)
            reconstruction_loss = criterion(decoded, batch)
            sparsity_loss = model.module.sparsity_loss(encoded) if isinstance(model, nn.DataParallel) else model.sparsity_loss(encoded)
            loss = reconstruction_loss + sparsity_loss
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')

# %%

# Save model
if isinstance(model, nn.DataParallel):
    model = model.module

model.to('cpu')
torch.save(model.state_dict(), 'GSAE.pth')
print("Model saved")


