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
import os

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
folder_path = 'data/gene_expression'

# %%
adata_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5ad')]
adata_paths

# %%
adatas = []

# %%
adatas = [sc.read_h5ad(path) for path in adata_paths]
print(f"Loaded {len(adatas)} datasets")

# %%
gene_sets = [set(adata.var_names) for adata in adatas]

# %%
common_genes = set.intersection(*gene_sets)

# %%
common_genes = sorted(list(common_genes))

# %%
filtered_data = [adata[:, common_genes].X.toarray() for adata in adatas]

# %%
data_tensors = [torch.tensor(data) for data in filtered_data]


# %%
train_ratio = 0.8  # 80% for training, 20% for validation
val_ratio = 0.2

# Calculate the number of samples for each set
total_size = len(data_tensors)
train_size = int(total_size * train_ratio)
val_size = total_size - train_size

# Split the data into training and validation sets
train_data, val_data = random_split(data_tensors, [train_size, val_size])

# If using DataLoader, create loaders for training and validation
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# %%
input_dim = len(common_genes)
hidden_dim = 128  
transformer_dim = 64  

model = GSAE(input_dim=input_dim, hidden_dim=hidden_dim, transformer_dim=transformer_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in data_tensors:
        optimizer.zero_grad()
        encoded, decoded = model(batch)
        reconstruction_loss = criterion(decoded, batch)
        sparsity_loss = model.sparsity_loss(encoded)
        loss = reconstruction_loss + sparsity_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss}')

# %%
torch.save(model.state_dict(), 'GSAE.pth')
print("Model saved")

