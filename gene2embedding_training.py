import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import scanpy as sc
import numpy as np
from tqdm import tqdm
import os
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

class GSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, transformer_dim=None, sparsity_penalty=1e-5, num_heads=2, num_layers=1):
        super(GSAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transformer_dim = transformer_dim if transformer_dim is not None else hidden_dim
        self.sparsity_penalty = sparsity_penalty

        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.final_encoder = nn.Sequential(
            nn.Linear(hidden_dim, transformer_dim),
            nn.ReLU()
        )
        
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
        
        x = self.initial_linear(x)
        x = self.transformer(x)
        encoded = self.final_encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded[:, :original_size[1]]
        
        return encoded, decoded

    def sparsity_loss(self, encoded):
        return self.sparsity_penalty * torch.mean(torch.abs(encoded))

class GeneExpressionDataset(Dataset):
    def __init__(self, adata):
        self.data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.cell_ids = adata.obs.index.tolist()

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        return self.data[idx], self.cell_ids[idx]

def parse_arguments():
    parser = argparse.ArgumentParser(description='GSAE model training')
    parser.add_argument('--default_input_dim', type=int, default=18085, help='Default input dimension if common_genes is not available')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--transformer_dim', type=int, default=64, help='Transformer dimension')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/Xenium/breast_cancer/outs/', help='Path to data folder')
    parser.add_argument('--output_model', type=str, default='GSAE.pth', help='Output model file name')
    return parser.parse_args()

def main():
    args = parse_arguments()

    folder_path = args.data_path
    adata_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
    adatas = [sc.read_10x_h5(path) for path in adata_paths]
    print(f"Loaded {len(adatas)} datasets")

    gene_sets = [set(adata.var_names) for adata in adatas]
    common_genes = sorted(list(set.intersection(*gene_sets)))

    input_dim = len(common_genes) if common_genes else args.default_input_dim
    print(f"Input dimension: {input_dim}")

    filtered_adatas = [adata[:, common_genes] for adata in adatas]
    combined_adata = sc.concat(filtered_adatas, join='outer')
    dataset = GeneExpressionDataset(combined_adata)

    train_ratio = 0.8
    val_ratio = 0.2
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = GSAE(input_dim=input_dim, hidden_dim=args.hidden_dim, transformer_dim=args.transformer_dim)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        for batch, _ in tqdm(train_loader, desc="Training Batches", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            reconstruction_loss = criterion(decoded, batch)
            sparsity_loss = model.module.sparsity_loss(encoded) if isinstance(model, nn.DataParallel) else model.sparsity_loss(encoded)
            loss = reconstruction_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, _ in tqdm(val_loader, desc="Validation Batches", leave=False):
                batch = batch.to(device)
                encoded, decoded = model(batch)
                reconstruction_loss = criterion(decoded, batch)
                sparsity_loss = model.module.sparsity_loss(encoded) if isinstance(model, nn.DataParallel) else model.sparsity_loss(encoded)
                loss = reconstruction_loss + sparsity_loss
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.to('cpu')
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved as {args.output_model}")

if __name__ == "__main__":
    main()