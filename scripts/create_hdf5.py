import h5py
import numpy as np
import scanpy as sc
from pathlib import Path
import torch
from tqdm import tqdm

def create_hdf5_dataset(cell_embedding_path, image_embedding_path, output_file):
    cell_files = list(Path(cell_embedding_path).glob('*.h5ad'))
    
    with h5py.File(output_file, 'w') as f:
        all_cell_embeddings = []
        all_image_embeddings = []
        
        for cell_file in tqdm(cell_files, desc="Processing cell files"):
            adata = sc.read_h5ad(cell_file)
            cell_embeddings = adata.obsm['emb']
            
            image_folder = Path(image_embedding_path) / cell_file.stem
            valid_cell_embeddings = []
            valid_image_embeddings = []
            
            for idx, cell_emb in zip(adata.obs.index, cell_embeddings):
                image_file = image_folder / f"cell_{idx}.pt"
                if image_file.exists():
                    image_emb = torch.load(image_file).numpy()
                    valid_cell_embeddings.append(cell_emb)
                    valid_image_embeddings.append(image_emb)
            
            if valid_cell_embeddings:
                all_cell_embeddings.extend(valid_cell_embeddings)
                all_image_embeddings.extend(valid_image_embeddings)
            
            print(f"File {cell_file.name}: found {len(valid_cell_embeddings)} valid pairs")
        
        if all_cell_embeddings:
            print("Creating cell embeddings dataset...")
            f.create_dataset('cell_embeddings', data=np.array(all_cell_embeddings))
            
            print("Creating image embeddings dataset...")
            f.create_dataset('image_embeddings', data=np.array(all_image_embeddings))
            
            print(f"Total pairs of embeddings: {len(all_cell_embeddings)}")
        else:
            print("No valid embedding pairs found.")

    print("HDF5 file creation completed.")


create_hdf5_dataset('../data/embeddings/cell_embeddings', '../data/embeddings/image_embeddings', '../data/embeddings/clip_embeddings.h5')
create_hdf5_dataset('../data/embeddings/cell_embeddings_test', '../data/embeddings/image_embeddings', '../data/embeddings/clip_embeddings_test.h5')
