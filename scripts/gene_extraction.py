import scanpy as sc
import anndata as ad
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Load the specific dataset from the .h5 file
adata = sc.read_h5ad('../data/hd_wsi/Xenium/breast_cancer/outs/cell_feature_matrix.h5ad')

# Extract unique cell IDs
cell_ids = adata.obs.index.unique()

# Function to save a single cell's data
def save_cell_data(cell_id):
    try:
        # Subset the data for a single cell_id
        adata_cell = adata[adata.obs.index == cell_id, :]
        # Ensure the subset is not empty
        if adata_cell.shape[0] > 0:
            # Save as a .h5ad file, naming by cell_id
            adata_cell.write(f'../data/gene_expression/cell_{cell_id}.h5ad')
        else:
            print(f"Warning: No data found for cell_id {cell_id}")
    except Exception as e:
        print(f"Error processing cell_id {cell_id}: {e}")

# Use ProcessPoolExecutor to parallelize the saving process
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(save_cell_data, cell_ids), total=len(cell_ids), desc="Saving cell data"))
