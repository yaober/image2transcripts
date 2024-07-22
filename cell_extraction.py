import zarr
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import csv
import argparse
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Increase PIL's pixel limit
Image.MAX_IMAGE_PIXELS = None

def get_neighbors(mask, cell_id, max_neighbors=10):
    cell_mask = (mask == cell_id)
    dilated = np.zeros_like(cell_mask)
    dilated[1:, :] |= cell_mask[:-1, :]
    dilated[:-1, :] |= cell_mask[1:, :]
    dilated[:, 1:] |= cell_mask[:, :-1]
    dilated[:, :-1] |= cell_mask[:, 1:]
    neighbor_ids = np.unique(mask[dilated & ~cell_mask])
    return [nid for nid in neighbor_ids if nid != 0 and nid != cell_id][:max_neighbors]

def process_cell(args):
    cell_id, cell_masks, wsi_width, wsi_height, whole_slide_image, output_dir, cell_count, plot_mask = args
    
    # Get current cell mask and neighbor cells
    cell_mask = (cell_masks == cell_id)
    neighbor_ids = get_neighbors(cell_masks, cell_id, max_neighbors=cell_count)
    # Create extended mask including neighbors
    extended_mask = cell_mask.copy()
    for nid in neighbor_ids:
        extended_mask |= (cell_masks == nid)
    # Find the bounding box of the extended mask
    rows, cols = np.where(extended_mask)
    if len(rows) == 0 or len(cols) == 0:
        return cell_id, neighbor_ids, None # Skip empty masks
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    # Ensure boundaries are within the image
    top, bottom = max(0, top), min(wsi_height-1, bottom)
    left, right = max(0, left), min(wsi_width-1, right)
    # Crop the corresponding area from the whole slide image
    cell_image = np.array(whole_slide_image.crop((left, top, right+1, bottom+1)))
    
    if plot_mask:
        # Create a color mask for visualization
        color_mask = np.zeros((*cell_image.shape[:2], 3), dtype=np.uint8)
        boundaries = find_boundaries(cell_masks[top:bottom+1, left:right+1], mode='thick')
        color_mask[boundaries] = [255, 255, 255] # White boundaries
        color_mask[cell_masks[top:bottom+1, left:right+1] == cell_id] = [255, 0, 0] # Red for central cell
        for i, nid in enumerate(neighbor_ids):
            color = np.array(plt.cm.rainbow(i / len(neighbor_ids))[:3]) * 255
            color_mask[cell_masks[top:bottom+1, left:right+1] == nid] = color.astype(np.uint8)
        # Blend the color mask with the original image
        blended_image = (0.7 * cell_image + 0.3 * color_mask).astype(np.uint8)
        # Save the image
        cell_id_str = f"{cell_id:010d}"
        Image.fromarray(blended_image).save(os.path.join(output_dir, f'cell_{cell_id}.png'))
    else:
        # Save the image
        cell_id_str = f"{cell_id:010d}"
        Image.fromarray(cell_image).save(os.path.join(output_dir, f'cell_{cell_id}.png'))
    
    return cell_id, neighbor_ids, True

def process_cells(wsi_path, zarr_path, output_dir, cell_count=10, plot_mask=False):
    # 1. Read the zarr file
    print('Starting cell extraction process...')
    print(f"Loading zarr file from {zarr_path}...")
    cells = zarr.open(zarr_path, mode='r')
    print(f"Loaded zarr file from {zarr_path}.")
    # 2. Read the whole slide image
    print(f"Loading whole slide image from {wsi_path}...")
    whole_slide_image = Image.open(wsi_path)
    print(f"Loaded whole slide image from {wsi_path}.")

    # 3. Cut out the HE image corresponding to each cell
    # Get cell mask
    cell_masks = cells['masks'][1][:] # index 1 corresponds to cell segmentation mask
    if cell_masks.ndim != 2:
        raise ValueError(f"Expected cell_masks to be 2D, but got {cell_masks.ndim}D")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Get the dimensions of the whole slide image
    wsi_width, wsi_height = whole_slide_image.size
    # Get unique cell ids (excluding background, which is usually 0)
    unique_cell_ids = np.unique(cell_masks)
    unique_cell_ids = unique_cell_ids[unique_cell_ids != 0]
    
    # Prepare CSV file
    csv_file = open(os.path.join(output_dir, 'cell_neighbors.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['cell_id', 'neighbor_ids'])

    # Multiprocessing pool
    pool = Pool(cpu_count())
    tasks = [(cell_id, cell_masks, wsi_width, wsi_height, whole_slide_image, output_dir, cell_count, plot_mask) for cell_id in unique_cell_ids]
    
    for result in tqdm(pool.imap_unordered(process_cell, tasks), total=len(tasks), desc="Processing cells"):
        cell_id, neighbor_ids, success = result
        if success:
            csv_writer.writerow([cell_id, ','.join(map(str, neighbor_ids))])
            csv_file.flush()  # Ensure data is written to file
    
    pool.close()
    pool.join()

    # Close CSV file
    csv_file.close()
    print(f"All cell images have been saved and neighbor information has been written to {os.path.join(output_dir, 'cell_neighbors.csv')}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WSI and zarr files to extract cell images and neighbors.")
    parser.add_argument('--wsi', type=str, help='Path to the whole slide image file.')
    parser.add_argument('--mask', type=str, help='Path to the mask file.')
    parser.add_argument('--output', type=str, help='Output directory to save processed images and CSV.')
    parser.add_argument('--cell_count', type=int, default=10, help='Number of neighboring cells to include. Default is 10.')
    parser.add_argument('--plot_mask', action='store_true', help='Whether to plot and save masks on the cell images.')
    
    args = parser.parse_args()
    
    process_cells(args.wsi, args.mask, args.output, cell_count=args.cell_count, plot_mask=args.plot_mask)
