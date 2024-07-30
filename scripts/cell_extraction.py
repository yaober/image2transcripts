import zarr
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import csv
import argparse
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
import matplotlib.pyplot as plt

# Increase PIL's pixel limit
Image.MAX_IMAGE_PIXELS = None

def get_neighbors(props, cell_id, cell_masks, max_neighbors=10):
    # Get the bounding box of the current cell
    min_row, min_col, max_row, max_col = props[cell_id].bbox
    # Pad the bounding box by 1 pixel
    min_row, min_col = max(0, min_row - 1), max(0, min_col - 1)
    max_row, max_col = min(cell_masks.shape[0], max_row + 1), min(cell_masks.shape[1], max_col + 1)
    # Get the unique labels in the padded region, excluding 0 and the current cell_id
    neighbor_ids = np.unique(cell_masks[min_row:max_row, min_col:max_col])
    neighbor_ids = [nid for nid in neighbor_ids if nid != 0 and nid != cell_id]
    return neighbor_ids[:max_neighbors]

def process_cell(cell_id, props, cell_masks, wsi_width, wsi_height, whole_slide_image, output_dir, cell_count, plot_mask):
    # Get current cell properties and neighbor cells
    cell_prop = props[cell_id]
    neighbor_ids = get_neighbors(props, cell_id, cell_masks, max_neighbors=cell_count)
    
    # Get bounding box for current cell and its neighbors
    min_row, min_col, max_row, max_col = cell_prop.bbox
    for nid in neighbor_ids:
        n_min_row, n_min_col, n_max_row, n_max_col = props[nid].bbox
        min_row, min_col = min(min_row, n_min_row), min(min_col, n_min_col)
        max_row, max_col = max(max_row, n_max_row), max(max_col, n_max_col)
    
    # Ensure boundaries are within the image
    top, bottom = max(0, min_row), min(wsi_height-1, max_row)
    left, right = max(0, min_col), min(wsi_width-1, max_col)
    #debug
    if left >= right or top >= bottom:
        print(f"Warning: Invalid crop coordinates for cell {cell_id}.")
        print(f"Coordinates: top={top}, bottom={bottom}, left={left}, right={right}")
        print(f"Original bbox: {props[cell_id].bbox}")
        print(f"Image dimensions: {wsi_width}x{wsi_height}")
        return cell_id, neighbor_ids, None
    # Check if the coordinates are valid for cropping
    if left >= right or top >= bottom:
        print(f"Warning: Invalid crop coordinates for cell {cell_id}. Skipping this cell.")
        return cell_id, neighbor_ids, None
    
    # Crop the corresponding area from the whole slide image
    try:
        cell_image = np.array(whole_slide_image.crop((left, top, right+1, bottom+1)))
    except ValueError as e:
        print(f"Error cropping image for cell {cell_id}: {str(e)}")
        return cell_id, neighbor_ids, None
    
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
    print('Starting cell extraction process...')
    print(f"Loading zarr file from {zarr_path}...")
    cells = zarr.open(zarr_path, mode='r')
    print(f"Loaded zarr file from {zarr_path}.")
    
    print(f"Loading whole slide image from {wsi_path}...")
    whole_slide_image = Image.open(wsi_path)
    print(f"Loaded whole slide image from {wsi_path}.")

    cell_masks = cells['masks'][1][:]  # index 1 corresponds to cell segmentation mask
    if cell_masks.ndim != 2:
        raise ValueError(f"Expected cell_masks to be 2D, but got {cell_masks.ndim}D")
    
    os.makedirs(output_dir, exist_ok=True)
    wsi_width, wsi_height = whole_slide_image.size
    
    print("Computing region properties...")
    props = regionprops(cell_masks)
    props = {p.label: p for p in props}  # Convert to dictionary for faster lookup
    
    csv_file = open(os.path.join(output_dir, 'cell_neighbors.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['cell_id', 'neighbor_ids'])
    #debug
    error_count = 0
    for cell_id in tqdm(props.keys(), desc="Processing cells"):
        result = process_cell(cell_id, props, cell_masks, wsi_width, wsi_height, whole_slide_image, output_dir, cell_count, plot_mask)
        cell_id, neighbor_ids, success = result
        if success:
            csv_writer.writerow([cell_id, ','.join(map(str, neighbor_ids))])
            csv_file.flush()  # Ensure data is written to file
        else:
            error_count += 1

    csv_file.close()
    print(f"All cell images have been saved and neighbor information has been written to {os.path.join(output_dir, 'cell_neighbors.csv')}.")
    print(f"Number of cells with errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WSI and zarr files to extract cell images and neighbors.")
    parser.add_argument('--wsi', type=str, help='Path to the whole slide image file.')
    parser.add_argument('--mask', type=str, help='Path to the mask file.')
    parser.add_argument('--output', type=str, help='Output directory to save processed images and CSV.')
    parser.add_argument('--cell_count', type=int, default=10, help='Number of neighboring cells to include. Default is 10.')
    parser.add_argument('--plot_mask', action='store_true', help='Whether to plot and save masks on the cell images.')
    
    args = parser.parse_args()
    
    process_cells(args.wsi, args.mask, args.output, cell_count=args.cell_count, plot_mask=args.plot_mask)