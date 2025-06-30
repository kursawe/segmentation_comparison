import cellpose.models
import cellpose.io
import os
import imageio
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(os.path.join(script_dir, '..'))
from cell_seg_comparison import *

def create_cellpose_masks():
    # number_frames = 2
    output_dir = os.path.join(script_dir, '..', 'output', 'cellpose_masks')
    os.makedirs(output_dir, exist_ok=True)
 
    t0 = time.time()
    avi_path = os.path.join(script_dir,'..','data','movie1_AB060922a_Job3_All_25_fps.avi')
    tifffolder = os.path.join(script_dir,'..','data','tiffstack')
    tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')])
    # print(tiff_files)
    images = [cellpose.io.imread(os.path.join(tifffolder, f)) for f in tiff_files]
    # images = cellpose.io.imread(avi_path)
    # reader = imageio.get_reader(avi_path)
    # images = [cellpose.io.imread((np.asarray(frame))) for frame in reader]
    images = images[239:1784:4]
    # images = images[240:260:4]
    # images = images[:number_frames]
    print(f"Loaded {len(images)} frames from {avi_path}")
    print(f"Time to load images: {time.time() - t0:.2f} seconds")
    print("Processing images with Cellpose...")

    t2 = time.time()
    # apply cellpose to each image in the tiff sequence
    cellpose_model = cellpose.models.CellposeModel(gpu=True)  # set gpu=False if you don't have a GPU
    print("Cellpose model loaded.")
    cellpose_masks = []
    for index, image in enumerate(images):
        # img should be a numpy array (H, W) or (H, W, C)
        mask, _, _ = cellpose_model.eval(image)
        cellpose_masks.append(mask)
        cellpose.io.imsave(os.path.join(output_dir, f"mask_{index:04d}.png"), mask)
    
    print(f"Segmented {len(cellpose_masks)} frames.")
    print(f"Time to segment images: {time.time() - t2:.2f} seconds")

def quantify_cellpose_performance():
    t0 = time.time()
    # match identified and ground truth masks
    cellpose_mask_dir = os.path.join(script_dir, '..', 'output', 'cellpose_masks')
    cellpose_mask_files = sorted([f for f in os.listdir(cellpose_mask_dir) if f.endswith('.png')])
    cellpose_masks = [cellpose.io.imread(os.path.join(cellpose_mask_dir, f)) for f in cellpose_mask_files]
    cellpose_masks = cellpose_masks[:len(cellpose_masks)]  # match number of frames
    print(f"Loaded {len(cellpose_masks)} ground truth masks from {cellpose_mask_dir}")
    print(f"Time to load cellpose masks: {time.time() - t0:.2f} seconds")
 
    t1 = time.time()
    # match identified and ground truth masks
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    ground_truth_masks = [cellpose.io.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
    ground_truth_masks = ground_truth_masks[:len(cellpose_masks)]  # match number of frames
    print(f"Loaded {len(ground_truth_masks)} ground truth masks from {ground_truth_mask_dir}")
    print(f"Time to load ground truth masks: {time.time() - t1:.2f} seconds")
 
    t2 = time.time()
    # Quantification
    percent_found, global_iot = quantify_segmentation(cellpose_masks, ground_truth_masks)
    print(f"Percentage of ground truth masks found: {(100*percent_found):.2f}%")
    print(f"Global IoT score: {global_iot:.4f}")
    print(f"Time to quantify segmentation: {time.time() - t2:.2f} seconds")

    t3 = time.time()
    # Visualization
    mp4_path = os.path.join(script_dir, '..', 'output', 'segmentation_comparison_cellpose.mp4')
    visualize_segmentation(cellpose_masks, ground_truth_masks,mp4_path)
    print(f"Time to create visualization: {time.time() - t3:.2f} seconds")
    print(f"Total time: {time.time() - t3:.2f} seconds")
    
def make_cellpose_tracking_movie():
    """
    Create a movie showing the tracking of cellpose masks.
    """
    t0 = time.time()
    print("Creating Cellpose tracking movie...")
    # Load cellpose masks
    cellpose_mask_dir = os.path.join(script_dir, '..', 'output', 'cellpose_masks')
    cellpose_mask_files = sorted([f for f in os.listdir(cellpose_mask_dir) if f.endswith('.png')])
    cellpose_masks = [cellpose.io.imread(os.path.join(cellpose_mask_dir, f)) for f in cellpose_mask_files]
    print(f"Loaded {len(cellpose_masks)} cellpose masks from {cellpose_mask_dir}")
    print(f"Time to load cellpose masks: {time.time() - t0:.2f} seconds")
    
    print("Tracking cells over time...")
    t1 = time.time()
    tracked_masks = track_cells_over_time(cellpose_masks)
    print(f"Time to track cells: {time.time() - t1:.2f} seconds")
    
    print("Converting tracked masks to RGB frames...")
    t2 = time.time()
    rgb_frames = tracked_masks_to_rgb(tracked_masks)
    print(f"Time to convert masks to RGB: {time.time() - t2:.2f} seconds")
    
    t3 = time.time()
    print("Creating movie from tracked masks...")
    # Create a movie from the masks
    output_movie_path = os.path.join(script_dir, '..', 'output', 'cellpose_tracking.mp4')
    imageio.mimsave(output_movie_path, rgb_frames, fps=5)
    print(f"Saved tracking movie to {output_movie_path}")
    print(f"Time to create movie: {time.time() - t3:.2f} seconds")

if __name__ == "__main__":
    create_cellpose_masks()
    quantify_cellpose_performance()
    make_cellpose_tracking_movie()
