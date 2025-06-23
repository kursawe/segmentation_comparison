import cellpose.models
import cellpose.io
import torch
import os
import imageio
import numpy as np
from skimage.io import imread
import time
import distinctipy

from skimage.measure import label
from scipy.optimize import linear_sum_assignment

script_dir = os.path.dirname(os.path.abspath(__file__))

#njit does not accelerate this function
def compute_iou_matrix(predicted_mask: np.ndarray,
                       ground_truth_mask: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Intersection over Union (IoU) matrix between predicted and ground truth masks.

    Args:
        predicted_mask (np.ndarray): Labeled mask from prediction.
        ground_truth_mask (np.ndarray): Labeled mask from ground truth.

    Returns:
        iou_matrix (np.ndarray): IoU values for each pair of predicted and ground truth objects.
        predicted_labels (np.ndarray): Array of predicted object labels.
        ground_truth_labels (np.ndarray): Array of ground truth object labels.
    """
    predicted_labels = np.unique(predicted_mask)[1:]  # skip background
    ground_truth_labels = np.unique(ground_truth_mask)[1:]
    iou_matrix = np.zeros((len(predicted_labels), len(ground_truth_labels)))
    for i, pred_label in enumerate(predicted_labels):
        pred_obj = predicted_mask == pred_label
        for j, gt_label in enumerate(ground_truth_labels):
            gt_obj = ground_truth_mask == gt_label
            intersection = np.logical_and(pred_obj, gt_obj).sum()
            union = np.logical_or(pred_obj, gt_obj).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    return iou_matrix, predicted_labels, ground_truth_labels

def match_masks(predicted_mask, ground_truth_mask, iou_threshold=0.5):
    """
    Match predicted and ground truth objects using the IoU matrix and Hungarian algorithm.

    Args:
        predicted_mask (np.ndarray): Labeled mask from prediction.
        ground_truth_mask (np.ndarray): Labeled mask from ground truth.
        iou_threshold (float): Minimum IoU for a match.

    Returns:
        matches (list): List of tuples (pred_label, gt_label, intersection).
    """
    iou_matrix, predicted_labels, ground_truth_labels = compute_iou_matrix(predicted_mask, ground_truth_mask)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)  # maximize IoU
    matches = []
    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] >= iou_threshold:
            intersection = np.logical_and(predicted_mask == predicted_labels[row], ground_truth_mask == ground_truth_labels[col]).sum()
            matches.append((predicted_labels[row], ground_truth_labels[col], intersection))
    return matches

def quantify_segmentation(predicted_masks, ground_truth_masks, iou_threshold=0.5):
    """
    Quantify segmentation performance across all images.
    Returns:
        proportion_found (float): Proportion of ground truth masks found.
        global_iot (float): Global intersection-over-truth score.
    """
    total_ground_truth_objects = 0
    matched_ground_truth_objects = 0
    total_intersection = 0
    total_ground_truth = 0

    for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
        matches = match_masks(pred_mask, gt_mask, iou_threshold=iou_threshold)
        matched_gt_labels = set(gt_label for _, gt_label, _ in matches)
        total_ground_truth_objects += len(np.unique(gt_mask)[1:])  # skip background
        matched_ground_truth_objects += len(matched_gt_labels)

        matches_np = np.array(matches)
        intersection = np.sum(matches_np[:,2])
        total_intersection += intersection
        total_ground_truth += np.sum(gt_mask > 0)

    proportion_found = matched_ground_truth_objects / total_ground_truth_objects if total_ground_truth_objects > 0 else 0
    global_iot = total_intersection / total_ground_truth if total_ground_truth > 0 else 0
    return proportion_found, global_iot

def visualize_segmentation(predicted_masks, ground_truth_masks, output_path):
    """
    Create a visualization movie showing correct (green) and incorrect (red) mask areas.
    """

    visualisation_frames = []
    for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
        matches = match_masks(pred_mask, gt_mask, iou_threshold=0.5)
        this_frame = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        correct = np.zeros_like(pred_mask, dtype=bool)
        for pred_label, gt_label, _ in matches:
            correct |= (pred_mask == pred_label) & (gt_mask == gt_label)
        incorrect = (gt_mask > 0) & ~correct
        this_frame[correct] = [0, 255, 0]    # green for correct
        this_frame[incorrect] = [255, 0, 0]  # red for incorrect
        visualisation_frames.append(this_frame)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, visualisation_frames, fps=5)
    print(f"Saved visualization movie to {output_path}")

def create_cellpose_masks():
    number_frames = 2
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
    images = images[:number_frames]
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
        print("blubdidub")
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
    
def track_cells_over_time(masks):
    """ Track cells over time using a hungarian algorithm. 
    
    Args:
        masks (list of np.ndarray): List of masks for each frame.

    Returns:
        tracked_masks (list of np.ndarray): List of masks with tracked cell IDs.
    """
    tracked_masks = []
    max_id = 0
    first_ids = np.unique(masks[0])[1:]  # skip background
    new_mask = np.zeros_like(masks[0])
    for first_id in first_ids:
        max_id += 1
        new_mask[masks[0] == first_id] = max_id
    tracked_masks.append(new_mask)

    # Iterate through each frame and track cells
    for frame_index, current_mask in enumerate(masks[1:], start=1):
        previous_mask = tracked_masks[frame_index - 1]
        current_ids = np.unique(current_mask)[1:]  # skip background
        new_mask = np.zeros_like(current_mask)
        
        matches = match_masks(current_mask, previous_mask, iou_threshold=0.5)
        label_matches = np.array([(match[0], match[1]) for match in matches])
        
        for current_label in current_ids:
            # Check if the current label has a match in the previous frame
            print('ladidaa')
            matched = (current_label in label_matches[:,0])
            if matched:
                # If matched, keep the same ID
                print('testitest?')
                previous_label = label_matches[label_matches[:,0] == current_label, 1][0]
                new_mask[current_mask == current_label] = previous_label
            else:
                # If not matched, assign a new ID
                max_id += 1
                new_mask[current_mask == current_label] = max_id
        
        tracked_masks.append(new_mask)
    
    return tracked_masks

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

def tracked_masks_to_rgb(tracked_masks):
    """
    Convert tracked masks to RGB frames using distinctipy for high-contrast colors.
    
    Args:
        tracked_masks (list of np.ndarray): List of masks with tracked cell IDs.

    Returns:
        rgb_frames (list of np.ndarray): List of RGB frames corresponding to the tracked masks.
    """
    tracked_masks_np = np.array(tracked_masks)
    print("number of masks in first frame: ", len(np.unique(tracked_masks_np[0])))
    all_ids = np.unique(tracked_masks_np)
    print('total_number_of_ids: ', len(all_ids))
    number_of_ids = len(all_ids) - 1  # Exclude background (0)

    # Generate distinct colors
    colors = distinctipy.get_colors(number_of_ids, pastel_factor=0.5, rng=42)
    color_map = {cell_id: tuple(int(255 * c) for c in color) for cell_id, color in zip(all_ids, colors)}

    rgb_frames = []
    for mask in tracked_masks:
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cell_id, color in color_map.items():
            rgb[mask == cell_id] = color
        rgb_frames.append(rgb)
    return rgb_frames


if __name__ == "__main__":
    # create_cellpose_masks()
    # quantify_cellpose_performance()
    make_cellpose_tracking_movie()