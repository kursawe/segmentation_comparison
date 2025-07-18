import os
import torch
import imageio
import numpy as np
import distinctipy
import time
import colorsys

from scipy.optimize import linear_sum_assignment

this_dir = os.path.dirname(os.path.abspath(__file__))

# njit does not accelerate this function
# @njit
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

    # Compute centroids for predicted and ground truth labels
    pred_centroids = []
    for label in predicted_labels:
        coords = np.column_stack(np.where(predicted_mask == label))
        pred_centroids.append(coords.mean(axis=0))
    pred_centroids = np.array(pred_centroids)

    gt_centroids = []
    for label in ground_truth_labels:
        coords = np.column_stack(np.where(ground_truth_mask == label))
        gt_centroids.append(coords.mean(axis=0))
    gt_centroids = np.array(gt_centroids)

    # Compute distances between all centroids
    # distances = cdist(pred_centroids, gt_centroids)

    for i, pred_label in enumerate(predicted_labels):
        # Find indices of k closest ground truth labels
        dists = np.empty(len(ground_truth_labels), dtype=np.float64)
        for j in range(len(ground_truth_labels)):
            dists[j] = np.sqrt(
                (pred_centroids[i, 0] - gt_centroids[j, 0]) ** 2 +
                (pred_centroids[i, 1] - gt_centroids[j, 1]) ** 2
            )
        closest_indices = np.argsort(dists)[:3]
        # closest_indices = np.argsort(distances[i])[:3]
        pred_obj = predicted_mask == pred_label
        for j in closest_indices:
            gt_label = ground_truth_labels[j]
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
        intersection = 0
        if len(matches) >0:
            matched_gt_labels = set(gt_label for _, gt_label, _ in matches)
            matched_ground_truth_objects += len(matched_gt_labels)
            matches_np = np.array(matches)
            intersection = np.sum(matches_np[:,2])
        total_ground_truth_objects += len(np.unique(gt_mask)[1:])  # skip background
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
            matched = (current_label in label_matches[:,0])
            if matched:
                # If matched, keep the same ID
                previous_label = label_matches[label_matches[:,0] == current_label, 1][0]
                new_mask[current_mask == current_label] = previous_label
            else:
                # If not matched, assign a new ID
                max_id += 1
                new_mask[current_mask == current_label] = max_id
        
        tracked_masks.append(new_mask)
    
    return tracked_masks

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
    #distinctipy.get_distinct_colors(number_of_ids, random_seed=42)
    colors = get_distinct_colors(number_of_ids)
    color_map = {cell_id: tuple(int(255 * c) for c in color) for cell_id, color in zip(all_ids, colors)}

    rgb_frames = []
    for mask in tracked_masks:
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cell_id, color in color_map.items():
            rgb[mask == cell_id] = color
        rgb_frames.append(rgb)
    return rgb_frames

def get_distinct_colors(number_of_colors):
    """Returns number_of_colors different colors
    
    Parameters
    ----------
    
    number_of_colors : int
    
    Returns
    -------
    
    colormap : list
        a list where each entry is a different color
    """
    HSV_tuples = [(color_index*1.0/number_of_colors, 1.0, 0.7) for color_index in range(number_of_colors)]
    RGB_tuples = list(map(lambda color: colorsys.hsv_to_rgb(*color), HSV_tuples))
    np.random.shuffle(RGB_tuples)

    return RGB_tuples

def get_torch_device():
    """
    Get the current PyTorch device.
    
    Returns:
        torch.device: string
            The current device (CPU, CUDA, or MPS).
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    return device

