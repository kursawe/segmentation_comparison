import cellpose.models
import cellpose.io
import cellpose.train
import os
import imageio
import time
from matplotlib import pyplot as plt
from skimage.morphology import binary_dilation as skimage_dilation
from skimage.morphology import disk as skimage_disk

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
    images = images[240:1785:4]
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
    
def train_cellpose_sam(use_reduced_masks=False):
    """
    Train a Cellpose model using Rebeeca's data
    """
    t0 = time.time()
    print("Training Cellpose model")
    
    model = cellpose.models.CellposeModel(gpu=True)

    if use_reduced_masks:
        model_name = "abdomen_model_reduced_masks"
    else:
        model_name = "abdomen_model"

    # default training params
    n_epochs = 30
    # n_epochs = 2
    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 1
    
    # training data:
    if use_reduced_masks:
        print("Using reduced masks for training")
        tifffolder = os.path.join(script_dir, '..', 'output', 'cellpose_training_data')
        training_mask_files = sorted([f for f in os.listdir(tifffolder) if f.endswith('.png')])
        images = [cellpose.io.imread(os.path.join(tifffolder, f)) for f in training_mask_files]
    else:
        tifffolder = os.path.join(script_dir,'..','data','tiffstack')
        tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')])
        images = [cellpose.io.imread(os.path.join(tifffolder, f)) for f in tiff_files]
        images = images[240:1784:4]
    print(f"Loaded {len(images)} frames from {tifffolder}")
    
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    ground_truth_masks = [cellpose.io.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
    # ground_truth_masks = ground_truth_masks[:len(cellpose_masks)]  # match number of frames

    training_data = images[:320] + images[340:]  # select frames for training
    training_labels = ground_truth_masks[:320] + ground_truth_masks[340:]  # select corresponding masks for training
    test_data = images[320:340]  # select frames for testing
    test_labels = ground_truth_masks[320:340]  # select corresponding masks for testing
    
    # training_data = training_data[:1]
    # training_labels = training_labels[:1]
    # test_data = test_data[:1]
    # test_labels = test_labels[:1]

    new_model_path, train_losses, test_losses = cellpose.train.train_seg(model.net,
                                                            train_data=training_data,
                                                            train_labels=training_labels,
                                                            test_data=test_data,
                                                            test_labels=test_labels,
                                                            batch_size=batch_size,
                                                            n_epochs=n_epochs,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            nimg_per_epoch=len(training_data), # can change this
                                                            save_path = os.path.join(script_dir,'..','models'),
                                                            model_name=model_name)
    print(f"train losses are {train_losses}")
    print(f"test losses are {test_losses}")
    
    print(f"Training completed. Model saved to {new_model_path}")
    
    plt.figure(figsize=(4.5,2.5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, '..', 'output', f'{model_name}_losses.pdf'))

def quantify_trained_cellpose_performance(use_reduced_masks=False):
    """
    Quantify the performance of the trained Cellpose model.
    """
    t0 = time.time()
    print("Quantifying performance of trained Cellpose model...")
    
    # Load the trained model
    if use_reduced_masks:
        new_model_path = os.path.join(script_dir,'..', 'models', 'abdomen_model_reduced_masks')
    else:
        new_model_path = os.path.join(script_dir,'..', 'models', 'abdomen_model')
    model = cellpose.models.CellposeModel(gpu=True,
                                          pretrained_model = new_model_path)
    
    # Load test data
    tifffolder = os.path.join(script_dir,'..','data','tiffstack')
    tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')])
    images = [cellpose.io.imread(os.path.join(tifffolder, f)) for f in tiff_files]
    images = images[240:1784:4]
    print(f"Loaded {len(images)} frames from {tifffolder}")
    validation_images = images[320:340]  # match number of frames
    # validation_images = validation_images[:1]  # for testing, use only one frame
    
    # Load ground truth masks
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    ground_truth_masks = [cellpose.io.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
    ground_truth_masks = ground_truth_masks[320:340]  # match number of frames
    
    # Evaluate the model on the test data
    cellpose_masks = []
    for image in validation_images:
        mask, _, _ = model.eval(image)
        cellpose_masks.append(mask)
    
    # Quantification
    percent_found, global_iot = quantify_segmentation(cellpose_masks, ground_truth_masks)
    
    print(f"Percentage of ground truth masks found: {(100*percent_found):.2f}%")
    print(f"Global IoT score: {global_iot:.4f}")
    if use_reduced_masks:
        mp4_path = os.path.join(script_dir, '..', 'output', 'segmentation_comparison_improved_trained_cellpose.mp4')
    else:
        mp4_path = os.path.join(script_dir, '..', 'output', 'segmentation_comparison_trained_cellpose.mp4')
    visualize_segmentation(cellpose_masks, ground_truth_masks,mp4_path)
 
    print("Tracking cells over time...")
    tracked_masks = track_cells_over_time(cellpose_masks)
    
    print("Converting tracked masks to RGB frames...")
    rgb_frames = tracked_masks_to_rgb(tracked_masks)
    
    print("Creating movie from tracked masks...")
    # Create a movie from the masks
    if use_reduced_masks:
        output_movie_path = os.path.join(script_dir, '..', 'output', 'cellpose_trained_tracking_reduced_masks.mp4')
    else:
        output_movie_path = os.path.join(script_dir, '..', 'output', 'cellpose_trained_tracking.mp4')
    imageio.mimsave(output_movie_path, rgb_frames, fps=5)
    print(f"Saved tracking movie to {output_movie_path}")
    
def make_cellpose_training_data():
    """
    Loads a movie and ground truth masks, dilates the ground truth mask, and applies it to the movie frames.
    Only keeps intensity data where ground truth cells exist (with dilation).
    """
    output_dir = os.path.join(script_dir, '..', 'output', 'cellpose_training_data')
    os.makedirs(output_dir, exist_ok=True)

    # Load movie frames (assuming TIFF stack or list of images)
    tifffolder = os.path.join(script_dir,'..','data','tiffstack')
    tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')])
    images = [cellpose.io.imread(os.path.join(tifffolder, f)) for f in tiff_files]
    images = images[240:1785:4]

    # Load ground truth masks
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    ground_truth_masks = [cellpose.io.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
 
    masked_frames = []
    masked_frame = np.zeros_like(images[0])
    for index, (frame, gt_mask) in enumerate(zip(images, ground_truth_masks)):
        # Make binary mask: 1 where any cell exists
        binary_mask = (gt_mask > 0)
        # Dilate the mask
        dilated_mask = skimage_dilation(binary_mask, skimage_disk(5))
        # Apply mask to frame
        masked_frame[:] = 0
        masked_frame[dilated_mask] = frame[dilated_mask]
        masked_frames.append(np.copy(masked_frame))
        # Optionally save each masked frame
        cellpose.io.imsave(os.path.join(output_dir, f"masked_frame_{index:04d}.png"), masked_frame)
    
    
    output_movie_path = os.path.join(script_dir, '..', 'output', 'cellpose_new_training_data.mp4')
    imageio.mimsave(output_movie_path, masked_frames, fps=5)

if __name__ == "__main__":
    # create_cellpose_masks()
    # quantify_cellpose_performance()
    # make_cellpose_tracking_movie()
    # make_cellpose_training_data()
    # train_cellpose_sam(use_reduced_masks = False)
    # train_cellpose_sam(use_reduced_masks = True)
    quantify_trained_cellpose_performance(use_reduced_masks = False)
    quantify_trained_cellpose_performance(use_reduced_masks = True)
