import os
import imageio
import time
import importlib.resources
from PIL import Image

import hydra
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

script_dir = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(os.path.join(script_dir, '..'))
from cell_seg_comparison import *

sam2_checkpoint = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_large.pt')
with importlib.resources.path('sam2','configs') as cfg_path:
    sam2_model_cfg_file = os.path.join(cfg_path, 'sam2.1', 'sam2.1_hiera_l.yaml')
    sam2_config_path = os.path.join(cfg_path, 'sam2.1')

# print(f"Using SAM2 model config: {sam2_model_cfg_file}")
# sam2_model_config = OmegaConf.load(sam2_model_cfg_file)
# sam2_model_cfg_file = os.path.join(script_dir, '..', 'configs', 'sam2.1_hiera_l.yaml')
# sam2_config_path = os.path.join(script_dir, '..', 'configs')
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(sam2_config_path)

# hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
# hydra.initialize_config_module(sam2_config_path, version_base=None)

torch_device = get_torch_device()

def down_sample_video_to_segmented_frames():
    """
    Downsample the video to a sequence of frames for segmentation.
    """
    output_dir = os.path.join(script_dir, '..', 'output', 'tiffstack_reduced')
    os.makedirs(output_dir, exist_ok=True)
    
    avi_path = os.path.join(script_dir,'..','data','movie1_AB060922a_Job3_All_25_fps.avi')
    reader = imageio.get_reader(avi_path)
    
    # downsample by taking every 4th frame
    frames = [frame for i, frame in enumerate(reader) if i % 4 == 0]
    
    for idx, frame in enumerate(frames):
        # frame = frame[300:556, 300:556]  # crop to 256x256
        imageio.imwrite(os.path.join(output_dir, f"{idx:04d}.jpg"), frame, quality=100, subsampling=0)
    
    print(f"Downsampled video to {len(frames)} frames and saved to {output_dir}")

def create_sam2_masks_unsupervised():
    # number_frames = 2
    output_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks')
    os.makedirs(output_dir, exist_ok=True)
 
    t0 = time.time()
    predictor = build_sam2_video_predictor('sam2.1_hiera_l.yaml', sam2_checkpoint, device=torch_device)
    # mask_generator = SAM2AutomaticMaskGenerator(predictor, pred_iou_thresh = 0.1)
    mask_generator = SAM2AutomaticMaskGenerator(predictor, stability_score_thresh = 0.3)
    print(f"Loaded SAM2 model from {sam2_checkpoint} with config {sam2_model_cfg_file}")
    print(f"Time to load SAM2 model: {time.time() - t0:.2f} seconds")

    t1 = time.time()
    tifffolder = os.path.join(script_dir,'..','output','tiffstack_reduced')
    tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')])
    first_tiff_file = tiff_files[0]
    first_image = Image.open(os.path.join(tifffolder, first_tiff_file))
    first_image = np.array(first_image.convert("RGB"))
    inference_state = predictor.init_state(video_path=tifffolder, offload_video_to_cpu=True, offload_state_to_cpu = True)
    print('generated inference state')
    first_masks = mask_generator.generate(first_image)
    print(f"Generated {len(first_masks)} masks for the first image.")
    print(f"Time to load first image and generate masks: {time.time() - t1:.2f} seconds")

    # for each returned mask, provide it to the video predictor as a starting point
    for cell_id, mask in enumerate(first_masks):
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state = inference_state,
            frame_idx = 0,
            obj_id = cell_id,
            mask = mask['segmentation'])
    print(f"Time to load first image and generate masks: {time.time() - t1:.2f} seconds")

    t2 = time.time()
    # run propagation throughout the video and collect the results in a dict
    # segmented_frames = []  # video_segments contains the per-frame segmentation results
    current_frame = np.zeros_like(first_image[:,:,0], dtype=np.uint16)  # create a blank frame for segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        current_frame[:] = 0  # reset the current frame 
        for out_obj_id, these_out_mask_logits in zip(out_obj_ids, out_mask_logits):
            # out_mask_logits is a tensor of shape (H, W) with logits for the mask
            # convert logits to binary mask
            current_frame[these_out_mask_logits[0].cpu().numpy() > 0.0] = out_obj_id +1
        # segmented_frames.append(current_frame)
        this_PIL_image = Image.fromarray(current_frame)
        # save the masks as PNG files
        this_PIL_image.save(os.path.join(output_dir, f"mask_{out_frame_idx:04d}.png"))
        # stop after processing the first `number_frames` frames
        # if out_frame_idx >= number_frames - 1:
         #    break

    # print(f"Segmented {number_frames} frames.")
    print(f"Segmented all frames.")
    print(f"Time to segment images: {time.time() - t2:.2f} seconds")

def quantify_sam2_performance_unsupervised():
    t0 = time.time()
    # match identified and ground truth masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks')
    sam2_mask_files = sorted([f for f in os.listdir(sam2_mask_dir) if f.endswith('.png')])
    sam2_masks = [imageio.imread(os.path.join(sam2_mask_dir, f)) for f in sam2_mask_files]
    print(f"Time to load sam2 masks: {time.time() - t0:.2f} seconds")
 
    t1 = time.time()
    # match identified and ground truth masks
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    ground_truth_masks = [imageio.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
    ground_truth_masks = ground_truth_masks[:len(sam2_masks)]  # match number of frames
    ground_truth_masks_shrunk = []
    for mask in ground_truth_masks:
        # mask = mask[300:556, 300:556]  # crop to 256x256
        ground_truth_masks_shrunk.append(mask)
    ground_truth_masks = ground_truth_masks_shrunk
    print(f"Loaded {len(ground_truth_masks)} ground truth masks from {ground_truth_mask_dir}")
    print(f"Time to load ground truth masks: {time.time() - t1:.2f} seconds")
 
    t2 = time.time()
    # Quantification
    percent_found, global_iot = quantify_segmentation(sam2_masks, ground_truth_masks)
    print(f"Percentage of ground truth masks found: {(100*percent_found):.2f}%")
    print(f"Global IoT score: {global_iot:.4f}")
    print(f"Time to quantify segmentation: {time.time() - t2:.2f} seconds")

    t3 = time.time()
    # Visualization
    mp4_path = os.path.join(script_dir, '..', 'output', 'segmentation_comparison_sam2.mp4')
    visualize_segmentation(sam2_masks, ground_truth_masks,mp4_path)
    print(f"Time to create visualization: {time.time() - t3:.2f} seconds")
    print(f"Total time: {time.time() - t3:.2f} seconds")
    
def make_sam2_tracking_movie_unsupervised():
    """
    Create a movie showing the tracking of sam2 masks.
    """
    t0 = time.time()
    print("Creating Cellpose tracking movie...")
    # Load sam2 masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks')
    sam2_mask_files = sorted([f for f in os.listdir(sam2_mask_dir) if f.endswith('.png')])
    sam2_masks = [imageio.imread(os.path.join(sam2_mask_dir, f)) + 1 for f in sam2_mask_files]
    print(f"Loaded {len(sam2_masks)} sam2 masks from {sam2_mask_dir}")
    print(f"Time to load sam2 masks: {time.time() - t0:.2f} seconds")
    
    print("Converting tracked masks to RGB frames...")
    t2 = time.time()
    rgb_frames = tracked_masks_to_rgb(sam2_masks)
    print(f"Time to convert masks to RGB: {time.time() - t2:.2f} seconds")
    
    t3 = time.time()
    print("Creating movie from tracked masks...")
    # Create a movie from the masks
    output_movie_path = os.path.join(script_dir, '..', 'output', 'sam2_tracking.mp4')
    imageio.mimsave(output_movie_path, rgb_frames, fps=5)
    print(f"Saved tracking movie to {output_movie_path}")
    print(f"Time to create movie: {time.time() - t3:.2f} seconds")

def create_sam2_masks_prompted():
    # number_frames = 2
    output_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_prompted')
    os.makedirs(output_dir, exist_ok=True)
 
    t0 = time.time()
    predictor = build_sam2_video_predictor('sam2.1_hiera_l.yaml', sam2_checkpoint, device=torch_device)
    print(f"Loaded SAM2 model from {sam2_checkpoint} with config {sam2_model_cfg_file}")
    print(f"Time to load SAM2 model: {time.time() - t0:.2f} seconds")

    t1 = time.time()
    tifffolder = os.path.join(script_dir,'..','output','tiffstack_reduced')
    tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')])
    first_tiff_file = tiff_files[0]
    first_image = Image.open(os.path.join(tifffolder, first_tiff_file))
    first_image = np.array(first_image.convert("RGB"))
    inference_state = predictor.init_state(video_path=tifffolder, offload_video_to_cpu=True, offload_state_to_cpu=True )
    print('generated inference state')

    # get ground truth starting point
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    first_gt_mask_file = gt_mask_files[0]
    first_gt_mask = imageio.imread(os.path.join(ground_truth_mask_dir, first_gt_mask_file))
    # first_gt_mask = first_gt_mask[300:556, 300:556]  # crop to 256x256
    print(f"The first gt mask has unique values: {np.unique(first_gt_mask)} and shape {first_gt_mask.shape}")
 
    # for each returned mask, provide it to the video predictor as a starting point
    for cell_id, label in enumerate(np.unique(first_gt_mask)[1:], start = 1):  # skip background
        this_mask = first_gt_mask == label  # create a binary mask for the cell
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state = inference_state,
            frame_idx = 0,
            obj_id = cell_id,
            mask = this_mask)
    print(f"Time to load first image and generate masks: {time.time() - t1:.2f} seconds")

    t2 = time.time()
    # run propagation throughout the video and collect the results in a dict
    # segmented_frames = []  # video_segments contains the per-frame segmentation results
    current_frame = np.zeros_like(first_image[:,:,0], dtype=np.uint16)  # create a blank frame for segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        current_frame[:] = 0  # reset the current frame 
        for out_obj_id, these_out_mask_logits in zip(out_obj_ids, out_mask_logits):
            # out_mask_logits is a tensor of shape (H, W) with logits for the mask
            # convert logits to binary mask
            current_frame[these_out_mask_logits[0].cpu().numpy() > 0.0] = out_obj_id
        # segmented_frames.append(current_frame)
        this_PIL_image = Image.fromarray(current_frame)
        # save the masks as PNG files
        this_PIL_image.save(os.path.join(output_dir, f"mask_{out_frame_idx:04d}.png"))
        # stop after processing the first `number_frames` frames
        # if out_frame_idx >= number_frames - 1:
        #     break

    # print(f"Segmented {number_frames} frames.")
    print(f"Segmented all frames.")
    print(f"Time to segment images: {time.time() - t2:.2f} seconds")

def quantify_sam2_performance_prompted():
    t0 = time.time()
    # match identified and ground truth masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_prompted')
    sam2_mask_files = sorted([f for f in os.listdir(sam2_mask_dir) if f.endswith('.png')])
    sam2_masks = [imageio.imread(os.path.join(sam2_mask_dir, f)) for f in sam2_mask_files]
    print(f"Time to load sam2 masks: {time.time() - t0:.2f} seconds")
 
    t1 = time.time()
    # match identified and ground truth masks
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    ground_truth_masks = [imageio.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
    ground_truth_masks = ground_truth_masks[:len(sam2_masks)]  # match number of frames
    ground_truth_masks_shrunk = []
    for mask in ground_truth_masks:
        # mask = mask[300:556, 300:556]  # crop to 256x256
        ground_truth_masks_shrunk.append(mask)
    ground_truth_masks = ground_truth_masks_shrunk
    print(f"Loaded {len(ground_truth_masks)} ground truth masks from {ground_truth_mask_dir}")
    print(f"Time to load ground truth masks: {time.time() - t1:.2f} seconds")
 
    t2 = time.time()
    # Quantification
    percent_found, global_iot = quantify_segmentation(sam2_masks, ground_truth_masks)
    print(f"Percentage of ground truth masks found: {(100*percent_found):.2f}%")
    print(f"Global IoT score: {global_iot:.4f}")
    print(f"Time to quantify segmentation: {time.time() - t2:.2f} seconds")

    t3 = time.time()
    # Visualization
    mp4_path = os.path.join(script_dir, '..', 'output', 'segmentation_comparison_sam2_prompted.mp4')
    visualize_segmentation(sam2_masks, ground_truth_masks,mp4_path)
    print(f"Time to create visualization: {time.time() - t3:.2f} seconds")
    print(f"Total time: {time.time() - t3:.2f} seconds")
    
def make_sam2_tracking_movie_prompted():
    """
    Create a movie showing the tracking of sam2 masks.
    """
    t0 = time.time()
    print("Creating Cellpose tracking movie...")
    # Load sam2 masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_prompted')
    sam2_mask_files = sorted([f for f in os.listdir(sam2_mask_dir) if f.endswith('.png')])
    sam2_masks = [imageio.imread(os.path.join(sam2_mask_dir, f)) for f in sam2_mask_files]
    print(f"Loaded {len(sam2_masks)} sam2 masks from {sam2_mask_dir}")
    print(f"Time to load sam2 masks: {time.time() - t0:.2f} seconds")
    
    print("Converting tracked masks to RGB frames...")
    t2 = time.time()
    rgb_frames = tracked_masks_to_rgb(sam2_masks)
    print(f"Time to convert masks to RGB: {time.time() - t2:.2f} seconds")
    
    t3 = time.time()
    print("Creating movie from tracked masks...")
    # Create a movie from the masks
    output_movie_path = os.path.join(script_dir, '..', 'output', 'sam2_tracking_prompted.mp4')
    imageio.mimsave(output_movie_path, rgb_frames, fps=5)
    print(f"Saved tracking movie to {output_movie_path}")
    print(f"Time to create movie: {time.time() - t3:.2f} seconds")


if __name__ == "__main__":
    down_sample_video_to_segmented_frames()
    create_sam2_masks_unsupervised()
    quantify_sam2_performance_unsupervised()
    make_sam2_tracking_movie_unsupervised()
    create_sam2_masks_prompted()
    quantify_sam2_performance_prompted()
    make_sam2_tracking_movie_prompted()
