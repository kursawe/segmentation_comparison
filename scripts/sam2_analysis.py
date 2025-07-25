import os
import imageio
import time
import importlib.resources
from PIL import Image

import hydra
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt

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
    frames = [frame for i, frame in enumerate(reader) if i % 4 == 0 and i >= 240 and i <= 1786]
    
    for idx, frame in enumerate(frames):
        # frame = frame[300:556, 300:556]  # crop to 256x256
        imageio.imwrite(os.path.join(output_dir, f"{idx:04d}.jpg"), frame, quality=100, subsampling=0)
    
    print(f"Downsampled video to {len(frames)} frames and saved to {output_dir}")

def make_folders_for_training_and_testing():
    """
    Downsample the video to a sequence of frames for segmentation.
    """
    # Output directories for images and masks
    output_img_root = os.path.join(script_dir, '..', 'output', 'tiffstack_training')
    output_mask_root = os.path.join(script_dir, '..', 'output', 'masks_training')
    os.makedirs(output_img_root, exist_ok=True)
    os.makedirs(output_mask_root, exist_ok=True)
    
    # Load frames from video
    avi_path = os.path.join(script_dir, '..', 'data', 'movie1_AB060922a_Job3_All_25_fps.avi')
    reader = imageio.get_reader(avi_path)
    all_frames = [frame for i, frame in enumerate(reader) if i % 4 == 0 and i >= 240 and i <= 1783]
    first_frames = all_frames[:320]
    second_frames = all_frames[340:]
    training_frames = first_frames+second_frames

    # Load ground truth masks
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    all_ground_truth_masks = [imageio.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
    first_ground_truth_masks = all_ground_truth_masks[:320]  # match number of frames
    second_ground_truth_masks = all_ground_truth_masks[340:]  # match number of frames
    training_masks = first_ground_truth_masks + second_ground_truth_masks

    # For each frame, create a folder and save the image and mask inside
    for idx, (frame, mask) in enumerate(zip(training_frames, training_masks)):
        img_folder = os.path.join(output_img_root, f"{idx:04d}")
        mask_folder = os.path.join(output_mask_root, f"{idx:04d}")
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        imageio.imwrite(os.path.join(img_folder, f"{idx:04d}.jpg"), frame, quality=100, subsampling=0)
        imageio.imwrite(os.path.join(mask_folder, f"{idx:04d}.png"), mask)

    print(f"Saved {len(training_frames)} training image folders to {output_img_root}")
    print(f"Saved {len(training_masks)} training mask folders to {output_mask_root}")

def create_sam2_masks_unsupervised( model = 'standard'):
    # number_frames = 2
    output_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_' + model)
    os.makedirs(output_dir, exist_ok=True)
 
    t0 = time.time()

    if model == 'standard':
        sam2_checkpoint = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_large.pt')
    else:
        sam2_checkpoint = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_l+abdomen.pt')
    predictor = build_sam2_video_predictor('sam2.1_hiera_l.yaml', sam2_checkpoint, device=torch_device)
    # mask_generator = SAM2AutomaticMaskGenerator(predictor, pred_iou_thresh = 0.1)
    # mask_generator = SAM2AutomaticMaskGenerator(predictor, stability_score_thresh = 0.3)
    mask_generator = SAM2AutomaticMaskGenerator(predictor)
    print(f"Loaded SAM2 model from {sam2_checkpoint} with config {sam2_model_cfg_file}")
    print(f"Time to load SAM2 model: {time.time() - t0:.2f} seconds")

    t1 = time.time()
    tifffolder = os.path.join(script_dir,'..','output','tiffstack_reduced')
    tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')])
    if model == 'trained':
        tiff_files = tiff_files[320:340]
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

def quantify_sam2_performance_unsupervised(model = 'standard'):
    print(f"Quantifying SAM2 performance for model: {model}")
    t0 = time.time()
    # match identified and ground truth masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_' + model)
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
    if model == 'trained':
        ground_truth_masks = ground_truth_masks[320:340]
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
    mp4_path = os.path.join(script_dir, '..', 'output', 'segmentation_comparison_sam2_' + model + '.mp4')
    visualize_segmentation(sam2_masks, ground_truth_masks,mp4_path)
    print(f"Time to create visualization: {time.time() - t3:.2f} seconds")
    print(f"Total time: {time.time() - t3:.2f} seconds")

def make_sam2_tracking_movie_unsupervised(model = 'standard'):
    """
    Create a movie showing the tracking of sam2 masks.
    """
    t0 = time.time()
    print("Creating Cellpose tracking movie for model " + model + " ...")
    # Load sam2 masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_' + model)
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
    output_movie_path = os.path.join(script_dir, '..', 'output', 'sam2_tracking_' + model + '.mp4')
    imageio.mimsave(output_movie_path, rgb_frames, fps=5)
    print(f"Saved tracking movie to {output_movie_path}")
    print(f"Time to create movie: {time.time() - t3:.2f} seconds")

def create_sam2_masks_prompted(model = 'standard'):
    # number_frames = 2
    output_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_' + model + '_prompted')
    os.makedirs(output_dir, exist_ok=True)
 
    t0 = time.time()
    if model == 'standard':
        sam2_checkpoint = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_large.pt')
    else:
        sam2_checkpoint = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_l+abdomen.pt')

    predictor = build_sam2_video_predictor('sam2.1_hiera_l.yaml', sam2_checkpoint, device=torch_device)
    print(f"Loaded SAM2 model from {sam2_checkpoint} with config {sam2_model_cfg_file}")
    print(f"Time to load SAM2 model: {time.time() - t0:.2f} seconds")

    t1 = time.time()
    tifffolder = os.path.join(script_dir,'..','output','tiffstack_reduced')
    tiff_files = sorted([f for f in os.listdir(tifffolder) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')])
    if model == 'trained':
        tiff_files = tiff_files[320:340]
    first_tiff_file = tiff_files[0]
    first_image = Image.open(os.path.join(tifffolder, first_tiff_file))
    first_image = np.array(first_image.convert("RGB"))
    inference_state = predictor.init_state(video_path=tifffolder, offload_video_to_cpu=True, offload_state_to_cpu=True )
    print('generated inference state')

    # get ground truth starting point
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    if model == 'trained':
        gt_mask_files = gt_mask_files[320:340]
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

def quantify_sam2_performance_prompted(model = 'standard'):
    print(f"Quantifying supervised SAM2 performance for model: {model}")
    t0 = time.time()
    # match identified and ground truth masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_' + model + '_prompted')
    sam2_mask_files = sorted([f for f in os.listdir(sam2_mask_dir) if f.endswith('.png')])
    sam2_masks = [imageio.imread(os.path.join(sam2_mask_dir, f)) for f in sam2_mask_files]
    print(f"Time to load sam2 masks: {time.time() - t0:.2f} seconds")
 
    t1 = time.time()
    # match identified and ground truth masks
    ground_truth_mask_dir = os.path.join(script_dir, '..', 'data', 'masks')
    gt_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith('.png')])
    ground_truth_masks = [imageio.imread(os.path.join(ground_truth_mask_dir, f)) for f in gt_mask_files]
    if model == 'trained':
        ground_truth_masks = ground_truth_masks[320:340]
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
    mp4_path = os.path.join(script_dir, '..', 'output', 'segmentation_comparison_sam2_' + model + '_prompted.mp4')
    visualize_segmentation(sam2_masks, ground_truth_masks,mp4_path)
    print(f"Time to create visualization: {time.time() - t3:.2f} seconds")
    print(f"Total time: {time.time() - t3:.2f} seconds")
    
def make_sam2_tracking_movie_prompted(model = 'standard'):
    """
    Create a movie showing the tracking of sam2 masks.
    """
    t0 = time.time()
    print("Creating Cellpose tracking movie...")
    # Load sam2 masks
    sam2_mask_dir = os.path.join(script_dir, '..', 'output', 'sam2_masks_' + model + '_prompted')
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
    output_movie_path = os.path.join(script_dir, '..', 'output', 'sam2_tracking_' + model + '_prompted.mp4')
    imageio.mimsave(output_movie_path, rgb_frames, fps=5)
    print(f"Saved tracking movie to {output_movie_path}")
    print(f"Time to create movie: {time.time() - t3:.2f} seconds")

def plot_tensorboard_losses():

    log_dir = os.path.join(script_dir, '..', 'output','sam2_logs','tensorboard')
    # log_dir = "/tensorboard/"  # your log folder path

    tags_of_interest = [
        "Losses/train_all_loss",
         # "Losses/train_all_loss_mask",
         "Losses/train_all_loss_dice",
         "Losses/train_all_loss_iou",
         # "Losses/train_all_loss_class",
         "Losses/train_all_core_loss"
    ]
    
    scalars = {}  # to collect all scalar metrics
    
    # Load each event file
    for file in os.listdir(log_dir):
        if file.startswith("events.out"):
            ea = event_accumulator.EventAccumulator(os.path.join(log_dir, file))
            ea.Reload()
    
            for tag in tags_of_interest:
                if tag in ea.Tags().get("tensors", []):
                    tensor_events = ea.Tensors(tag)
                    steps = [e.step for e in tensor_events]
                    values = [e.tensor_proto.float_val[0] for e in tensor_events]
    
                    if tag not in scalars:
                        scalars[tag] = {"steps": steps, "values": values}
                    else:
                        scalars[tag]["steps"] += steps
                        scalars[tag]["values"] += values
    
#    for tag, data in scalars.items():
#        combined = sorted(zip(data["steps"], data["values"]))
#        scalars[tag]["steps"], scalars[tag]["values"] = zip(*combined)
    # Plot all found scalar tags

    for tag, data in scalars.items():
        if "core_loss" in tag:
            plt.plot(data["steps"], data["values"], label=tag.split("/")[-1], ls = "--")
        else:
            plt.plot(data["steps"], data["values"], label=tag.split("/")[-1])
    
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("TensorBoard Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, '..', 'output', 'tensorboard_metrics.pdf'))

def rewrite_checkpoint_memory_parameters():

    # === File paths ===
    source_ckpt_path = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_large.pt') 
    target_ckpt_path = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_l+abdomen.pt') 
    output_ckpt_path = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_l+abdomen_memory.pt') 

    # === Load checkpoints ===
    source_ckpt = torch.load(source_ckpt_path, map_location="cpu")
    target_ckpt = torch.load(target_ckpt_path, map_location="cpu")

    # === Sanity check ===
    if "model" in source_ckpt:
        source_state = source_ckpt["model"]
    else:
        source_state = source_ckpt

    if "model" in target_ckpt:
        target_state = target_ckpt["model"]
    else:
        target_state = target_ckpt

    # === Inject memory attention weights ===
    total_diff = 0.0
    for key in source_state:
        if "memory" in key and key in target_state:
            src_param = source_state[key]
            tgt_param = target_state[key]

            if src_param.shape != tgt_param.shape:
                print(f"Skipping {key}: shape mismatch {src_param.shape} vs {tgt_param.shape}")
                continue

            # Move to device if needed (CPU or GPU)
            src_tensor = src_param.to("cuda") if torch.cuda.is_available() else src_param
            tgt_tensor = tgt_param.to("cuda") if torch.cuda.is_available() else tgt_param

            diff = torch.nn.functional.mse_loss(src_tensor, tgt_tensor, reduction='sum')  # L2 squared
            total_diff += diff.item()

            print(f"Injecting {key}, delta_L2²={diff.item():.6f}")
            target_state[key] = src_param

    # === Save updated checkpoint ===
    # Optionally wrap in "model" key if needed
    print(f"Total L2² difference between memory parameters: {total_diff:.6f}")
    updated_ckpt = {"model": target_state}
    torch.save(updated_ckpt, output_ckpt_path)
    print(f"Updated checkpoint saved to {output_ckpt_path}")

def quantify_checkpoint_differences():

    # === File paths ===
    source_ckpt_path = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_large.pt') 
    target_ckpt_path = os.path.join(script_dir, '..', 'models', 'sam2.1_hiera_l+abdomen.pt') 

    # === Load checkpoints ===
    source_ckpt = torch.load(source_ckpt_path, map_location="cpu")
    target_ckpt = torch.load(target_ckpt_path, map_location="cpu")

    # === Sanity check ===
    if "model" in source_ckpt:
        source_state = source_ckpt["model"]
    else:
        source_state = source_ckpt

    if "model" in target_ckpt:
        target_state = target_ckpt["model"]
    else:
        target_state = target_ckpt

    # === Inject memory attention weights ===
    total_diff = 0.0
    for key in source_state:
        if key in target_state:
            src_param = source_state[key]
            tgt_param = target_state[key]

            if src_param.shape != tgt_param.shape:
                print(f"Skipping {key}: shape mismatch {src_param.shape} vs {tgt_param.shape}")
                continue

            # Move to device if needed (CPU or GPU)
            src_tensor = src_param.to("cuda") if torch.cuda.is_available() else src_param
            tgt_tensor = tgt_param.to("cuda") if torch.cuda.is_available() else tgt_param

            diff = torch.nn.functional.mse_loss(src_tensor, tgt_tensor, reduction='sum')  # L2 squared
            total_diff += diff.item()

            print(f"difference for {key}: delta_L2²={diff.item():.6f}")

    # === Save updated checkpoint ===
    # Optionally wrap in "model" key if needed
    print(f"Total L2² difference between memory parameters: {total_diff:.6f}")

if __name__ == "__main__":
    # down_sample_video_to_segmented_frames()
    # make_folders_for_training_and_testing()
    # create_sam2_masks_unsupervised(model = 'standard')
        #create_sam2_masks_unsupervised(model = 'trained')
    # quantify_sam2_performance_unsupervised(model = 'standard')
        #quantify_sam2_performance_unsupervised(model = 'trained')
    # make_sam2_tracking_movie_unsupervised(model = 'standard')
        #make_sam2_tracking_movie_unsupervised(model = 'trained')
    # create_sam2_masks_prompted(model = 'standard')
        #create_sam2_masks_prompted(model = 'trained')
    # quantify_sam2_performance_prompted(model = 'standard')
        #quantify_sam2_performance_prompted(model = 'trained')
    # make_sam2_tracking_movie_prompted(model = 'standard')
    #    make_sam2_tracking_movie_prompted(model = 'trained')
    # plot_tensorboard_losses()
    # rewrite_checkpoint_memory_parameters()
    quantify_checkpoint_differences()
