import torch
import numpy as np
import os
import argparse
import logging
from glob import glob
from omegaconf import OmegaConf
from tqdm import tqdm
import torchvision
from decord import VideoReader, cpu
import mediapy as media
import re
from PIL import Image

# --- Import necessary components from VideoVAEPlus ---
# Make sure these files are in the same directory or accessible.
from utils.common_utils import instantiate_from_config

# --- Basic Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def parse_args():
    """Parse command-line arguments for the decoder script."""
    parser = argparse.ArgumentParser(description="Chunk-based Latent to Video Decoder Script")
    parser.add_argument(
        "--latent_root",
        type=str,
        required=True,
        help="Path to the folder containing the .npy latent files to decode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the decoded videos.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the VideoVAE+ configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the PyTorch VAE model on (e.g., 'cuda:0').",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="Number of latent frames to decode per chunk (should match encoder).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride in VIDEO FRAMES for the blending window.",
    )
    parser.add_argument(
        "--save_png_frames",
        action='store_true',
        help="If set, saves each frame of the decoded video as a PNG file."
    )
    return parser.parse_args()


def save_video(tensor, save_path, fps: float):
    """Save video tensor to a file using mediapy."""
    try:
        # Denormalize from [-1, 1] to [0, 255]
        tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
        # Detach, move to CPU, remove batch dim, convert to uint8
        arr = tensor.detach().cpu().squeeze(0).to(torch.uint8)
        # Permute from [C, T, H, W] to mediapy's expected [T, H, W, C]
        arr_for_saving = arr.permute(1, 2, 3, 0).numpy()
        
        media.write_video(save_path, arr_for_saving, fps=fps)
        logging.info(f"Video saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving video {save_path}: {e}")


def save_images(tensor, output_dir, video_name):
    """Saves each frame of a video tensor as a PNG image."""
    try:
        frames_dir = os.path.join(output_dir, f"{video_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Denormalize from [-1, 1] to [0, 255]
        tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
        # Detach, move to CPU, remove batch dim, convert to uint8
        arr = tensor.detach().cpu().squeeze(0).to(torch.uint8)
        
        logging.info(f"Saving {arr.shape[1]} frames to '{frames_dir}'...")
        # Iterate through the temporal dimension (T) of the [C, T, H, W] tensor
        for i in range(arr.shape[1]):
            # Get a single frame and permute from [C, H, W] to [H, W, C] for PIL
            frame = arr[:, i, :, :].permute(1, 2, 0).numpy()
            img = Image.fromarray(frame)
            img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        logging.info("✅ PNG frames saved successfully.")
    except Exception as e:
        logging.error(f"Error saving PNG frames: {e}")


def main():
    """Main function for decoding latents into videos in chunks."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Initializing VideoVAE+ model from config...")
    config = OmegaConf.load(args.config_path)
    model = instantiate_from_config(config.model)
    model = model.to(args.device)
    model.eval()

    all_latent_files = sorted(glob(os.path.join(args.latent_root, "*.npy")))
    if not all_latent_files:
        logging.error(f"No .npy files found in {args.latent_root}")
        return

    for latent_path in tqdm(all_latent_files, desc="Decoding Latents"):
        logging.info(f"Processing latent file: {latent_path}")
        latent_cpu = torch.from_numpy(np.load(latent_path))
        num_latent_frames = latent_cpu.shape[1]
        
        # Determine VAE's temporal upsampling factor from the first chunk
        with torch.no_grad():
            first_chunk = latent_cpu[:, :args.chunk_size, :, :].unsqueeze(0).to(args.device)
            first_decoded_chunk = model.decode(first_chunk)
            video_chunk_size = first_decoded_chunk.shape[2]
            temporal_upsampling_factor = video_chunk_size // args.chunk_size
            logging.info(f"Detected VAE temporal upsampling factor: {temporal_upsampling_factor}x")

        # Create accumulators on the CPU
        num_chunks = (num_latent_frames - args.chunk_size) // args.chunk_size + 1
        num_output_frames = (num_chunks - 1) * args.stride + video_chunk_size

        output_shape = (1, 3, num_output_frames, latent_cpu.shape[2] * 8, latent_cpu.shape[3] * 8)
        decoded_sum = torch.zeros(output_shape, dtype=torch.float32)
        weight_sum = torch.zeros(output_shape, dtype=torch.float32)
        
        # Create a window for blending
        video_stride = args.stride
        window = torch.ones(video_chunk_size)
        window = window.view(1, 1, -1, 1, 1)

        # Loop through the latent file in non-overlapping chunks
        latent_start_indices = range(0, num_latent_frames, args.chunk_size)
        
        for i, start_f_latent in enumerate(tqdm(latent_start_indices, desc="Decoding and Blending", leave=False)):
            end_f_latent = start_f_latent + args.chunk_size
            if end_f_latent > num_latent_frames: continue
            
            latent_chunk = latent_cpu[:, start_f_latent:end_f_latent, :, :].unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                decoded_chunk = model.decode(latent_chunk)
            
            # Calculate the position in the final video
            start_f_video = i * video_stride
            end_f_video = start_f_video + video_chunk_size
            
            # Apply window and add to accumulators
            decoded_sum[:, :, start_f_video:end_f_video, :, :] += decoded_chunk.cpu() * window
            weight_sum[:, :, start_f_video:end_f_video, :, :] += window

        decoded_video = decoded_sum / torch.clamp(weight_sum, min=1e-6)

        video_fps = args.fps
        latent_name = os.path.basename(latent_path).split(".")[0]
        
        video_name = latent_name.replace('_latents', '_decoded')
        save_path = os.path.join(args.output_dir, f"{video_name}.mp4")

        save_video(decoded_video, save_path, video_fps)

        if args.save_png_frames:
            save_images(decoded_video, args.output_dir, video_name)


if __name__ == "__main__":
    main()
