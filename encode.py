import os
import torch
import argparse
import logging
from glob import glob
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from utils.common_utils import instantiate_from_config
from src.modules.t5 import T5Embedder
import torchvision
import mediapy as media

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Video VAE Inference Script with Sliding Window")
    parser.add_argument("--data_root",type=str,required=True,help="Path to the folder containing input videos.")
    parser.add_argument("--out_root", type=str, required=True, help="Path to save reconstructed videos and latents.")
    parser.add_argument("--config_path",type=str,required=True,help="Path to the model configuration file.")
    parser.add_argument("--device",type=str,default="cuda:0",help="Device to run inference on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--chunk_size",type=int,default=16,help="Number of frames per chunk for processing.")
    parser.add_argument("--stride", type=int, default=8, help="Stride for the sliding window in frames. Must be <= chunk_size.")
    parser.add_argument("--resolution",type=int,nargs=2,default=[720, 1280],help="Max resolution to process videos (height, width).")
    parser.add_argument("--save_latents", action='store_true', help="Also save the latent tensors.")
    parser.add_argument(
        "--save_format", 
        type=str, 
        default="numpy", 
        choices=["torch", "numpy"], 
        help="Format to save latents ('torch' for .pt, 'numpy' for .npy for JAX)."
    )
    return parser.parse_args()

def data_processing(video_path, resolution):
    """Load and preprocess video data using mediapy."""
    try:
        video_data = media.read_video(video_path)
        vid_fps = 24.0
        if video_data.metadata:
            vid_fps = video_data.metadata.fps
        
        frames_np = video_data
        
        frames_resized_np = media.resize_video(frames_np, shape=(resolution[0], resolution[1]))
        frames_torch = torch.from_numpy(frames_resized_np).permute(3, 0, 1, 2).float()
        frames_final = (frames_torch / 255.0 - 0.5) * 2.0
        
        return frames_final, vid_fps
    except Exception as e:
        logging.error(f"Error processing video {video_path} with mediapy: {e}")
        return None, None


def save_video(tensor, save_path, fps: float):
    """Save video tensor to a file using mediapy."""
    try:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
        arr = tensor.detach().cpu().squeeze(0).to(torch.uint8)
        arr_for_saving = arr.permute(1, 2, 3, 0).numpy()
        media.write_video(save_path, arr_for_saving, fps=fps)
        logging.info(f"Video saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving video {save_path}: {e}")


def process_in_chunks(video_data, model, chunk_size, stride, device="cuda:0"):
    """
    Processes video in chunks using a sliding window.
    Blends the reconstructed video for smoothness.
    Concatenates the raw latents without blending.
    """
    try:
        num_frames = video_data.size(1)
        logging.info(f"num_frames before pad: {num_frames}")

        # Pad video to make the frame count divisible by chunk_size
        padding_frames = 0
        if (num_frames - chunk_size) % stride != 0:
            padding_needed = stride - ((num_frames - chunk_size) % stride)
            padding = video_data[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
            video_data = torch.cat((video_data, padding), dim=1)
            num_frames = video_data.size(1)
        
        logging.info(f"num_frames after pad: {num_frames}")
        
        # Determine latent shape dynamically
        with torch.no_grad():
            dummy_chunk = video_data[:, :chunk_size, :, :].unsqueeze(0).to(device)
            _, latent_dist = model.forward(dummy_chunk, sample_posterior=False)
            dummy_latent = latent_dist.mode()
            _, C_latent, T_latent, H_latent, W_latent = dummy_latent.shape
            temporal_factor = chunk_size // T_latent
            num_latent_frames = num_frames // temporal_factor

        # Create accumulators on the CPU
        recon_sum = torch.zeros_like(video_data)
        weight_sum_recon = torch.zeros_like(recon_sum)
        output_latent_chunks = []

        # Create a window for smooth blending
        window = torch.ones(chunk_size)
        window = window.view(1, -1, 1, 1) # Reshape for broadcasting

        num_of_chunks = 0
        start_indices = range(0, num_frames - chunk_size + 1, stride)
        for start_f in tqdm(start_indices, desc="Processing Chunks", leave=False):
            end_f = start_f + chunk_size
            chunk = video_data[:, start_f:end_f, :, :]
            
            with torch.no_grad():
                chunk_gpu = chunk.unsqueeze(0).to(device)
                recon_chunk, latent_dist = model.forward(chunk_gpu, sample_posterior=False)
                latent_chunk = latent_dist.mode()

                # Blend the reconstructed video
                recon_sum[:, start_f:end_f, :, :] += recon_chunk.squeeze(0).cpu() * window
                weight_sum_recon[:, start_f:end_f, :, :] += window

                # Store the raw, unblended latent chunk
                output_latent_chunks.append(latent_chunk.cpu())
            
            num_of_chunks += 1

        logging.info(f"num_of_chunks: {num_of_chunks}")
        # Finalize the blended reconstruction
        recon_ret = recon_sum / torch.clamp(weight_sum_recon, min=1e-6)
        
        # Concatenate the raw latents
        latent_ret = torch.cat(output_latent_chunks, dim=2)
        
        return recon_ret.unsqueeze(0), latent_ret
    except Exception as e:
        logging.error(f"Error processing chunks: {e}")
        return None, None


def main():
    """Main function for video VAE inference."""
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    config = OmegaConf.load(args.config_path)

    model = instantiate_from_config(config.model)
    model = model.to(args.device)
    model.eval()

    all_videos = sorted(glob(os.path.join(args.data_root, "*.mp4")))
    if not all_videos:
        logging.error(f"No videos found in {args.data_root}")
        return

    for video_path in tqdm(all_videos, desc="Processing videos", unit="video"):
        logging.info(f"Processing video: {video_path}")
        frames, vid_fps = data_processing(video_path, args.resolution)
        if frames is None:
            continue

        video_name = os.path.basename(video_path).split(".")[0]
        
        stride = args.stride
        if stride > args.chunk_size:
            logging.warning(f"Stride ({stride}) is greater than chunk_size ({args.chunk_size}). Setting stride to chunk_size.")
            stride = args.chunk_size

        with torch.no_grad():
            video_recon, video_latents = process_in_chunks(frames, model, args.chunk_size, stride, device=args.device)

            if video_recon is not None:
                save_path = os.path.join(args.out_root, f"{video_name}_stride{stride}_reconstructed.mp4")
                save_video(video_recon, save_path, vid_fps)

                if args.save_latents and video_latents is not None:
                    if args.save_format == 'numpy':
                        latent_save_path = os.path.join(args.out_root, f"{video_name}_stride{stride}_latents.npy")
                        np.save(latent_save_path, video_latents.squeeze(0).cpu().numpy())
                        logging.info(f"Saved latent shape: {video_latents.squeeze(0).shape}")
                        logging.info(f"Latents saved in NumPy format to {latent_save_path}")
                    else: # Default to torch
                        latent_save_path = os.path.join(args.out_root, f"{video_name}_stride{stride}_latents.pt")
                        torch.save(video_latents.squeeze(0), latent_save_path)
                        logging.info(f"Latents saved in PyTorch format to {latent_save_path}")

if __name__ == "__main__":
    main()
