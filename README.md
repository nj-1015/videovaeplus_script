# videovaeplus_script
Helpful scripts to extract latent data and decode with temporal smooth reconstruction

## Code Usage
This repository provides scripts to encode videos into a latent space and decode latent tensors back into videos, using a strided processing method for temporally smooth results.

### Clone the repository:
```
git clone https://github.com/nj-1015/videovaeplus_script.git
```

### Install dependencies:
Please follow https://github.com/VideoVerses/VideoVAEPlus
Then, copy encode.py and decode.py to the VideoVAEPlus code directory.
```
!git clone https://github.com/VideoVerses/VideoVAEPlus.git
```
```
# Run this cell at the very top of your notebook
!pip install condacolab
import condacolab
condacolab.install()
```
```
# Install gdown if you don't have it
!pip install -q gdown
# Download the file directly to the target path
!gdown --id 1WEKBdRFjEUxwcBgX_thckXklD8s6dDTj -O /content/drive/MyDrive/VideoVAEPlus/ckpt/sota-4-4z.ckpt
```
```
!conda create --name vae python=3.10 -y
```
```
!conda run -n vae pip install -r requirements.txt
!conda run -n vae pip install -q mediapy
```

### Structure
```
├── ckpt/
├── configs/
├── data/
├── evaluatio/
├── examples/
├── scripts/
├── src/
├── utils/
├── encode.py
├── decode.py
├── requirements.txt
└── ...
```

### Encoding: Videos to Latents
The encode.py script converts a directory of videos into their corresponding latent representations. This is the primary data preparation step for training a generative model like a DiT.

```
Arguments:

--data_root: Path to the folder containing your input videos (.mp4, etc.).

--out_root: Path to the folder where the latent files (.npy or .pt) will be saved.

--config_path: Path to the model's .yaml configuration file.

--chunk_size: (Optional, default: 16) Number of frames to process in each chunk.

--stride: (Optional, default: 8) Number of frames to slide the processing window. A stride smaller than the chunk size is required for smooth blending.

--save_format: (Optional, default: numpy) Format to save latents (numpy for .npy, torch for .pt).
```

#### Example:

```
!conda run -n vae python inference_video_latents.py \
  --data_root examples/videos \
  --out_root examples/videos/latents \
  --config_path configs/inference/config_4z.yaml \
  --device cuda:0 \
  --chunk_size 16 \
  --stride 8 \
  --save_latents \
  --resolution 360 640 \
  --save_format numpy
```

This will process all videos in the videos folder and save the resulting latent files in the latents folder.

### Decoding: Latents to Videos
The decode.py script converts a single latent tensor file back into a video. This is used to visualize the output of your generative model.

```
Arguments:

--latent_path: Path to the input latent file (.npy or .pt).

--out_path: Full path, including filename, where the output video will be saved (e.g., reconstructions/output.mp4).

--config_path: Path to the model's .yaml configuration file.
```

#### Example:

```
!conda run -n vae python inference_decode.py \
  --latent_root examples/videos/latents \
  --fps 30 \
  --output_dir examples/videos/latents_decoded \
  --config_path configs/inference/config_4z.yaml \
  --device cuda:0 \
  --chunk_size 4 \
  --stride 8 \
  --save_png_frames
```
