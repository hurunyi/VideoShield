# VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking
Official implementation of **“VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking”**.

## Video Examples
### ModelScope
| Watermarked | Tampered | GT Mask | Pred Mask |
|-------------|----------|---------|-----------|
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/00/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/00/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/00/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/00/mask_pred.gif" width="150"> |
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/01/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/01/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/01/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/01/mask_pred.gif" width="150"> |
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/02/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/02/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/02/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/02/mask_pred.gif" width="150"> |
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/03/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/03/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/03/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/modelscope/03/mask_pred.gif" width="150"> |
### Stable-Video-Diffusion
| Watermarked | Tampered | GT Mask | Pred Mask |
|-------------|----------|---------|-----------|
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/00/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/00/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/00/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/00/mask_pred.gif" width="150"> |
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/01/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/01/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/01/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/01/mask_pred.gif" width="150"> |
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/02/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/02/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/02/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/02/mask_pred.gif" width="150"> |
| <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/03/watermarked.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/03/tampered.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/03/mask_gt.gif" width="150"> | <img src="https://github.com/hurunyi/VideoShield/blob/master/examples/stable-video-diffusion/03/mask_pred.gif" width="150"> |


## Environment Setup
```
pip install -r requirements.txt
```

## Model Download

Download the video model to your preferred directory.

- The text-to-video (T2V) model ModelScope can be downloaded from: https://huggingface.co/ali-vilab/text-to-video-ms-1.7b.
- The image-to-video (I2V) model Stable-Video-Diffusion can be download from: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt.

## Running the Scripts

### 1. Watermark Embedding and Extraction

- For ModelScope:

```bash
python3 watermark_embedding_and_extraction.py \
	--device 'cuda:0' \
	--model_name modelscope \
	--model_path <your_model_path> \
	--num_frames 16 \
	--height 256 \
	--width 256 \
	--frames_copy 8 \
	--hw_copy 4 \
	--channel_copy 1 \
	--num_inference_steps 25
```

- For Stable-Video-Diffusion:

```bash
python3 watermark_embedding_and_extraction.py \
	--device 'cuda:0' \
	--model_name stable-video-diffusion \
	--model_path <your_model_path> \
	--num_frames 16 \
	--height 512 \
	--width 512 \
	--frames_copy 8 \
	--hw_copy 8 \
	--channel_copy 1 \
	--num_inference_steps 25
```

Note:

- You can also skip specifying *--model_path* (skip **Model Download**). The script will automatically download the model to the default cache directory. 
- The generated watermarked videos and watermark information will be saved in the *./results* directory.

### 2. Temporal Tamper Localization

- For ModelScope:

```bash
python3 temporal_tamper_localization.py \
	--device 'cuda:0' \
	--model_name modelscope \
	--model_path <your_model_path> \
	--num_inversion_steps 25 \
	--video_frames_dir './results/modelscope/a_red_panda_eating_leaves/wm/frames'
```

- For Stable-Video-Diffusion:

```bash
python3 temporal_tamper_localization.py \
	--device 'cuda:0' \
	--model_name stable-video-diffusion \
	--model_path <your_model_path> \
	--num_inversion_steps 25 \
	--video_frames_dir './results/modelscope/a_red_panda_eating_leaves/wm/frames'
```

Default video frames directory: *'./results/stable-video-diffusion/a\_red\_panda\_eating\_leaves/wm/frames'* (can be modified as needed)

### 3. Spatial Tamper Localization

- For ModelScope:

```bash
python3 spatial_tamper_localization.py \
	--device 'cuda:0' \
	--model_name modelscope \
	--model_path <your_model_path> \
	--num_inversion_steps 25 \
	--video_frames_dir './results/modelscope/a_red_panda_eating_leaves/wm/frames'
```

- For Stable-Video-Diffusion:

```bash
python3 spatial_tamper_localization.py \
	--device 'cuda:0' \
	--model_name stable-video-diffusion \
	--model_path <your_model_path> \
	--num_inversion_steps 25 \
	--video_frames_dir './results/modelscope/a_red_panda_eating_leaves/wm/frames'
```

Default video frames directory: *'./results/stable-video-diffusion/a\_red\_panda\_eating\_leaves/wm/frames'* (can be modified as needed)


## Acknowledgements
This code builds on the code from the [GaussianShading](https://github.com/bsmhmmlf/Gaussian-Shading/tree/master).
