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

Download the video model to your preferred directory. For example, the ModelScope model can be downloaded from: https://huggingface.co/ali-vilab/text-to-video-ms-1.7b.

## Running the Scripts

### 1. Watermark Embedding and Extraction

```
python3 watermark_embedding_and_extraction.py --model_name modelscope --model_path <your_model_path>
```

Note: You can also skip specifying *--model_path* (skip **Model Download**). The script will automatically download the model to the default cache directory. Results will be saved in the ./results directory.

### 2. Temporal Tamper Localization

```
python3 temporal_tamper_localization.py --model_name modelscope --model_path <your_model_path>
```

Default video frames directory: *'./results/stable-video-diffusion/a\_red\_panda\_eating\_leaves/wm/frames'* (can be modified as needed)

### 3. Spatial Tamper Localization

```
python3 spatial_tamper_localization.py --model_name modelscope --model_path <your_model_path>
```

Default video frames directory: *'./results/stable-video-diffusion/a\_red\_panda\_eating\_leaves/wm/frames'* (can be modified as needed)


## Acknowledgements
This code builds on the code from the [GaussianShading](https://github.com/bsmhmmlf/Gaussian-Shading/tree/master).
