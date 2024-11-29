import os
import random
import argparse
import cv2
import torch
import numpy as np
from utils import transform_video, cv2_to_pil, pil_to_cv2, get_video_latents, latents_to_video, save_video_frames
from watermark import VideoShield
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline, DDIMInverseScheduler
from diffusers.utils import export_to_video
from sklearn.metrics import roc_auc_score


def calculate_pixel_f1(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    cross = np.logical_and(pd, gt)
    union = np.logical_or(pd, gt)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    auc = roc_auc_score(gt, pd)
    return f1, precision, recall, iou, auc


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def video_mask_postprocess(video_frames_mask):
    video_frames_mask_blur = []
    for img in video_frames_mask.copy():
        pil_img = cv2_to_pil(img)
        gray_image = pil_img.convert('L')
        bw_image = gray_image.point(lambda p: p <= 128 and 255)
        pil_img = bw_image.convert('RGB')
        img = pil_to_cv2(pil_img)
        video_frames_mask_blur.append(img)
    return video_frames_mask_blur


# Default Crop&Drop tamper
def spatial_tamper(video_frames):
    video_frames_mask = []
    video_frames_tampered = []
    random_choice = random.choice([0, 1])
    width, height, c = video_frames[0].shape
    if random_choice == 0:
        # Crop
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        for img in video_frames.copy():
            start_x, start_y = start_x + random.randint(2, 2), start_y + random.randint(2, 2)
            end_x, end_y = start_x + new_width, start_y + new_height
            padded_image = np.flip(img, axis=0).copy()
            padded_image[start_x:end_x, start_y:end_y] = img[start_x:end_x, start_y:end_y]
            mask_image = np.ones_like(img)
            mask_image[start_x:end_x, start_y:end_y] = 0
            video_frames_tampered.append(padded_image)
            video_frames_mask.append(mask_image)
    else:
        # Drop
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        for img in video_frames.copy():
            start_x, start_y = start_x + random.randint(2, 2), start_y + random.randint(2, 2)
            padded_image = np.flip(img[start_x:start_x + new_width, start_y:start_y + new_height], axis=0).copy()
            img[start_x:start_x + new_width, start_y:start_y + new_height] = padded_image
            mask_image = np.zeros_like(img)
            mask_image[start_x:start_x + new_width, start_y:start_y + new_height] = 1
            video_frames_tampered.append(img)
            video_frames_mask.append(mask_image)
    return video_frames_tampered, video_frames_mask


def main(args):
    modelscope_loc_threshold = {
        'default':
            {
                'loc4': [0.5703, 0.6640],
                'loc2': [0.5625, 0.6875],
                'loc1': [0.2500, 0.7500],
            },
        'distortion':
            {
                'loc4': [0.4843, 0.5703],
                'loc2': [0.3750, 0.6875],
                'loc1': [0.0000, 1.0000],
            },
    }
    stable_video_diffusion_loc_threshold = {
        'default':
            {
                'loc4': [0.4765, 0.5664],
                'loc2': [0.3750, 0.6875],
                'loc1': [0.0000, 1.0000],
            },
        'distortion':
            {
                'loc8': [0.4868, 0.5229],
                'loc4': [0.4531, 0.5664],
                'loc2': [0.3437, 0.6875],
                'loc1': [0.0000, 1.0000],
            },
    }

    model_name = args.model_name

    if args.model_path is not None:
        model_path = args.model_path
    else:
        if model_name == 'modelscope':
            model_path = 'damo-vilab/text-to-video-ms-1.7b'
        elif model_name == 'stable-video-diffusion':
            model_path = 'stabilityai/stable-video-diffusion-img2vid-xt'
        else:
            raise ValueError

    num_frames = 16
    num_inversion_steps = args.num_inversion_steps
    device = args.device

    if model_name == 'modelscope':
        video_pipe = TextToVideoSDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_path,
            subfolder='scheduler'
        )
        loc_threshold = modelscope_loc_threshold
        hierarchical_level = 3
        height = 256
        width = 256
        frame_factor = 8
        hw_factor = 4
    elif model_name == 'stable-video-diffusion':
        video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_path,
            subfolder='scheduler'
        )
        loc_threshold = stable_video_diffusion_loc_threshold
        hierarchical_level = 3
        height = 512
        width = 512
        frame_factor = 8
        hw_factor = 8
    else:
        raise ValueError

    video_pipe.scheduler = inverse_scheduler
    video_frames_path = args.video_frames_path
    video_frames_files = os.listdir(video_frames_path)
    video_frames_files.sort()
    video_frames_files = [f'{video_frames_path}/{file}' for file in video_frames_files]
    video_frames = [cv2.imread(frame_file) for frame_file in video_frames_files]
    video_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255. for frame in video_frames]
    first_frame = cv2_to_pil(video_frames[0])
    watermark = VideoShield(
        ch_factor=1, hw_factor=hw_factor, frame_factor=frame_factor,
        height=int(height / 8), width=int(width / 8),
        num_frames=num_frames, device=device,
    )
    prompt_dir = os.path.split(os.path.split(video_frames_path)[0])[0]
    wm_info = torch.load(f'{prompt_dir}/wm_info.bin', map_location=device)
    watermark.m = wm_info['m']
    watermark.watermark = wm_info['watermark']
    watermark.key = wm_info['key']
    watermark.nonce = wm_info['nonce']

    video_frames_tampered, video_frames_mask = spatial_tamper(video_frames)
    video_frames_tampered_th = transform_video(video_frames_tampered).to(video_pipe.vae.dtype).to(device)

    if model_name != 'stable-video-diffusion':
        video_latents = get_video_latents(video_pipe.vae, video_frames_tampered_th, sample=False, permute=True)
    else:
        video_latents = get_video_latents(video_pipe.vae, video_frames_tampered_th, sample=False, permute=False)

    if model_name == 'modelscope':
        reversed_latents = video_pipe(
            prompt='',
            latents=video_latents,
            num_inference_steps=num_inversion_steps,
            guidance_scale=1.,
            output_type='latent',
        ).frames
    elif model_name == 'stable-video-diffusion':
        reversed_latents = video_pipe(
            first_frame,
            height=height,
            width=width,
            latents=video_latents,
            num_frames=num_frames,
            output_type='latent',
            num_inference_steps=num_inversion_steps,
            max_guidance_scale=1.,
        ).frames
    else:
        raise ValueError

    if model_name == 'stable-video-diffusion':
        reversed_latents = reversed_latents.permute(0, 2, 1, 3, 4)

    video_mask, loc_info = watermark.tamper_localization(
        reversed_latents,
        loc_threshold['default'],
        hierarchical_level=hierarchical_level,
    )

    if model_name == 'stable-video-diffusion':
        video_mask = video_mask.permute(0, 2, 1, 3, 4)
        video_mask = latents_to_video(video_pipe, video_mask.half(), num_frames)
    else:
        video_mask = latents_to_video(video_pipe, video_mask.half())

    save_dir = os.path.join(prompt_dir, 'wm_spatial_tampered')
    save_mask_pred_dir = os.path.join(save_dir, f'frames_mask_pred')
    save_mask_gt_dir = os.path.join(save_dir, f'frames_mask_gt')
    save_video_tampered_dir = os.path.join(save_dir, f'frames_tampered')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_pred_dir, exist_ok=True)
    os.makedirs(save_mask_gt_dir, exist_ok=True)
    os.makedirs(save_video_tampered_dir, exist_ok=True)

    mask_pred = video_mask_postprocess(video_mask[0])
    export_to_video(
        mask_pred,
        os.path.join(save_dir, f'mask_pred.mp4')
    )
    save_video_frames(mask_pred, save_mask_pred_dir)

    mask_gt = video_frames_mask
    export_to_video(
        mask_gt,
        os.path.join(save_dir, f'mask_gt.mp4')
    )
    save_video_frames(mask_gt, save_mask_gt_dir)

    export_to_video(
        video_frames_tampered,
        os.path.join(save_dir, f'wm_tampered.mp4')
    )
    save_video_frames(video_frames_tampered, save_video_tampered_dir)

    f1, precision, recall, iou, auc = calculate_pixel_f1(
        np.stack(mask_pred, axis=0).flatten(),
        np.stack(mask_gt, axis=0).flatten()
    )
    print(f'Localization performance: f1={f1:.3f}, precision={precision:.3f}, recall={recall:.3f}, iou={iou:.3f}, auc={auc:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--model_name', default='modelscope')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--num_inversion_steps', default=25, type=int)
    parser.add_argument('--video_frames_path', default='./results/modelscope/a_red_panda_eating_leaves/wm/frames', type=str)
    args = parser.parse_args()
    main(args)
