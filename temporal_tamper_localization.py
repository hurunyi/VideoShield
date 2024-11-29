import os
import argparse
import copy
import cv2
import torch
import numpy as np
import random
from utils import transform_video, cv2_to_pil, get_video_latents
from watermark import VideoShield
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline, DDIMInverseScheduler


def temporal_tamper(video_frames, tampering_type_list):
    tampered_videos = {}
    video_frames_ids = list(range(len(video_frames)))
    for tampering_type in tampering_type_list:
        if tampering_type == 'identity':
            tampered_videos['identity'] = [copy.deepcopy(video_frames), copy.deepcopy(video_frames_ids)]
        if tampering_type == 'frame swap':
            frame_swap_steps = 4
            video_frames_tampered = copy.deepcopy(video_frames)
            video_frames_tampered_ids = copy.deepcopy(video_frames_ids)
            for i in range(1, len(video_frames) - 1, frame_swap_steps):
                video_frames_tampered[i] = video_frames[i + 1]
                video_frames_tampered[i + 1] = video_frames[i]
                video_frames_tampered_ids[i] = video_frames_ids[i + 1]
                video_frames_tampered_ids[i + 1] = video_frames_ids[i]
            tampered_videos['frame swap'] = [video_frames_tampered, video_frames_tampered_ids]
        if tampering_type == 'frame insert':
            video_frames_tampered = copy.deepcopy(video_frames)
            video_frames_tampered_ids = copy.deepcopy(video_frames_ids)
            insert_id = random.randint(1, len(video_frames))
            insert_id_frame_list = [
                [insert_id-1, video_frames[insert_id-1]],
                [-1, np.random.rand(*(video_frames[0].shape))]
            ]
            insert_id_frame = random.choice(insert_id_frame_list)
            video_frames_tampered_ids.insert(insert_id, insert_id_frame[0])
            video_frames_tampered.insert(insert_id, insert_id_frame[1])
            tampered_videos['frame insert'] = [video_frames_tampered, video_frames_tampered_ids]
        if tampering_type == 'frame drop':
            video_frames_tampered = copy.deepcopy(video_frames)
            video_frames_tampered_ids = copy.deepcopy(video_frames_ids)
            drop_id = random.randint(1, len(video_frames)-1)
            del video_frames_tampered[drop_id]
            del video_frames_tampered_ids[drop_id]
            tampered_videos['frame drop'] = [video_frames_tampered, video_frames_tampered_ids]
    return tampered_videos


def main(args):
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

    temporal_threshold = 0.55

    if model_name == 'modelscope':
        video_pipe = TextToVideoSDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_path,
            subfolder='scheduler'
        )
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
        height = 512
        width = 512
        frame_factor = 8
        hw_factor = 8
    else:
        raise ValueError

    video_pipe.scheduler = inverse_scheduler
    video_frames_dir = args.video_frames_dir

    temporal_tampering_type = ['frame swap', 'frame insert', 'frame drop']
    video_frames_files = sorted(os.listdir(video_frames_dir))
    video_frames_files = [f'{video_frames_dir}/{file}' for file in video_frames_files]
    video_frames = [cv2.imread(frame_file) for frame_file in video_frames_files]
    video_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255. for frame in video_frames]

    # temporal tamper
    video_frames_tampered = temporal_tamper(video_frames, temporal_tampering_type)
    for tamper_type, (video_frames_t, frame_ids) in video_frames_tampered.items():
        print(f'Temporal tamper type: {tamper_type}')
        tampered_num_frames = len(video_frames_t)
        first_frame = cv2_to_pil(video_frames_t[0])
        watermark = VideoShield(
            ch_factor=1, hw_factor=hw_factor, frame_factor=frame_factor,
            height=int(height / 8), width=int(width / 8),
            num_frames=num_frames, device=device,
        )
        wm_info_dir = os.path.split(os.path.split(video_frames_dir)[0])[0]
        wm_info = torch.load(f'{wm_info_dir}/wm_info.bin', map_location=device)
        watermark.m = wm_info['m']
        watermark.watermark = wm_info['watermark']
        watermark.key = wm_info['key']
        watermark.nonce = wm_info['nonce']

        video_frames_t = transform_video(video_frames_t).to(video_pipe.vae.dtype).to(device)
        if model_name != 'stable-video-diffusion':
            video_latents = get_video_latents(video_pipe.vae, video_frames_t, sample=False, permute=True)
        else:
            video_latents = get_video_latents(video_pipe.vae, video_frames_t, sample=False, permute=False)

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
                num_frames=tampered_num_frames,
                latents=video_latents,
                output_type='latent',
                num_inference_steps=num_inversion_steps,
                max_guidance_scale=1.,
            ).frames
        else:
            raise ValueError

        if model_name == 'stable-video-diffusion':
            reversed_latents = reversed_latents.permute(0, 2, 1, 3, 4)

        reversed_m = (reversed_latents > 0).int()
        template_m = watermark.m
        b, c, f, h, w = reversed_m.size()
        t_f = template_m.shape[2]
        reversed_m_repeat = reversed_m.permute(2, 0, 1, 3, 4).reshape(f, b * c * h * w).unsqueeze(1).repeat(1, t_f,
                                                                                                            1)
        template_m_repeat = template_m.permute(2, 0, 1, 3, 4).reshape(t_f, b * c * h * w).unsqueeze(0).repeat(f, 1,
                                                                                                              1)
        cmp_bits = (reversed_m_repeat == template_m_repeat).float().mean(dim=2)
        max_values, pred_frame_ids = cmp_bits.float().max(dim=1)
        pred_mask = (max_values < temporal_threshold)
        pred_frame_ids = pred_frame_ids.masked_fill(pred_mask, -1)[1:]
        gt_frame_ids = torch.Tensor(frame_ids)[1:]
        acc_metric = (gt_frame_ids.cpu() == pred_frame_ids.cpu()).float().mean().item()
        print(f'Localization accuracy: {acc_metric:.3f}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--model_name', default='modelscope')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--num_inversion_steps', default=25, type=int)
    parser.add_argument('--video_frames_dir', default=None, type=str)
    args = parser.parse_args()
    main(args)
