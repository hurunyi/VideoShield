import os
import json
import argparse
from tqdm import tqdm
import torch
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers import DDIMInverseScheduler
from utils import (
    transform_video,
    save_video_frames,
    cv2_to_pil,
    get_video_latents,
)
from watermark import VideoShield


def main(args):
    device = args.device
    model_name = args.model_name
    data_dir = args.data_dir
    output_dir = os.path.join(args.output_dir, model_name)

    os.makedirs(output_dir, exist_ok=True)

    assert model_name is not None, 'you must provide the model name!'

    if args.model_path is not None:
        model_path = args.model_path
    else:
        if model_name == 'modelscope':
            model_path = 'damo-vilab/text-to-video-ms-1.7b'
        elif model_name == 'stable-video-diffusion':
            model_path = 'stabilityai/stable-video-diffusion-img2vid-xt'
        else:
            raise ValueError

    if model_name == 'modelscope':
        video_pipe = TextToVideoSDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
    elif model_name == 'stable-video-diffusion':
        video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
    else:
        raise ValueError

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        model_path,
        subfolder='scheduler',
    )
    video_pipe.safety_checker = None
    scheduler = video_pipe.scheduler

    # class for watermark
    watermark = VideoShield(
        ch_factor=args.channel_copy, hw_factor=args.hw_copy, frame_factor=args.frames_copy,
        height=int(args.height / 8), width=int(args.width / 8), num_frames=args.num_frames, device=device,
    )

    with open(f'./{data_dir}/test_prompts.txt', 'r') as f:
        data = [line.strip() for line in f.readlines()]

    args_dict = vars(args)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    for i, row in tqdm(enumerate(data)):
        current_prompt = row
        video_id = current_prompt.replace(' ', '_')
        video_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)

        print(f'Generating for prompt: {current_prompt}.')

        # generate with watermark
        init_latents_w = watermark.create_watermark_and_return_w()
        if model_name == 'stable-video-diffusion':
            init_latents_w = init_latents_w.permute(0, 2, 1, 3, 4)
        torch.save(
            {'m': watermark.m, 'watermark': watermark.watermark, 'key': watermark.key, 'nonce': watermark.nonce},
            f'{video_dir}/wm_info.bin'
        )

        print(f'Video generation and watermark embedding:')
        video_pipe.scheduler = scheduler
        if model_name == 'modelscope':
            video_frames_w = video_pipe(
                prompt=current_prompt,
                latents=init_latents_w,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=9.0,
            ).frames[0]
        elif model_name == 'stable-video-diffusion':
            image_path = f'./{data_dir}/images_for_i2v/{video_id}.png'
            image = load_image(image_path)
            video_frames_w = video_pipe(
                image=image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                decode_chunk_size=16,
                latents=init_latents_w,
                output_type='np',
                num_inference_steps=args.num_inference_steps,
                max_guidance_scale=3.0,
            ).frames[0]
        else:
            raise ValueError

        save_mp4_path = os.path.join(video_dir, f'wm.mp4')
        export_to_video(video_frames_w, output_video_path=save_mp4_path)
        print(f'The generated video is saved to {save_mp4_path}.')
        frames_dir = f'{video_dir}/wm/frames'
        os.makedirs(frames_dir, exist_ok=True)
        save_video_frames(video_frames_w, frames_dir)
        print(f'The generated video frames are saved to {frames_dir}.')

        first_frame = cv2_to_pil(video_frames_w[0])
        video_frames_w = transform_video(video_frames_w).to(video_pipe.vae.dtype).to(device)
        if model_name == 'stable-video-diffusion':
            video_latents_w = get_video_latents(video_pipe.vae, video_frames_w, sample=False, permute=False)
        else:
            video_latents_w = get_video_latents(video_pipe.vae, video_frames_w, sample=False, permute=True)

        print(f'Watermark extraction:')
        video_pipe.scheduler = inverse_scheduler
        if model_name == 'modelscope':
            # assume at the detection time, the original prompt is unknown
            reversed_latents_w = video_pipe(
                prompt='',
                latents=video_latents_w,
                num_inference_steps=args.num_inversion_steps,
                guidance_scale=1.,
                output_type='latent',
            ).frames
        elif model_name == 'stable-video-diffusion':
            reversed_latents_w = video_pipe(
                image=first_frame,
                height=args.height,
                width=args.width,
                latents=video_latents_w,
                num_frames=args.num_frames,
                output_type='latent',
                num_inference_steps=args.num_inversion_steps,
                max_guidance_scale=1.,
            ).frames
        else:
            raise ValueError

        # acc metric
        if model_name == 'stable-video-diffusion':
            reversed_latents_w = reversed_latents_w.permute(0, 2, 1, 3, 4)
        acc_metric = watermark.eval_watermark(reversed_latents_w)
        print(f'Watermark extraction accuracy: {acc_metric}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VideoShield')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_frames', default=0, type=int)
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=25, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--frames_copy', default=8, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--model_name', default='modelscope')
    parser.add_argument('--model_path', default=None)

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
