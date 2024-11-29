# This code is adapted from https://github.com/bsmhmmlf/Gaussian-Shading/tree/master.

import torch
from scipy.stats import norm, truncnorm
from functools import reduce
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes


class VideoShield:
    def __init__(
        self, ch_factor, hw_factor, frame_factor=0,
        height=64, width=64, num_frames=0, local_copy=1,
        same_frames=False, device='cuda:0'
    ):
        self.ch = ch_factor
        self.hw = hw_factor
        self.fr = frame_factor
        self.local_copy = local_copy
        self.same_frames = same_frames
        self.nonce = None
        self.key = None
        self.watermark = None
        self.m = None
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.device = device

        if self.num_frames and not self.same_frames:
            self.latentlength = 1 * 4 * self.num_frames * self.height * self.width
            self.marklength = self.latentlength // (self.ch * self.fr * self.hw * self.hw)
            if self.fr == 1 and self.hw == 1 and self.ch == 1:
                self.threshold = 1
            else:
                self.threshold = self.ch * self.fr * self.hw * self.hw // 2
        else:
            self.latentlength = 1 * 4 * self.height * self.width
            self.marklength = self.latentlength // (self.ch * self.hw * self.hw)
            self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2

    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        if self.num_frames and not self.same_frames:
            z = torch.from_numpy(z).reshape(1, 4, self.num_frames, self.height, self.width).half()
        else:
            z = torch.from_numpy(z).reshape(1, 4, self.height, self.width).half()
        return z.to(self.device)

    def create_watermark_and_return_w(self):
        if self.num_frames and not self.same_frames:
            self.watermark = torch.randint(
                0, 2,
                [1, 4 // self.ch, self.num_frames // self.fr, self.height // self.hw, self.width // self.hw]
            ).to(self.device)

            sd = self.watermark.repeat_interleave(self.local_copy, dim=2)
            sd = sd.repeat_interleave(self.local_copy, dim=3)
            sd = sd.repeat_interleave(self.local_copy, dim=4)
            sd = sd.repeat(
                1,
                self.ch,
                self.fr // self.local_copy,
                self.hw // self.local_copy,
                self.hw // self.local_copy
            )
            m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
            w = self.truncSampling(m)
            self.m = torch.from_numpy(m).reshape(1, 4, self.num_frames, self.height, self.width)
        else:
            self.watermark = torch.randint(
                0, 2,
                [1, 4 // self.ch, self.height // self.hw, self.width // self.hw]
            ).to(self.device)
            sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)
            if self.same_frames:
                sd = sd.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
            m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
            w = self.truncSampling(m)
            self.m = torch.from_numpy(m).reshape(1, 4, self.height, self.width)
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        if self.num_frames:
            sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, self.num_frames, self.height, self.width).to(torch.uint8)
        else:
            sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, self.height, self.width).to(torch.uint8)
        return sd_tensor.to(self.device)

    def diffusion_inverse(self, watermark_r):
        if self.num_frames and not self.same_frames:
            ch_stride = 4 // self.ch
            frame_stride = self.num_frames // (self.fr // self.local_copy)
            h_stride = self.height // (self.hw // self.local_copy)
            w_stride = self.width // (self.hw // self.local_copy)

            ch_list = [ch_stride] * self.ch
            frame_list = [frame_stride] * (self.fr // self.local_copy)
            h_list = [h_stride] * (self.hw // self.local_copy)
            w_list = [w_stride] * (self.hw // self.local_copy)
            # watermark_r: [1, ch, frames, h, w]
            # split_dim1: [ch_factor, ch//ch_factor, frames, h, w]
            # split_dim2: [ch*frames_factor, ch//factor, frames//factor, h, w]
            # split_dim3: [ch*frames*h_factor, ch//factor, frames//factor, h//factor, w]
            # split_dim4: [ch*frames*h*w_factor, ch//factor, frames//factor, h//factor, w//factor]
            split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
            split_dim1 = torch.cat(torch.split(split_dim1, tuple(frame_list), dim=2), dim=0)
            split_dim1_list = split_dim1.reshape(
                self.ch * (self.fr // self.local_copy),
                4 // self.ch,
                self.num_frames // self.fr,
                self.local_copy,
                self.height,
                self.width
            ).chunk(self.local_copy, dim=3)
            split_dim1_list = [t.squeeze(3) for t in split_dim1_list]

            split_dim2 = torch.cat(split_dim1_list, dim=0)
            split_dim2 = torch.cat(torch.split(split_dim2, tuple(h_list), dim=3), dim=0)
            split_dim2_list = split_dim2.reshape(
                self.ch * self.fr * (self.hw // self.local_copy),
                4 // self.ch,
                self.num_frames // self.fr,
                self.height // self.hw,
                self.local_copy,
                self.width
            ).chunk(self.local_copy, dim=4)
            split_dim2_list = [t.squeeze(4) for t in split_dim2_list]

            split_dim3 = torch.cat(split_dim2_list, dim=0)
            split_dim3 = torch.cat(torch.split(split_dim3, tuple(w_list), dim=4), dim=0)
            split_dim3_list = split_dim3.reshape(
                self.ch * self.fr * self.hw * (self.hw // self.local_copy),
                4 // self.ch,
                self.num_frames // self.fr,
                self.height // self.hw,
                self.width // self.hw,
                self.local_copy
            ).chunk(self.local_copy, dim=5)
            split_dim3_list = [t.squeeze(5) for t in split_dim3_list]
            split_dim4 = torch.cat(split_dim3_list, dim=0)

            vote = torch.sum(split_dim4, dim=0).clone()
            vote[vote <= self.threshold] = 0
            vote[vote > self.threshold] = 1
        else:
            ch_stride = 4 // self.ch
            hw_stride = self.height // self.hw
            ch_list = [ch_stride] * self.ch
            hw_list = [hw_stride] * self.hw
            split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
            split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
            split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
            vote = torch.sum(split_dim3, dim=0).clone()
            vote[vote <= self.threshold] = 0
            vote[vote > self.threshold] = 1

        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        return correct

    def tamper_localization(self, reversed_w, loc_threshold, hierarchical_level=3,):
        f_ch = 4

        def video_tensor_split(video_tensor, local_copy_f, local_copy_hw):
            # (b*ch, c//ch, f, h, w)
            split_dim1 = torch.cat(
                torch.split(video_tensor, 4//f_ch * f_ch, dim=1),
                dim=0
            )
            split_dim1_list = split_dim1.reshape(
                f_ch,
                4 // f_ch,
                self.num_frames // local_copy_f,
                local_copy_f,
                self.height,
                self.width
            ).chunk(local_copy_f, dim=3)
            split_dim1_list = [t.squeeze(3) for t in split_dim1_list]

            # (b*ch*loc, c//ch, f//loc, h, w)
            split_dim2 = torch.cat(split_dim1_list, dim=0)
            split_dim2_list = split_dim2.reshape(
                f_ch * local_copy_f,
                4 // f_ch,
                self.num_frames // local_copy_f,
                self.height // local_copy_hw,
                local_copy_hw,
                self.width
            ).chunk(local_copy_hw, dim=4)
            split_dim2_list = [t.squeeze(4) for t in split_dim2_list]

            # (b*ch*loc*loc, c//ch, f//loc, h//loc, w)
            split_dim3 = torch.cat(split_dim2_list, dim=0)
            split_dim3_list = split_dim3.reshape(
                f_ch * local_copy_f * local_copy_hw,
                4 // f_ch,
                self.num_frames // local_copy_f,
                self.height // local_copy_hw,
                self.width // local_copy_hw,
                local_copy_hw
            ).chunk(local_copy_hw, dim=5)
            split_dim3_list = [t.squeeze(5) for t in split_dim3_list]

            # (b*ch*loc*loc*loc, c//ch, f//loc, h//loc, w//loc)
            split_dim4 = torch.cat(split_dim3_list, dim=0)
            return split_dim4

        reversed_m = (reversed_w > 0).int()
        reversed_watermark = reversed_m

        # (b*ch*loc*loc*loc, c//ch, f//loc, h//loc, w//loc)
        reversed_watermark = (reversed_watermark == self.m.to(self.device))

        video_mask_loc = {}
        video_mask_hierarchical = []
        hierarchical_local_copy = 2 ** (hierarchical_level-1)

        for i in range(hierarchical_level):
            local_copy_now = hierarchical_local_copy//(2**i)
            local_copy_f = local_copy_now
            local_copy_hw = local_copy_now
            split_reversed_watermark = video_tensor_split(reversed_watermark.clone(), local_copy_f, local_copy_hw)
            video_mask = split_reversed_watermark.float().mean(dim=0)
            loc_key = f'loc{local_copy_now}'
            video_mask_loc[loc_key] = video_mask.clone()

            low = loc_threshold[f'loc{local_copy_now}'][0]
            high = loc_threshold[f'loc{local_copy_now}'][1]
            video_mask[video_mask < low] = 0
            video_mask[video_mask > high] = 1

            video_mask = video_mask.unsqueeze(0)
            video_mask = video_mask.repeat_interleave(local_copy_f, dim=2)
            video_mask = video_mask.repeat_interleave(local_copy_hw, dim=3)
            video_mask = video_mask.repeat_interleave(local_copy_hw, dim=4)
            video_mask_hierarchical.append(video_mask)

        D = torch.stack(video_mask_hierarchical, dim=0).mean(dim=0)
        D = 2 * D.float() - 1

        video_mask_final = D.repeat(1, f_ch, 1, 1, 1)

        return video_mask_final, video_mask_loc
