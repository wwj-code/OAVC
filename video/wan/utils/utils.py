# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp

import imageio
import torch
import torchvision  # 保留，不影响；cache_image 仍需要

__all__ = ['cache_video', 'cache_image', 'str2bool']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(
    tensor,
    save_file=None,
    fps=30,
    suffix='.mp4',
    nrow=8,                 # 保留参数兼容，但本实现不再依赖 make_grid
    normalize=True,         # 保留参数兼容
    value_range=(-1, 1),
    retry=5
):
    """
    tensor expected shape:
      - [B, C, T, H, W]  (common)
      - or [C, T, H, W]
    value_range default (-1,1). Will be mapped to uint8 [0,255].
    """
    cache_file = osp.join('/tmp', rand_name(suffix=suffix)) if save_file is None else save_file

    error = None
    for _ in range(retry):
        try:
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)

            # ---- ensure float for safe math ----
            tensor = tensor.to(torch.float32)

            # ---- clamp to expected range ----
            lo, hi = float(value_range[0]), float(value_range[1])
            tensor = tensor.clamp(min=lo, max=hi)

            # ---- handle shapes ----
            # [B,C,T,H,W] -> take first batch -> [C,T,H,W]
            if tensor.dim() == 5:
                tensor = tensor[0]
            # [C,T,H,W] expected now
            if tensor.dim() != 4:
                raise ValueError(f"cache_video expects 4D/5D tensor, got shape={tuple(tensor.shape)}")

            # [C,T,H,W] -> [T,H,W,C]
            tensor = tensor.permute(1, 2, 3, 0).contiguous()

            # ---- map (lo,hi) -> (0,255) ----
            # avoid division by zero
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            tensor = (tensor - lo) / denom * 255.0
            tensor = tensor.round().clamp(0, 255).to(torch.uint8).cpu()

            # ---- write video ----
            writer = imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():  # frame: [H,W,C], uint8
                writer.append_data(frame)
            writer.close()
            return cache_file

        except Exception as e:
            error = e
            continue

    print(f'cache_video failed, error: {error}', flush=True)
    return None


def cache_image(
    tensor,
    save_file,
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
    retry=5
):
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp']:
        suffix = '.png'

    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range
            )
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')