# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .dnaedit_wan_video_utils import wan_dnaedit_video_weighted
import torch.nn.functional as F
from typing import Optional


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

        return videos[0] if self.rank == 0 else None
    def edit_DNAEdit(self,video,
                 src_prompt,
                 tgt_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 tgt_guide_scale=5.0,
                 jmp=12,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            src_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        video = video.to('cuda')
        latents = self.vae.encode(video) #[b,c,t,h,w]
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_src = self.text_encoder([src_prompt], self.device)
            context_tgt = self.text_encoder([tgt_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        # else:
        #     context = self.text_encoder([src_prompt], torch.device('cpu'))
        #     context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        #     context = [t.to(self.device) for t in context]
        #     context_null = [t.to(self.device) for t in context_null]

        

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos

            arg_src = {'context': context_src, 'seq_len': seq_len}
            arg_tgt = {'context': context_tgt, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            start_latents=latents #[]
            rand_noise = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device)
            comp_lst=[]
            v_lst=[]
            last_lst=[]
            prev_latent = latents[0]
            timesteps = timesteps.flip(dims=[0])
            for i, t_curr in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                t_prev = 0 if i==0 else timesteps[i-1]
                timestep = [t_curr]
                timestep = torch.stack(timestep)
                curr_latent = rand_noise*(t_curr-t_prev)/(1000.0-t_prev)+ prev_latent*(1000.0-t_curr)/(1000-t_prev)
                # src_latent = [t_prev/1000.0*noise[0] + (1000-t_prev)/1000.0*start_latents[0]]
                self.model.to(self.device)
                noise_pred_cond_curr = self.model(
                    [curr_latent], t=timestep, **arg_src)[0]
                noise_pred_uncond_curr = self.model(
                    [curr_latent], t=timestep, **arg_null)[0]
                noise_pred_curr = noise_pred_uncond_curr + guide_scale * (
                    noise_pred_cond_curr - noise_pred_uncond_curr)
                delta_v = noise_pred_curr-((curr_latent-prev_latent)/((t_curr-t_prev)/1000.0))
                
                curr_latent = curr_latent+ delta_v*(t_curr-t_prev)/1000.0
                prev_latent=curr_latent
                rand_noise = rand_noise + delta_v * (1000.0-t_prev)/1000.0
                # import ipdb
                # ipdb.set_trace()
                comp_lst.append((delta_v*(t_curr-t_prev)/1000.0).clone())
                v_lst.append(noise_pred_curr.clone())
                last_lst.append(prev_latent.clone())
            
            comp_lst=comp_lst[::-1]
            v_lst=v_lst[::-1]
            last_lst=last_lst[::-1]
            edit_latent = last_lst[jmp]
            timesteps = timesteps.flip(dims=[0])
            # import ipdb
            # ipdb.set_trace()
            for i,t_prev in enumerate(tqdm(timesteps)):
                t_next = 0 if i==len(timesteps)-1 else timesteps[i+1]
                timestep = [t_prev]
                timestep = torch.stack(timestep)
                if i<jmp:
                    continue
                noise_pred_cond_tgt = self.model(
                    [edit_latent+comp_lst[i]], t=timestep, **arg_tgt)[0]
                noise_pred_uncond_tgt = self.model(
                    [edit_latent+comp_lst[i]], t=timestep, **arg_null)[0]
                noise_pred_tgt = noise_pred_uncond_tgt + tgt_guide_scale * (
                    noise_pred_cond_tgt - noise_pred_uncond_tgt)
                
                edit_latent = edit_latent+ (t_next-t_prev)*noise_pred_tgt/1000.0
                mv_latent = edit_latent
               
            x0 = [mv_latent]
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        # del noise, latents
        # del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
    
    def edit_DNAEdit_weighted(
        self,
        video,
        src_prompt,
        tgt_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        tgt_guide_scale=5.0,
        jmp=12,
        n_prompt="",
        seed=-1,
        offload_model=True,

        # --------- NEW: mask tensor from edit.py ----------
        # expected raw mask at pixel scale: [B,1,T,H,W] in [0,1], same T and H,W as input video
        edit_mask_video: Optional[torch.Tensor] = None,

        # --------- mask knobs ----------
        mask_thresh: float = 0.3,
        fg_dilate: int = 2,
        bd_dilate: int = 4,
        beta_max: float = 0.45,
        beta_pow: float = 2.0,

        # --------- Stage1 knobs ----------
        stage1_lambda_fg: float = 0.7,
        stage1_bg_dilate: int = 1,
        stage1_proj_bg: float = 0.1,

        # --------- Stage2 knobs ----------
        use_projected_injection: bool = True,
        use_conf_gate: bool = True,
        gate_decay: float = 0.95,
        gate_k: float = 2.0,
        gate_temp: float = 0.5,
        ring_highpass: bool = True,
        hp_kernel: int = 3,

        # --------- other ----------
        hard_bg_blend: bool = False,
    ):
        # preprocess
        Fnum = frame_num
        video = video.to(self.device)

        # encode to latent [B,C,T,H,W]
        latents_5d = self.vae.encode(video)  # [b,c,t,h,w]
        latents_src_4d = latents_5d[0].contiguous()  # [C,T,H,W]

        target_shape = (
            self.vae.model.z_dim,
            (Fnum - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # prompts
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_src = self.text_encoder([src_prompt], self.device)
            context_tgt = self.text_encoder([tgt_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            # keep parity with your code style (if you later need cpu path)
            context_src = self.text_encoder([src_prompt], torch.device('cpu'))
            context_tgt = self.text_encoder([tgt_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context_src = [t.to(self.device) for t in context_src]
            context_tgt = [t.to(self.device) for t in context_tgt]
            context_null = [t.to(self.device) for t in context_null]

        # scheduler
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # build rand_noise in latent shape [C,T,H,W]
        rand_noise_4d = torch.randn(
            target_shape[0], target_shape[1], target_shape[2], target_shape[3],
            dtype=torch.float32, device=self.device, generator=seed_g
        )

        # ---------------- mask downsample to latent scale ----------------
        edit_mask_latent_4d = None
        if edit_mask_video is not None:
            # edit_mask_video: [B,1,T,H,W] at pixel scale, in [0,1]
            m = edit_mask_video.to(device=self.device, dtype=torch.float32).clamp(0, 1)
            if m.shape[0] != 1:
                m = m[:1]

            _, _, T, Hpx, Wpx = m.shape

            # latent shape: latents_src_4d is [B, C, T_lat, H_lat, W_lat] or [C, T_lat, H_lat, W_lat]?
            # make it robust:
            if latents_src_4d.dim() == 5:
                T_lat = latents_src_4d.shape[-3]
                H_lat, W_lat = latents_src_4d.shape[-2], latents_src_4d.shape[-1]
            elif latents_src_4d.dim() == 4:
                T_lat = latents_src_4d.shape[-3]
                H_lat, W_lat = latents_src_4d.shape[-2], latents_src_4d.shape[-1]
            else:
                raise ValueError(f"Unexpected latents_src_4d dim: {latents_src_4d.dim()}")

            # 1) spatial resize per-frame to latent H,W
            m_bt = m.permute(0, 2, 1, 3, 4).reshape(1 * T, 1, Hpx, Wpx)   # [T,1,H,W]
            m_bt = F.interpolate(m_bt, size=(H_lat, W_lat), mode="bilinear", align_corners=False)
            m_lat_fullT = m_bt.reshape(1, T, 1, H_lat, W_lat).permute(0, 2, 1, 3, 4)  # [1,1,T,H_lat,W_lat]

            # 2) temporal downsample T -> T_lat
            stride_t = int(getattr(self, "vae_stride", [4, 8, 8])[0])  # Wan 默认通常 stride_t=4
            # 保险：如果 stride_t 不对，就用 T 和 T_lat 推一下
            if T_lat != (T - 1) // stride_t + 1:
                stride_t = max(1, int(round((T - 1) / max(T_lat - 1, 1))))
            # 现在 stride_t 应该能让 41 -> 11

            # ---- 方式A：窗口 max 聚合（推荐，稳，不容易漏运动物体）----
            # 让每个 latent 帧对应一段原视频帧窗口 [k*stride_t, (k+1)*stride_t)
            chunks = []
            for k in range(T_lat):
                s = k * stride_t
                e = min((k + 1) * stride_t, T)
                # 如果最后不足一个窗口，至少取最后一帧
                if s >= T:
                    s = T - 1
                    e = T
                chunk = m_lat_fullT[:, :, s:e]         # [1,1,win,H,W]
                chunk = chunk.max(dim=2).values        # [1,1,H,W]
                chunks.append(chunk)
            m_lat = torch.stack(chunks, dim=2)         # [1,1,T_lat,H,W]

            # ---- 方式B：直接取样（更像“取关键帧”，但可能漏掉运动物体）----
            # idx = [min(k*stride_t, T-1) for k in range(T_lat)]
            # m_lat = m_lat_fullT[:, :, idx]            # [1,1,T_lat,H,W]

            edit_mask_latent_4d = m_lat.squeeze(1)     # [1,T_lat,H_lat,W_lat]


        # run weighted edit in latent space
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            self.model.to(self.device)

            edited_latent_4d = wan_dnaedit_video_weighted(
                model=self.model,
                sample_scheduler=sample_scheduler,
                timesteps=timesteps,
                latents_src_4d=latents_src_4d,
                rand_noise_4d=rand_noise_4d,
                context_src=context_src,
                context_tgt=context_tgt,
                context_null=context_null,
                seq_len=seq_len,
                guide_scale=guide_scale,
                tgt_guide_scale=tgt_guide_scale,
                jmp=jmp,

                edit_mask_latent_4d=edit_mask_latent_4d,

                mask_thresh=mask_thresh,
                fg_dilate=fg_dilate,
                bd_dilate=bd_dilate,
                beta_max=beta_max,
                beta_pow=beta_pow,

                stage1_lambda_fg=stage1_lambda_fg,
                stage1_bg_dilate=stage1_bg_dilate,
                stage1_proj_bg=stage1_proj_bg,

                use_projected_injection=use_projected_injection,
                use_conf_gate=use_conf_gate,
                gate_decay=gate_decay,
                gate_k=gate_k,
                gate_temp=gate_temp,
                ring_highpass=ring_highpass,
                hp_kernel=hp_kernel,

                hard_bg_blend=hard_bg_blend,
            )

            x0 = [edited_latent_4d]
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
