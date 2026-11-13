# oavc_sampler.py: OAVC samplers for SD3.5 and FLUX (built on DNAEdit)
from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


# -------------------- basic helpers --------------------

def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward process in flow-matching
    """
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample
    return sample


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def calc_v_sd3(
    pipe,
    src_tar_latent_model_input,
    src_tar_prompt_embeds,
    src_tar_pooled_prompt_embeds,
    src_guidance_scale,
    tar_guidance_scale,
    t,
):
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred


# -------------------- mask / weight map helpers --------------------

def _dilate_mask(mask: torch.Tensor, iters: int) -> torch.Tensor:
    """
    mask: [B,1,H,W] in [0,1], dtype follows input
    """
    if iters <= 0:
        return mask
    out = mask
    for _ in range(iters):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
    return out


def build_weight_map(
    edit_mask: torch.Tensor,
    fg_dilate: int = 1,
    bd_dilate: int = 4,
    beta_t: float = 0.45,
) -> torch.Tensor:
    """
    Construct W_t:
      - foreground: 1
      - ring: beta_t
      - background: 0
    return dtype matches edit_mask
    """
    if edit_mask.dim() == 3:
        edit_mask = edit_mask.unsqueeze(0)

    m = (edit_mask > 0.5).to(dtype=edit_mask.dtype)
    fg = _dilate_mask(m, fg_dilate)
    bd = _dilate_mask(fg, bd_dilate)
    ring = (bd - fg).clamp(0.0, 1.0)

    beta = edit_mask.new_tensor(beta_t)
    W = (fg + beta * ring).clamp(0.0, 1.0)
    return W


def _dilate(mask01: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """3x3 maxpool dilation, mask01: [B,1,H,W] float in [0,1]"""
    out = mask01
    for _ in range(max(0, iters)):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
    return out


def build_fg_bd_from_mask(
    edit_mask: torch.Tensor,
    thresh: float = 0.3,
    fg_dilate: int = 0,
    bd_dilate: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    input edit_mask (soft/hard), output:
      M_fg: foreground
      M_bd: ring band
    shapes: [B,1,H,W]
    """
    M_fg = (edit_mask > thresh).float()
    if fg_dilate > 0:
        M_fg = _dilate(M_fg, fg_dilate).clamp(0, 1)

    M_d = _dilate(M_fg, bd_dilate).clamp(0, 1)
    M_bd = (M_d - M_fg).clamp(0, 1)
    return M_fg, M_bd


def beta_schedule(t_curr_f: float, t_start_f: float, beta_max: float, beta_pow: float) -> float:
    """
    ratio = t_curr_f / t_start_f (clamp 0..1)
    beta = beta_max * ratio^beta_pow
    """
    if t_start_f <= 1e-8:
        return 0.0
    r = float(t_curr_f / t_start_f)
    r = max(0.0, min(1.0, r))
    return float(beta_max * (r ** beta_pow))


# -------------------- NEW: bg-weighted projection / gating / high-pass --------------------

def _sum_over_nonbatch(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(1, x.dim()))
    return x.sum(dim=dims, keepdim=True)


def proj_parallel_bg(delta: torch.Tensor, ref: torch.Tensor, W_bg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Background-weighted projection of delta onto ref.

    - delta, ref: [B,...] same shape (e.g. SD3 [B,C,H,W], FLUX packed [B,L,D])
    - W_bg: same shape as delta/ref OR broadcastable (e.g. [B,1,H,W] to [B,C,H,W])

    Define <a,b>_bg = sum((W_bg*a)*(W_bg*b)).
    Return: Proj^{bg}_{ref}(delta) = alpha * (W_bg ⊙ ref), where
            alpha = <W_bg⊙delta, W_bg⊙ref> / (||W_bg⊙ref||^2 + eps)
    """
    W = W_bg
    # ensure broadcast OK
    r_w = W * ref
    d_w = W * delta

    num = _sum_over_nonbatch(d_w * r_w)          # [B,1,1,1] or [B,1,1]
    den = _sum_over_nonbatch(r_w * r_w).clamp_min(eps)

    alpha = num / den
    return alpha * r_w


def proj_perp_bg(delta: torch.Tensor, ref: torch.Tensor, W_bg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Perpendicular component under bg-weighted projection:
      r = W_bg ⊙ delta
      r_perp = r - Proj^{bg}_{ref}(r)
    Note: output stays supported on background due to W_bg.
    """
    r = W_bg * delta
    return r - proj_parallel_bg(r, ref, W_bg, eps=eps)


def proj_parallel(delta: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Global (non-weighted) projection, per-sample.
    """
    num = _sum_over_nonbatch(delta * ref)
    den = _sum_over_nonbatch(ref * ref).clamp_min(eps)
    return (num / den) * ref


def proj_perp(delta: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return delta - proj_parallel(delta, ref, eps=eps)


def high_pass_spatial(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    simple HP for spatial tensor: x - avgpool(x)
    x: [B,C,H,W]
    """
    if x.dim() != 4:
        return x
    pad = k // 2
    blur = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
    return x - blur


def high_pass_token(x: torch.Tensor) -> torch.Tensor:
    """
    token high-pass for packed sequence: subtract token-wise mean (low-freq analogue)
    x: [B,L,D]
    """
    if x.dim() != 3:
        return x
    return x - x.mean(dim=1, keepdim=True)


class ScalarGateEMA:
    """
    light-weight confidence gate:
      - track EMA of rms magnitude
      - if current rms >> EMA, gate decreases
    """
    def __init__(self, decay: float = 0.95, k: float = 2.0, temp: float = 0.5, eps: float = 1e-6):
        self.decay = float(decay)
        self.k = float(k)
        self.temp = float(temp)
        self.eps = float(eps)
        self.ema = None

    def step(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.detach().float().pow(2).mean().sqrt().clamp_min(self.eps)  # scalar tensor
        if self.ema is None:
            self.ema = rms
        else:
            self.ema = self.decay * self.ema + (1.0 - self.decay) * rms
        ratio = (rms / (self.ema + self.eps)).clamp_min(self.eps)
        gate = torch.sigmoid((self.k - ratio) / self.temp)
        return gate


# ===================== SD3: Stage1 Anchor-DNA + Stage2 Projected/Constrained Editing =====================

@torch.no_grad()
def OAVC_SD3(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 40,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 3.5,
    T_start: int = 13,
    mvg: float = 0.85,
    # mask
    edit_mask: Optional[torch.FloatTensor] = None,
    fg_dilate: int = 1,
    bd_dilate: int = 4,
    beta_max: float = 0.45,
    beta_pow: float = 2.0,
    # -------- Stage1 (Anchor-DNA) knobs --------
    stage1_lambda_fg: float = 0.7,     # 0 => original DNAEdit; >0 => soften foreground alignment
    stage1_bg_dilate: int = 1,         # bg mask uses dilated fg then invert
    stage1_proj_bg: float = 0.0,       # in [0,1], mix in bg-orthogonal residual
    # -------- Stage2 knobs --------
    use_projected_injection: bool = True,
    use_conf_gate: bool = True,
    gate_decay: float = 0.95,
    gate_k: float = 2.0,
    gate_temp: float = 0.5,
    ring_highpass: bool = True,
    hp_kernel: int = 3,
):
    """
    SD3.5:
      Stage1: Anchor-DNA (posterior alignment, but soften FG alignment; optional bg-direction suppression)
      Stage2: Projected/Constrained Editing (projection injection + confidence gate + optional ring high-pass)
    """
    device = x_src.device
    B, C_lat, H_lat, W_lat = x_src.shape

    # ===== mask -> latent scale =====
    M = None
    M_fg = None
    M_bg = None
    if edit_mask is not None:
        if edit_mask.dim() == 3:
            edit_mask = edit_mask.unsqueeze(0)
        if edit_mask.shape[0] == 1 and B > 1:
            edit_mask = edit_mask.expand(B, -1, -1, -1)
        M = edit_mask.to(device=device, dtype=x_src.dtype).clamp(0, 1)

        m_bin = (M > 0.5).to(dtype=x_src.dtype)
        fg = _dilate_mask(m_bin, max(0, fg_dilate))
        fg = _dilate_mask(fg, max(0, stage1_bg_dilate))
        M_fg = fg.clamp(0, 1)                       # [B,1,H,W]
        M_bg = (1.0 - M_fg).clamp(0, 1)             # [B,1,H,W]

    # ===== timesteps =====
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    # ===== encode prompts =====
    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # CFG concat
    pipe._guidance_scale = src_guidance_scale
    if pipe.do_classifier_free_guidance:
        src_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds], dim=0)
        src_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds], dim=0)

    pipe._guidance_scale = tar_guidance_scale
    if pipe.do_classifier_free_guidance:
        tar_prompt_embeds = torch.cat([tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
        tar_pooled_prompt_embeds = torch.cat([tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

    # ===================== Stage 1: Anchor-DNA =====================
    timesteps_ext = torch.cat([timesteps, torch.tensor([0], device=timesteps.device)])
    inver_timesteps = torch.flip(timesteps_ext, dims=[0])

    jmp = T_start
    last = x_src.clone()
    last_lst, v_lst, dx_lst = [], [], []
    random_noise = torch.randn_like(x_src)

    for i, (t_curr, t_prev) in enumerate(zip(inver_timesteps[:-1], inver_timesteps[1:])):
        if len(inver_timesteps) - 1 - i == jmp:
            break

        t_curr_f, t_prev_f = (t_curr / 1000.0), (t_prev / 1000.0)
        x_curr = last
        x_prev = (t_prev_f - t_curr_f) / (1 - t_curr_f) * (random_noise - last) + last

        # evaluate v_src at x_prev
        if pipe.do_classifier_free_guidance:
            src_latent_model_input = torch.cat([x_prev, x_prev], dim=0)
        else:
            src_latent_model_input = x_prev
        src_latent_model_input = src_latent_model_input.to(dtype=x_src.dtype)

        noise_pred_src = pipe.transformer(
            hidden_states=src_latent_model_input,
            timestep=(t_prev_f * 1000).expand(src_latent_model_input.shape[0]),
            encoder_hidden_states=src_prompt_embeds,
            pooled_projections=src_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_uncond, src_text = noise_pred_src.chunk(2)
            noise_pred_src = src_uncond + src_guidance_scale * (src_text - src_uncond)

        # delta_v_dna := v_linear - v_src
        delta_v_dna = (x_prev - x_curr) / (t_prev_f - t_curr_f) - noise_pred_src

        # ---- (A) weighted LS: soften FG alignment (keep iteration stable, leave edit freedom) ----
        if (M_fg is not None) and (stage1_lambda_fg > 0):
            w = (1.0 - stage1_lambda_fg * M_fg).clamp(0.0, 1.0).to(dtype=delta_v_dna.dtype)  # bg=1, fg<1
            delta_v_used = delta_v_dna * w
        else:
            delta_v_used = delta_v_dna

        # ---- (B)(C) bg-weighted orthogonal residual (direction = v_src, coeff computed on bg support) ----
        if (M_bg is not None) and (stage1_proj_bg > 0):
            dv_bg_perp = proj_perp_bg(delta_v_used, noise_pred_src, M_bg.to(delta_v_used.dtype))
            # mix: keep some original (for stability), inject some perp (for bg anti-drift)
            dv_bg_mix = (1.0 - stage1_proj_bg) * (M_bg * delta_v_used) + stage1_proj_bg * dv_bg_perp
            # combine with non-bg part unchanged
            delta_v_used = (1.0 - M_bg) * delta_v_used + dv_bg_mix

        # integrate DNA correction (keep your float32 safety)
        x_prev_f = x_prev.to(torch.float32)
        last = x_prev_f - delta_v_used.to(torch.float32) * (t_prev_f - t_curr_f)
        dx = delta_v_used * (t_prev_f - t_curr_f)
        last = last.to(noise_pred_src.dtype)

        random_noise_f = random_noise.to(torch.float32)
        random_noise = (random_noise_f - delta_v_used.to(torch.float32) * (1 - t_curr_f)).to(noise_pred_src.dtype)

        last_lst.append(last)
        dx_lst.append(dx.to(noise_pred_src.dtype))
        v_lst.append(noise_pred_src)

    last_lst = last_lst[::-1]
    v_lst = v_lst[::-1]
    dx_lst = dx_lst[::-1]

    random_noise = last_lst[0]
    x_ref = x_src.clone()

    # gate state
    gate = ScalarGateEMA(decay=gate_decay, k=gate_k, temp=gate_temp) if use_conf_gate else None

    # ===================== Stage 2: Projected / Constrained Editing =====================
    t_start_f = float(timesteps_ext[jmp].item() / 1000.0) if jmp < len(timesteps_ext) else 1.0
    for i, (t_curr, t_prev) in enumerate(zip(timesteps_ext[:-1], timesteps_ext[1:])):
        if i < jmp:
            continue

        timestep = t_curr
        t_curr_f, t_prev_f = (t_curr / 1000.0), (t_prev / 1000.0)
        dt = (t_prev_f - t_curr_f)

        # corrected latent for target evaluation
        if pipe.do_classifier_free_guidance:
            tgt_latent_model_input = torch.cat(
                [random_noise + dx_lst[i - jmp], random_noise + dx_lst[i - jmp]], dim=0
            )
        else:
            tgt_latent_model_input = random_noise + dx_lst[i - jmp]
        tgt_latent_model_input = tgt_latent_model_input.to(dtype=x_src.dtype)

        noise_pred_tgt = pipe.transformer(
            hidden_states=tgt_latent_model_input,
            timestep=timestep.expand(tgt_latent_model_input.shape[0]),
            encoder_hidden_states=tar_prompt_embeds,
            pooled_projections=tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            tgt_uncond, tgt_text = noise_pred_tgt.chunk(2)
            noise_pred_tgt = tgt_uncond + tar_guidance_scale * (tgt_text - tgt_uncond)

        v_src = v_lst[i - jmp]

        # build W_ref / W_edit / W_ring
        W_ref = None
        W_edit = None
        W_ring = None
        if M is not None:
            # tt = float(max(t_curr_f, 1e-6))
            # beta_t = float(beta_max * (tt ** beta_pow))
            # W_ref = build_weight_map(M, fg_dilate=fg_dilate, bd_dilate=bd_dilate, beta_t=beta_t).to(dtype=x_src.dtype)
            beta_t = beta_schedule(
                t_curr_f=float(t_curr_f),
                t_start_f=t_start_f,
                beta_max=beta_max,
                beta_pow=beta_pow,
            )
            W_ref = build_weight_map(
                M,
                fg_dilate=fg_dilate,
                bd_dilate=bd_dilate,
                beta_t=float(beta_t),
            ).to(dtype=x_src.dtype)

            m_bin = (M > 0.5).to(dtype=x_src.dtype)
            W_edit = _dilate_mask(m_bin, max(0, fg_dilate)).clamp(0, 1)  # foreground only
            W_ring = (W_ref - W_edit).clamp(0, 1)

        # delta between target and source
        delta = noise_pred_tgt - v_src

        # projected injection (global perp to v_src)
        if use_projected_injection:
            delta = proj_perp(delta, v_src)

        # ring high-pass (spatial)
        if ring_highpass and (W_ring is not None):
            delta_hp = high_pass_spatial(delta, k=hp_kernel)
            delta = delta * (1.0 - W_ring) + delta_hp * W_ring

        # scalar confidence gate
        g = delta.new_tensor(1.0)
        if gate is not None:
            g = gate.step(delta)

        # reference update uses W_ref
        delta_ref = delta * g
        if W_ref is not None:
            delta_ref = delta_ref * W_ref

        x_ref = x_ref.to(torch.float32)
        x_ref = (x_ref + dt * delta_ref.to(torch.float32)).to(noise_pred_tgt.dtype)

        # MVG correction
        v_mvg = (random_noise - x_ref) / max(float(t_curr_f), 1e-6)
        v_fg = mvg * noise_pred_tgt + (1.0 - mvg) * v_mvg

        # final injected delta (also optionally perpendicular)
        if use_projected_injection:
            delta_fg = proj_perp(v_fg - v_src, v_src)
        else:
            delta_fg = (v_fg - v_src)

        # foreground-only injection (NOT UniEdit fusion)
        if W_edit is not None:
            v = v_src + (W_edit * g) * delta_fg
        else:
            v = v_src + g * delta_fg

        random_noise = random_noise.to(torch.float32)
        random_noise = (random_noise + v.to(torch.float32) * dt).to(v.dtype)

    return random_noise


# ===================== FLUX: Stage1 Anchor-DNA + Stage2 Projected/Constrained Editing =====================

@torch.no_grad()
def OAVC_FLUX(
    pipe,
    scheduler,
    x_src,  # spatial VAE latent: [B,4,H_lat,W_lat]
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    T_steps: int = 28,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 2.5,
    T_start: int = 4,
    mvg: float = 0.85,
    edit_mask: Optional[torch.FloatTensor] = None,  # [1,1,H_lat,W_lat]
    mask_thresh: float = 0.3,
    fg_dilate: int = 0,
    bd_dilate: int = 1,
    beta_max: float = 0.45,
    beta_pow: float = 2.0,
    # -------- Stage1 knobs --------
    stage1_lambda_fg: float = 0.7,   # (补齐) 0 => original DNAEdit
    stage1_proj_bg: float = 0.0,     # mix-in bg-orth residual
    # -------- Stage2 knobs --------
    use_projected_injection: bool = True,
    use_conf_gate: bool = True,
    gate_decay: float = 0.95,
    gate_k: float = 2.0,
    gate_temp: float = 0.5,
    ring_highpass: bool = True,
    hp_kernel: int = 3
):
    """
    FLUX:
      - keep your debug logic: timestep/1000, height/width, pack/unpack
      - Stage1: Anchor-DNA (soften FG alignment + optional bg-weighted orth residual)
      - Stage2: projected injection + confidence gate + token high-pass on ring
    """
    device = x_src.device

    # (0) height/width
    orig_height = x_src.shape[2] * pipe.vae_scale_factor
    orig_width = x_src.shape[3] * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    # (1) check inputs
    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    # (2) prepare latents & pack
    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src,
    )
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3]
    )
    latent_tar_image_ids = latent_src_image_ids

    # (3) timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    # (4) encode prompt
    src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, device=device
    )
    tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt, prompt_2=None, device=device
    )

    # (5) guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # (6) mask -> packed fg/ring + background mask (packed)
    W_fg_packed = None
    W_bd_packed = None
    M_bg_packed = None

    if edit_mask is not None:
        H_lat, W_lat = x_src.shape[2], x_src.shape[3]
        if edit_mask.dim() == 3:
            edit_mask = edit_mask.unsqueeze(0)

        edit_mask = F.interpolate(
            edit_mask.to(device=device, dtype=torch.float32),
            size=(H_lat, W_lat),
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

        M_fg, M_bd = build_fg_bd_from_mask(
            edit_mask,
            thresh=mask_thresh,
            fg_dilate=fg_dilate,
            bd_dilate=bd_dilate,
        )
        M_bg = (1.0 - M_fg).clamp(0, 1)

        # repeat channels then pack
        M_fg_c = M_fg.repeat(1, num_channels_latents, 1, 1)
        M_bd_c = M_bd.repeat(1, num_channels_latents, 1, 1)
        M_bg_c = M_bg.repeat(1, num_channels_latents, 1, 1)

        W_fg_packed = pipe._pack_latents(M_fg_c, M_fg_c.shape[0], num_channels_latents, H_lat, W_lat).clamp(0, 1).to(x_src_packed.dtype)
        W_bd_packed = pipe._pack_latents(M_bd_c, M_bd_c.shape[0], num_channels_latents, H_lat, W_lat).clamp(0, 1).to(x_src_packed.dtype)
        M_bg_packed = pipe._pack_latents(M_bg_c, M_bg_c.shape[0], num_channels_latents, H_lat, W_lat).clamp(0, 1).to(x_src_packed.dtype)

    # ===================== Stage1: Anchor-DNA =====================
    timesteps_ext = torch.cat([timesteps, torch.tensor([0], device=timesteps.device)])
    inver_timesteps = torch.flip(timesteps_ext, dims=[0])

    jmp = T_start
    last = x_src_packed.clone()
    last_lst, v_lst, dx_lst = [], [], []
    random_noise = torch.randn_like(x_src_packed)

    for step_idx, (t_curr, t_prev) in enumerate(zip(inver_timesteps[:-1], inver_timesteps[1:])):
        if len(inver_timesteps) - 1 - step_idx == jmp:
            break

        t_curr_f, t_prev_f = t_curr / 1000.0, t_prev / 1000.0
        x_curr = last
        x_prev = (t_prev_f - t_curr_f) / (1 - t_curr_f) * (random_noise - last) + last

        v_src = pipe.transformer(
            hidden_states=x_prev,
            timestep=(t_prev).expand(x_prev.shape[0]) / 1000.0,
            guidance=src_guidance,
            encoder_hidden_states=src_prompt_embeds,
            txt_ids=src_text_ids,
            img_ids=latent_src_image_ids,
            pooled_projections=src_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # delta_v_dna := v_linear - v_src
        delta_v_dna = (x_prev - x_curr) / (t_prev_f - t_curr_f) - v_src

        # (A) soften FG alignment
        if (W_fg_packed is not None) and (stage1_lambda_fg > 0):
            w = (1.0 - stage1_lambda_fg * W_fg_packed).clamp(0.0, 1.0).to(dtype=delta_v_dna.dtype)
            delta_v_used = delta_v_dna * w
        else:
            delta_v_used = delta_v_dna

        # (B)(C) bg-weighted orth residual (direction uses v_src, coeff computed on bg support)
        if (M_bg_packed is not None) and (stage1_proj_bg > 0):
            dv_bg_perp = proj_perp_bg(delta_v_used, v_src, M_bg_packed)
            dv_bg_mix = (1.0 - stage1_proj_bg) * (M_bg_packed * delta_v_used) + stage1_proj_bg * dv_bg_perp
            delta_v_used = (1.0 - M_bg_packed) * delta_v_used + dv_bg_mix

        # integrate DNA correction
        last = (x_prev.to(torch.float32) - delta_v_used.to(torch.float32) * (t_prev_f - t_curr_f)).to(v_src.dtype)
        dx = (delta_v_used * (t_prev_f - t_curr_f)).to(v_src.dtype)

        random_noise = (
            random_noise.to(torch.float32) - delta_v_used.to(torch.float32) * (1 - t_curr_f)
        ).to(v_src.dtype)

        last_lst.append(last)
        dx_lst.append(dx)
        v_lst.append(v_src)

    last_lst = last_lst[::-1]
    v_lst = v_lst[::-1]
    dx_lst = dx_lst[::-1]

    random_noise = last_lst[0]
    x_ref = x_src_packed.clone()

    # gate state
    gate = ScalarGateEMA(decay=gate_decay, k=gate_k, temp=gate_temp) if use_conf_gate else None

    # ===================== Stage2: Projected / Constrained Editing =====================
    t_start_f = float(timesteps_ext[jmp].item() / 1000.0) if jmp < len(timesteps_ext) else 1.0

    for i, (t_curr, t_prev) in enumerate(zip(timesteps_ext[:-1], timesteps_ext[1:])):
        if i < jmp:
            continue

        t_curr_f, t_prev_f = t_curr / 1000.0, t_prev / 1000.0
        dt = (t_prev_f - t_curr_f)

        # W_t (fg + ring)
        W_t = None
        W_ring = None
        if (W_fg_packed is not None) and (W_bd_packed is not None):
            b = beta_schedule(
                t_curr_f=float(t_curr_f),
                t_start_f=t_start_f,
                beta_max=beta_max,
                beta_pow=beta_pow,
            )
            W_t = (W_fg_packed + b * W_bd_packed).clamp(0, 1)
            W_ring = (W_t - W_fg_packed).clamp(0, 1)

        # corrected latent for target eval
        tgt_latent_model_input = random_noise + dx_lst[i - jmp]

        v_tgt = pipe.transformer(
            hidden_states=tgt_latent_model_input,
            timestep=(t_curr).expand(tgt_latent_model_input.shape[0]) / 1000.0,
            guidance=tar_guidance,
            encoder_hidden_states=tar_prompt_embeds,
            txt_ids=tar_text_ids,
            img_ids=latent_tar_image_ids,
            pooled_projections=tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        v_src = v_lst[i - jmp]

        delta = v_tgt - v_src
        if use_projected_injection:
            delta = proj_perp(delta, v_src)

        # ring token high-pass
        if ring_highpass and (W_ring is not None):
            delta_hp = high_pass_token(delta)
            delta = delta * (1.0 - W_ring) + delta_hp * W_ring

        # gate
        g = delta.new_tensor(1.0)
        if gate is not None:
            g = gate.step(delta)

        # reference update uses W_t
        delta_ref = delta * g
        if W_t is not None:
            delta_ref = delta_ref * W_t

        x_ref = (x_ref.to(torch.float32) + dt * delta_ref.to(torch.float32)).to(v_tgt.dtype)

        # MVG
        v_mvg = (random_noise - x_ref) / max(float(t_curr_f), 1e-6)
        v_fg = mvg * v_tgt + (1.0 - mvg) * v_mvg

        if use_projected_injection:
            delta_fg = proj_perp(v_fg - v_src, v_src)
        else:
            delta_fg = (v_fg - v_src)

        # foreground-only injection
        if W_fg_packed is not None:
            v = v_src + (W_fg_packed * g) * delta_fg
        else:
            v = v_src + g * delta_fg

        random_noise = (random_noise.to(torch.float32) + v.to(torch.float32) * dt).to(v_tgt.dtype)

    # unpack to spatial latent
    unpacked_out = pipe._unpack_latents(random_noise, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out
