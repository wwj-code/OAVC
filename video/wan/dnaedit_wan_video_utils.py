# wan/dnaedit_wan_video_utils.py
# ==========================================================
# Your DNAEdit-based innovations migrated to Wan2.1 video latent
# - Stage1: Anchor-DNA with soften FG + optional bg-weighted projection residual
# - Stage2: Projected / Constrained editing with projected injection + EMA gate + ring high-pass
#
# NOTE:
# Wan latent is typically [C, T, H, W] per sample (your current code uses latents[0]).
# We implement utilities for tensors of shape [C,T,H,W] and masks [1,T,H,W].
# ==========================================================

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# -------------------- basic helpers --------------------

def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    # make [1, ...]
    return x.unsqueeze(0) if x.dim() >= 1 and x.shape[0] != 1 else x

def _sum_over_nonbatch(x: torch.Tensor) -> torch.Tensor:
    # x: [B, ...]
    dims = list(range(1, x.dim()))
    return x.sum(dim=dims, keepdim=True)

def proj_parallel(delta: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # delta/ref: [B,...]
    num = _sum_over_nonbatch(delta * ref)
    den = _sum_over_nonbatch(ref * ref).clamp_min(eps)
    return (num / den) * ref

def proj_perp(delta: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return delta - proj_parallel(delta, ref, eps=eps)

def proj_parallel_bg(delta: torch.Tensor, ref: torch.Tensor, W_bg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # <a,b>_bg = sum((W_bg*a)*(W_bg*b))
    r_w = W_bg * ref
    d_w = W_bg * delta
    num = _sum_over_nonbatch(d_w * r_w)
    den = _sum_over_nonbatch(r_w * r_w).clamp_min(eps)
    alpha = num / den
    return alpha * r_w

def proj_perp_bg(delta: torch.Tensor, ref: torch.Tensor, W_bg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r = W_bg * delta
    return r - proj_parallel_bg(r, ref, W_bg, eps=eps)


# -------------------- mask helpers (video) --------------------

def dilate_mask_video(mask: torch.Tensor, iters: int) -> torch.Tensor:
    """
    mask: [B,1,T,H,W] float in [0,1]
    dilation is applied per-frame (2D).
    """
    if iters <= 0:
        return mask
    B, _, T, H, W = mask.shape
    x = mask.reshape(B * T, 1, H, W)
    for _ in range(iters):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x.reshape(B, 1, T, H, W)

def build_fg_ring_bg_video(
    edit_mask: torch.Tensor,
    thresh: float = 0.3,
    fg_dilate: int = 2,
    bd_dilate: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    edit_mask: [B,1,T,H,W] float in [0,1]
    returns:
      M_fg   [B,1,T,H,W]
      M_ring [B,1,T,H,W]
      M_bg   [B,1,T,H,W]
    """
    M_fg = (edit_mask > thresh).float()
    M_fg = dilate_mask_video(M_fg, fg_dilate).clamp(0, 1)
    M_bd = dilate_mask_video(M_fg, bd_dilate).clamp(0, 1)
    M_ring = (M_bd - M_fg).clamp(0, 1)
    M_bg = (1.0 - M_fg).clamp(0, 1)
    return M_fg, M_ring, M_bg

def beta_schedule(t_curr_f: float, t_start_f: float, beta_max: float, beta_pow: float) -> float:
    if t_start_f <= 1e-8:
        return 0.0
    r = float(t_curr_f / t_start_f)
    r = max(0.0, min(1.0, r))
    return float(beta_max * (r ** beta_pow))


# -------------------- ring high-pass (video latent) --------------------

def high_pass_video(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    x: [B,C,T,H,W]
    apply x - avgpool2d(x) per frame
    """
    if x.dim() != 5:
        return x
    B, C, T, H, W = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # [B*T, C, H, W]
    blur = F.avg_pool2d(y, kernel_size=k, stride=1, padding=k // 2)
    hp = (y - blur).reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return hp


# -------------------- confidence gate --------------------

class ScalarGateEMA:
    """
    scalar confidence gate:
    - track EMA of RMS magnitude
    - if current RMS >> EMA, gate decreases
    """
    def __init__(self, decay: float = 0.95, k: float = 2.0, temp: float = 0.5, eps: float = 1e-6):
        self.decay = float(decay)
        self.k = float(k)
        self.temp = float(temp)
        self.eps = float(eps)
        self.ema = None

    def step(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.detach().float().pow(2).mean().sqrt().clamp_min(self.eps)  # scalar
        if self.ema is None:
            self.ema = rms
        else:
            self.ema = self.decay * self.ema + (1.0 - self.decay) * rms
        ratio = (rms / (self.ema + self.eps)).clamp_min(self.eps)
        gate = torch.sigmoid((self.k - ratio) / self.temp)
        return gate


# ==========================================================
# Main: Wan video editing with your weighted DNAEdit innovations
# ==========================================================

@torch.no_grad()
def wan_dnaedit_video_weighted(
    model,                       # WanModel
    sample_scheduler,            # FlowUniPC/DPMSolver scheduler
    timesteps,                   # torch 1D timesteps
    latents_src_4d: torch.Tensor,   # [C,T,H,W] from VAE.encode(video)[0]
    rand_noise_4d: torch.Tensor,    # [C,T,H,W] initial rand noise used in stage-1
    context_src, context_tgt, context_null,
    seq_len: int,
    guide_scale: float,
    tgt_guide_scale: float,
    jmp: int = 12,

    # mask at latent scale (preferred): [1,T,H,W] in [0,1]
    edit_mask_latent_4d: Optional[torch.Tensor] = None,

    # mask knobs
    mask_thresh: float = 0.3,
    fg_dilate: int = 2,
    bd_dilate: int = 4,
    beta_max: float = 0.45,
    beta_pow: float = 2.0,

    # Stage-1 knobs
    stage1_lambda_fg: float = 0.7,
    stage1_bg_dilate: int = 1,
    stage1_proj_bg: float = 0.1,

    # Stage-2 knobs
    use_projected_injection: bool = True,
    use_conf_gate: bool = True,
    gate_decay: float = 0.95,
    gate_k: float = 2.0,
    gate_temp: float = 0.5,
    ring_highpass: bool = True,
    hp_kernel: int = 3,

    # optional: hard bg blend at the end
    hard_bg_blend: bool = False,
):
    """
    Returns edited latent [C,T,H,W]
    """

    device = latents_src_4d.device
    arg_src = {'context': context_src, 'seq_len': seq_len}
    arg_tgt = {'context': context_tgt, 'seq_len': seq_len}
    arg_null = {'context': context_null, 'seq_len': seq_len}

    # ---------------- mask build ----------------
    M_fg = M_ring = M_bg = None
    if edit_mask_latent_4d is not None:
        # make [B,1,T,H,W]
        if edit_mask_latent_4d.dim() == 4:
            # [1,T,H,W] -> [1,1,T,H,W]
            edit_mask = edit_mask_latent_4d.unsqueeze(1)
        elif edit_mask_latent_4d.dim() == 5:
            edit_mask = edit_mask_latent_4d
        else:
            raise ValueError(f"edit_mask_latent_4d must be 4D/5D, got {edit_mask_latent_4d.shape}")

        edit_mask = edit_mask.to(device=device, dtype=torch.float32).clamp(0, 1)
        M_fg, M_ring, M_bg = build_fg_ring_bg_video(
            edit_mask,
            thresh=mask_thresh,
            fg_dilate=fg_dilate,
            bd_dilate=bd_dilate,
        )

        # Stage1 bg uses extra dilation on fg then invert (as your SD3 version)
        if stage1_bg_dilate > 0:
            M_fg_s1 = dilate_mask_video((edit_mask > 0.5).float(), stage1_bg_dilate).clamp(0, 1)
            M_bg_s1 = (1.0 - M_fg_s1).clamp(0, 1)
        else:
            M_bg_s1 = M_bg
    else:
        M_bg_s1 = None

    # ---------------- Stage 1: Anchor-DNA ----------------
    # Your current Stage-1 is implemented on 4D [C,T,H,W] without batch.
    # We keep the same structure and only replace delta_v -> delta_v_used with your weights.

    latents_src = latents_src_4d.clone()  # [C,T,H,W]
    prev_latent = latents_src.clone()
    rand_noise = rand_noise_4d.clone()

    comp_lst = []
    v_lst = []
    last_lst = []

    # NOTE: your current code flips timesteps for stage1
    tlist = timesteps.flip(dims=[0])

    for i, t_curr in enumerate(tlist):
        t_prev = torch.tensor(0, device=device, dtype=t_curr.dtype) if i == 0 else tlist[i - 1]

        # build curr_latent by your interpolation formula
        curr_latent = rand_noise * (t_curr - t_prev) / (1000.0 - t_prev) + prev_latent * (1000.0 - t_curr) / (1000.0 - t_prev)

        timestep = torch.stack([t_curr])

        # predict v_src at curr_latent
        noise_pred_cond_curr = model([curr_latent], t=timestep, **arg_src)[0]
        noise_pred_uncond_curr = model([curr_latent], t=timestep, **arg_null)[0]
        v_src = noise_pred_uncond_curr + guide_scale * (noise_pred_cond_curr - noise_pred_uncond_curr)

        dt = (t_curr - t_prev) / 1000.0
        v_linear = (curr_latent - prev_latent) / dt
        # your code: delta_v = v_src - v_linear
        delta_v = v_src - v_linear

        # -------- (A) soften FG alignment (reduce correction on FG) --------
        if (M_fg is not None) and (stage1_lambda_fg > 0):
            # w: bg=1, fg<1
            # M_fg is [B,1,T,H,W], delta_v is [C,T,H,W]
            w = (1.0 - stage1_lambda_fg * M_fg).clamp(0.0, 1.0)  # [1,1,T,H,W]
            w = w.expand(1, delta_v.shape[0], -1, -1, -1)        # [1,C,T,H,W]
            delta_v_used = delta_v.unsqueeze(0) * w
            delta_v_used = delta_v_used.squeeze(0)
        else:
            delta_v_used = delta_v

        # -------- (B)(C) bg-weighted orth residual (direction = v_src, coeff computed on bg) --------
        if (M_bg_s1 is not None) and (stage1_proj_bg > 0):
            # prepare batch tensors
            dv_b = delta_v_used.unsqueeze(0)          # [1,C,T,H,W]
            vs_b = v_src.unsqueeze(0)                 # [1,C,T,H,W]
            # background support expanded to channels
            Wbg = M_bg_s1.expand(1, dv_b.shape[1], -1, -1, -1).to(dv_b.dtype)  # [1,C,T,H,W]

            dv_bg_perp = proj_perp_bg(dv_b, vs_b, Wbg)                         # [1,C,T,H,W]
            dv_bg_mix = (1.0 - stage1_proj_bg) * (Wbg * dv_b) + stage1_proj_bg * dv_bg_perp
            dv_new = (1.0 - Wbg) * dv_b + dv_bg_mix
            delta_v_used = dv_new.squeeze(0)

        # integrate correction (same as your code)
        curr_latent = curr_latent + delta_v_used * dt
        prev_latent = curr_latent
        rand_noise = rand_noise + delta_v_used * (1000.0 - t_prev) / 1000.0

        comp_lst.append((delta_v_used * dt).clone())
        v_lst.append(v_src.clone())
        last_lst.append(prev_latent.clone())

    # reverse lists as in your code
    comp_lst = comp_lst[::-1]
    v_lst = v_lst[::-1]
    last_lst = last_lst[::-1]

    # stage2 starts from last_lst[jmp]
    if jmp < 0 or jmp >= len(last_lst):
        raise ValueError(f"jmp={jmp} out of range, got len(last_lst)={len(last_lst)}")
    edit_latent = last_lst[jmp].clone()    # [C,T,H,W]
    x_ref = latents_src_4d.clone()         # reference latent for MVG

    gate = ScalarGateEMA(decay=gate_decay, k=gate_k, temp=gate_temp) if use_conf_gate else None

    # restore descending timesteps for stage2 (same as your code)
    tlist2 = tlist.flip(dims=[0])  # back

    # t_start_f for beta schedule
    t_start_f = float(tlist2[jmp].item() / 1000.0)

    # ---------------- Stage 2: Projected / Constrained Editing ----------------
    for i, t_curr in enumerate(tlist2):
        if i < jmp:
            continue
        t_next = torch.tensor(0, device=device, dtype=t_curr.dtype) if i == len(tlist2) - 1 else tlist2[i + 1]
        timestep = torch.stack([t_curr])

        # evaluate target velocity at (edit_latent + comp_lst[i]) (your original style)
        inp = edit_latent + comp_lst[i]

        noise_pred_cond_tgt = model([inp], t=timestep, **arg_tgt)[0]
        noise_pred_uncond_tgt = model([inp], t=timestep, **arg_null)[0]
        v_tgt = noise_pred_uncond_tgt + tgt_guide_scale * (noise_pred_cond_tgt - noise_pred_uncond_tgt)

        v_src = v_lst[i]  # cached

        dt = (t_next - t_curr) / 1000.0  # negative

        # -------- build weights W_fg / W_ring / W_t at this timestep --------
        W_fg = W_ring = W_t = None
        if M_fg is not None and M_ring is not None:
            b = beta_schedule(
                t_curr_f=float(t_curr.item() / 1000.0),
                t_start_f=t_start_f,
                beta_max=beta_max,
                beta_pow=beta_pow,
            )
            # M_fg/M_ring: [1,1,T,H,W]
            W_fg = M_fg.to(dtype=torch.float32)       # [1,1,T,H,W]
            W_ring = M_ring.to(dtype=torch.float32)   # [1,1,T,H,W]
            W_t = (W_fg + b * W_ring).clamp(0, 1)     # [1,1,T,H,W]

            # expand to channels
            W_fg = W_fg.expand(1, edit_latent.shape[0], -1, -1, -1).to(edit_latent.dtype)      # [1,C,T,H,W]
            W_ring = W_ring.expand(1, edit_latent.shape[0], -1, -1, -1).to(edit_latent.dtype)
            W_t = W_t.expand(1, edit_latent.shape[0], -1, -1, -1).to(edit_latent.dtype)

        # delta = v_tgt - v_src
        delta = (v_tgt - v_src)

        # projected injection: keep only perp to v_src
        if use_projected_injection:
            delta = proj_perp(delta.unsqueeze(0), v_src.unsqueeze(0)).squeeze(0)

        # ring high-pass (spatial)
        if ring_highpass and (W_ring is not None):
            delta_b = delta.unsqueeze(0)                          # [1,C,T,H,W]
            delta_hp = high_pass_video(delta_b, k=hp_kernel)
            delta_b = delta_b * (1.0 - W_ring) + delta_hp * W_ring
            delta = delta_b.squeeze(0)

        # gate
        g = delta.new_tensor(1.0)
        if gate is not None:
            g = gate.step(delta)

        # reference update: x_ref += dt * (W_t * g * delta)
        if W_t is not None:
            delta_ref = (delta.unsqueeze(0) * W_t) * g
            delta_ref = delta_ref.squeeze(0)
        else:
            delta_ref = delta * g

        x_ref = (x_ref.to(torch.float32) + dt * delta_ref.to(torch.float32)).to(edit_latent.dtype)

        # MVG
        t_curr_f = max(float(t_curr.item() / 1000.0), 1e-6)
        v_mvg = (edit_latent - x_ref) / t_curr_f
        v_fg = 0.85 * v_tgt + (1.0 - 0.85) * v_mvg   # keep mvg fixed at 0.85 here; you can pass it if needed

        # final injected delta
        delta_fg = (v_fg - v_src)
        if use_projected_injection:
            delta_fg = proj_perp(delta_fg.unsqueeze(0), v_src.unsqueeze(0)).squeeze(0)

        # foreground-only injection: v = v_src + (W_fg*g) * delta_fg
        if W_fg is not None:
            v = v_src.unsqueeze(0) + (W_fg * g) * delta_fg.unsqueeze(0)
            v = v.squeeze(0)
        else:
            v = v_src + g * delta_fg

        edit_latent = (edit_latent.to(torch.float32) + v.to(torch.float32) * dt).to(edit_latent.dtype)

    # optional hard bg blend (use mask foreground, keep background from src)
    if hard_bg_blend and (M_fg is not None):
        # M_fg: [1,1,T,H,W] -> expand channels
        W = M_fg.expand(1, edit_latent.shape[0], -1, -1, -1).to(edit_latent.dtype).squeeze(0)
        edit_latent = W * edit_latent + (1.0 - W) * latents_src_4d

    return edit_latent
