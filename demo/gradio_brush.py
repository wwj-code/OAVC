"""Interactive OAVC demo (Gradio).

Upload an image, paint the region to edit with the brush (and/or let SAM3 find the object from a
text prompt), enter source / target prompts and run OAVC. The brush is useful when the automatic
object support is wrong or incomplete (see the interactive brush refinement example in the paper).

    python demo/gradio_brush.py --model_type SD3                       # brush only
    python demo/gradio_brush.py --model_type FLUX --sam3_ckpt checkpoints/sam3/sam3.pt
"""
import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oavc"))
from oavc_sampler import OAVC_FLUX, OAVC_SD3  # noqa: E402

# Paper settings (configs/sd35.yaml, configs/flux.yaml and the run_pie_bench.py defaults).
PAPER_DEFAULTS = {
    "SD3": dict(T_steps=40, T_start=13, src_guidance_scale=1.0, tar_guidance_scale=3.5, mvg=0.8),
    "FLUX": dict(T_steps=28, T_start=4, src_guidance_scale=1.0, tar_guidance_scale=2.5, mvg=0.8),
}
OAVC_DEFAULTS = dict(mask_thresh=0.5, fg_dilate=2, bd_dilate=4, beta_max=0.45, beta_pow=2.0,
                     stage1_lambda_fg=0.6, stage1_bg_dilate=1, stage1_proj_bg=0.1,
                     use_projected_injection=True, use_conf_gate=True, gate_decay=0.95, gate_k=2.0,
                     gate_temp=0.5, ring_highpass=True, hp_kernel=3)
# SAM3 settings used to build the PIE-Bench mask cache (oavc/precompute_sam3_masks.py).
SAM3_DEFAULTS = dict(thresh=0.25, dilate_iter=2, score_thresh=0.3, area_min_ratio=0.01)


# -----------------------------
# Helpers
# -----------------------------
def crop_to_multiple_of_16(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % 16, : w - w % 16]


def read_canvas(value: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Gradio ImageEditor value -> (RGB image HxWx3 uint8, brush alpha HxW in [0,1] or None)."""
    if value is None or value.get("background") is None:
        raise gr.Error("Please upload an image first.")
    bg = value["background"]
    bg = np.array(bg.convert("RGBA")) if isinstance(bg, Image.Image) else np.asarray(bg)
    rgb = crop_to_multiple_of_16(bg)[:, :, :3].astype(np.uint8)

    alpha = None
    for layer in value.get("layers") or []:
        layer = np.array(layer.convert("RGBA")) if isinstance(layer, Image.Image) else np.asarray(layer)
        if layer.ndim == 3 and layer.shape[-1] == 4:
            a = crop_to_multiple_of_16(layer)[:, :, 3].astype(np.float32) / 255.0
            alpha = a if alpha is None else np.maximum(alpha, a)
    if alpha is not None and alpha.max() <= 0:
        alpha = None
    return rgb, alpha


def to_latent_mask(mask01: np.ndarray, h_lat: int, w_lat: int, device, thresh: float, soft: bool) -> torch.Tensor:
    m = torch.from_numpy(mask01).float()[None, None]
    m = F.interpolate(m, size=(h_lat, w_lat), mode="bilinear", align_corners=False).clamp(0, 1)
    if not soft:
        m = (m > thresh).float()
    return m.to(device)


def overlay(rgb: np.ndarray, mask01: Optional[np.ndarray]) -> Image.Image:
    if mask01 is None:
        return Image.fromarray(rgb)
    a = np.clip(mask01[..., None], 0, 1) * 0.45
    red = np.zeros_like(rgb, dtype=np.float32); red[..., 0] = 255
    return Image.fromarray(np.clip(rgb * (1 - a) + red * a, 0, 255).astype(np.uint8))


# -----------------------------
# Editor
# -----------------------------
class OAVCEditor:
    def __init__(self, model_type: str, sd3_ckpt: str, flux_ckpt: str, device: str = "cuda:0",
                 sam3_ckpt: Optional[str] = None):
        self.model_type = model_type.upper()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.model_type == "SD3":
            self.pipe = StableDiffusion3Pipeline.from_pretrained(sd3_ckpt, torch_dtype=torch.float16).to(self.device)
        elif self.model_type == "FLUX":
            self.pipe = FluxPipeline.from_pretrained(flux_ckpt, torch_dtype=torch.float16).to(self.device)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.sam3 = None
        if sam3_ckpt:
            from sam3_helper import SAM3Helper  # needs the SAM3 package
            self.sam3 = SAM3Helper(checkpoint_path=sam3_ckpt, device=self.device)

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray) -> torch.Tensor:
        x = self.pipe.image_processor.preprocess(Image.fromarray(rgb)).to(self.device).half()
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            z = self.pipe.vae.encode(x).latent_dist.mode()
        return (z - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, hw: Tuple[int, int]) -> Image.Image:
        z = z / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            img = self.pipe.vae.decode(z, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(img)[0].resize((hw[1], hw[0]), Image.BILINEAR)

    def sam3_mask(self, rgb: np.ndarray, text: str, h_lat: int, w_lat: int) -> Optional[torch.Tensor]:
        if self.sam3 is None:
            raise gr.Error("SAM3 is not loaded. Start the demo with --sam3_ckpt, or use the brush.")
        if not text.strip():
            raise gr.Error("Enter the object to segment in 'SAM3 object prompt'.")
        m = self.sam3.get_mask_for_latent(pil_image=Image.fromarray(rgb), text_prompt=text.strip(),
                                          H_lat=h_lat, W_lat=w_lat, **SAM3_DEFAULTS)
        return None if m is None else m.to(self.device).float()

    @torch.inference_mode()
    def edit(self, canvas, mask_source, combine_mode, sam3_text, brush_soft, src_prompt, tar_prompt,
             negative_prompt, T_steps, T_start, src_guidance_scale, tar_guidance_scale, mvg, seed,
             mask_thresh, fg_dilate, bd_dilate, beta_max, beta_pow, stage1_lambda_fg, stage1_bg_dilate,
             stage1_proj_bg, use_projected_injection, use_conf_gate, gate_decay, gate_k, gate_temp,
             ring_highpass, hp_kernel):
        rgb, brush = read_canvas(canvas)
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        x0_src = self.encode(rgb)
        h_lat, w_lat = x0_src.shape[-2:]

        brush_m = None if brush is None else to_latent_mask(brush, h_lat, w_lat, self.device, mask_thresh, brush_soft)
        if mask_source == "brush":
            mask = brush_m
        elif mask_source == "SAM3":
            mask = self.sam3_mask(rgb, sam3_text, h_lat, w_lat)
        elif mask_source == "SAM3 + brush":
            sam_m = self.sam3_mask(rgb, sam3_text, h_lat, w_lat)
            if sam_m is None or brush_m is None:
                mask = sam_m if brush_m is None else brush_m
            else:
                mask = torch.maximum(sam_m, brush_m) if combine_mode == "union" else torch.minimum(sam_m, brush_m)
        else:  # whole image
            mask = torch.ones((1, 1, h_lat, w_lat), device=self.device)
        if mask is None or float(mask.max()) <= 0:
            raise gr.Error("Empty edit region: paint the object with the brush (or check the SAM3 prompt).")

        common = dict(x_src=x0_src, src_prompt=src_prompt, tar_prompt=tar_prompt, negative_prompt=negative_prompt,
                      T_steps=int(T_steps), T_start=int(T_start), src_guidance_scale=float(src_guidance_scale),
                      tar_guidance_scale=float(tar_guidance_scale), mvg=float(mvg), edit_mask=mask,
                      fg_dilate=int(fg_dilate), bd_dilate=int(bd_dilate), beta_max=float(beta_max),
                      beta_pow=float(beta_pow), stage1_lambda_fg=float(stage1_lambda_fg),
                      stage1_proj_bg=float(stage1_proj_bg), use_projected_injection=bool(use_projected_injection),
                      use_conf_gate=bool(use_conf_gate), gate_decay=float(gate_decay), gate_k=float(gate_k),
                      gate_temp=float(gate_temp), ring_highpass=bool(ring_highpass), hp_kernel=int(hp_kernel))
        if self.model_type == "SD3":
            x0_tar = OAVC_SD3(self.pipe, self.pipe.scheduler, stage1_bg_dilate=int(stage1_bg_dilate), **common)
        else:
            x0_tar = OAVC_FLUX(self.pipe, self.pipe.scheduler, mask_thresh=float(mask_thresh), **common)

        out = self.decode(x0_tar, rgb.shape[:2])
        mask_img = F.interpolate(mask.float().cpu(), size=rgb.shape[:2], mode="nearest")[0, 0].numpy()
        return out, overlay(rgb, mask_img)


# -----------------------------
# UI
# -----------------------------
def create_demo(editor: OAVCEditor) -> gr.Blocks:
    d = {**PAPER_DEFAULTS[editor.model_type], **OAVC_DEFAULTS}
    sources = ["brush", "SAM3", "SAM3 + brush", "whole image"] if editor.sam3 is not None else ["brush", "whole image"]
    with gr.Blocks(title="OAVC interactive editing") as demo:
        gr.Markdown(f"# OAVC interactive editing ({editor.model_type})\n"
                    "1. Upload an image. 2. Paint the object to edit with the brush"
                    + (" (or describe it for SAM3)" if editor.sam3 is not None else "") +
                    ". 3. Enter the source and target prompts and click **Run OAVC**.")
        with gr.Row():
            with gr.Column():
                canvas = gr.ImageEditor(label="Image + brush (paint the region to edit)", type="numpy",
                                        sources=["upload"],
                                        brush=gr.Brush(colors=["#FF0000"], default_color="#FF0000", color_mode="fixed"))
                src_prompt = gr.Textbox(label="Source prompt", placeholder="a cat sitting on a wooden bench")
                tar_prompt = gr.Textbox(label="Target prompt", placeholder="a dog sitting on a wooden bench")
                with gr.Row():
                    mask_source = gr.Dropdown(sources, value="brush", label="Edit region")
                    combine_mode = gr.Dropdown(["union", "intersect"], value="union", label="SAM3 + brush combine",
                                               visible=editor.sam3 is not None)
                sam3_text = gr.Textbox(label="SAM3 object prompt", placeholder="cat", visible=editor.sam3 is not None)
                run_btn = gr.Button("Run OAVC", variant="primary")
                with gr.Accordion("Sampling (paper defaults)", open=False):
                    T_steps = gr.Slider(4, 80, value=d["T_steps"], step=1, label="T_steps")
                    T_start = gr.Slider(0, 40, value=d["T_start"], step=1, label="T_start")
                    src_gs = gr.Slider(0.0, 20.0, value=d["src_guidance_scale"], step=0.1, label="source guidance")
                    tar_gs = gr.Slider(0.0, 20.0, value=d["tar_guidance_scale"], step=0.1, label="target guidance")
                    mvg = gr.Slider(0.0, 1.0, value=d["mvg"], step=0.01, label="mvg")
                    seed = gr.Number(value=0, precision=0, label="seed")
                    negative_prompt = gr.Textbox(label="Negative prompt", value="")
                with gr.Accordion("OAVC parameters (paper defaults)", open=False):
                    brush_soft = gr.Checkbox(label="soft brush mask", value=False)
                    mask_thresh = gr.Slider(0.0, 1.0, value=d["mask_thresh"], step=0.01, label="mask threshold")
                    fg_dilate = gr.Slider(0, 8, value=d["fg_dilate"], step=1, label="r_fg (fg_dilate)")
                    bd_dilate = gr.Slider(0, 12, value=d["bd_dilate"], step=1, label="r_bd (bd_dilate)")
                    beta_max = gr.Slider(0.0, 1.0, value=d["beta_max"], step=0.01, label="beta_max")
                    beta_pow = gr.Slider(0.0, 4.0, value=d["beta_pow"], step=0.1, label="gamma (beta_pow)")
                    s1_fg = gr.Slider(0.0, 1.0, value=d["stage1_lambda_fg"], step=0.01, label="lambda_fg (stage 1)")
                    s1_bgd = gr.Slider(0, 8, value=d["stage1_bg_dilate"], step=1, label="stage-1 bg dilate (SD3)")
                    s1_bg = gr.Slider(0.0, 1.0, value=d["stage1_proj_bg"], step=0.01, label="rho_bg (stage 1)")
                    proj = gr.Checkbox(label="projected injection", value=d["use_projected_injection"])
                    gate = gr.Checkbox(label="confidence gate", value=d["use_conf_gate"])
                    gate_decay = gr.Slider(0.0, 1.0, value=d["gate_decay"], step=0.01, label="gate decay")
                    gate_k = gr.Slider(0.0, 5.0, value=d["gate_k"], step=0.1, label="gate k")
                    gate_temp = gr.Slider(0.01, 2.0, value=d["gate_temp"], step=0.01, label="gate temperature")
                    hp = gr.Checkbox(label="ring high-pass", value=d["ring_highpass"])
                    hp_kernel = gr.Slider(1, 9, value=d["hp_kernel"], step=2, label="high-pass kernel (SD3)")
            with gr.Column():
                result = gr.Image(label="OAVC result")
                mask_preview = gr.Image(label="Edit region used")
        run_btn.click(editor.edit,
                      inputs=[canvas, mask_source, combine_mode, sam3_text, brush_soft, src_prompt, tar_prompt,
                              negative_prompt, T_steps, T_start, src_gs, tar_gs, mvg, seed, mask_thresh, fg_dilate,
                              bd_dilate, beta_max, beta_pow, s1_fg, s1_bgd, s1_bg, proj, gate, gate_decay, gate_k,
                              gate_temp, hp, hp_kernel],
                      outputs=[result, mask_preview])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", default="SD3", choices=["SD3", "FLUX"])
    p.add_argument("--sd3_ckpt", default="checkpoints/stable-diffusion-3.5-medium")
    p.add_argument("--flux_ckpt", default="checkpoints/FLUX.1-dev")
    p.add_argument("--sam3_ckpt", default=None, help="optional: checkpoints/sam3/sam3.pt to enable SAM3 masks")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--server_name", default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    editor = OAVCEditor(args.model_type, args.sd3_ckpt, args.flux_ckpt, args.device, args.sam3_ckpt)
    create_demo(editor).launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
