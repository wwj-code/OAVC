# run_pie_bench.py: run OAVC on PIE-Bench with cached SAM3 object supports.
import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline, FluxPipeline

from oavc_sampler import OAVC_SD3, OAVC_FLUX


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="OAVC on PIE-Bench (cached SAM3 supports)")
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--exp_yaml", type=str, required=True, help="configs/sd35.yaml or configs/flux.yaml")
    parser.add_argument("--save", type=str, required=True, help="output folder")

    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--pie_img_root", type=str, required=True)
    parser.add_argument("--sd3_ckpt", type=str, default="checkpoints/stable-diffusion-3.5-medium")
    parser.add_argument("--flux_ckpt", type=str, default="checkpoints/FLUX.1-dev")

    parser.add_argument("--mask_cache_json", type=str, required=True,
                        help="mask_cache.json written by precompute_sam3_masks.py")
    parser.add_argument("--mask_cache_root", type=str, default=None,
                        help="if set, load masks from <root>/masks/<file name> instead of the stored path")
    parser.add_argument("--fallback_full_mask", action="store_true",
                        help="use a full-image support when no mask is available")
    parser.add_argument("--only_ids", type=str, default=None,
                        help="comma-separated sample ids to run (default: all)")

    # support -> weight maps
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--fg_dilate", type=int, default=2, help="r_fg")
    parser.add_argument("--bd_dilate", type=int, default=4, help="r_bd")
    parser.add_argument("--beta_max", type=float, default=0.45)
    parser.add_argument("--beta_pow", type=float, default=2.0, help="gamma of the ring schedule")

    # Stage-1
    parser.add_argument("--stage1_lambda_fg", type=float, default=0.6, help="lambda_fg")
    parser.add_argument("--stage1_bg_dilate", type=int, default=1)
    parser.add_argument("--stage1_proj_bg", type=float, default=0.1, help="rho_bg")

    # Stage-2
    parser.add_argument("--use_projected_injection", action="store_true", default=True)
    parser.add_argument("--use_conf_gate", action="store_true", default=True)
    parser.add_argument("--gate_decay", type=float, default=0.95)
    parser.add_argument("--gate_k", type=float, default=2.0)
    parser.add_argument("--gate_temp", type=float, default=0.5)
    parser.add_argument("--ring_highpass", action="store_true", default=True)
    parser.add_argument("--hp_kernel", type=int, default=3)

    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")

    with open(args.exp_yaml, "r", encoding="utf-8") as f:
        exp_configs = yaml.load(f, Loader=yaml.FullLoader)
    if not exp_configs or "model_type" not in exp_configs[0]:
        raise ValueError("exp_yaml must contain at least one config dict with key: model_type")

    model_type = exp_configs[0]["model_type"].upper()
    if model_type == "SD3":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.sd3_ckpt, torch_dtype=torch.float16).to(device)
    elif model_type == "FLUX":
        pipe = FluxPipeline.from_pretrained(args.flux_ckpt, torch_dtype=torch.float16).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    scheduler = pipe.scheduler

    dataset_dict = load_json(args.mapping_json)
    if args.only_ids:
        keep = set(s.strip() for s in args.only_ids.split(",") if s.strip())
        dataset_dict = {k: v for k, v in dataset_dict.items() if k in keep}
    mask_cache = load_json(args.mask_cache_json)

    for exp_dict in exp_configs:
        T_steps = int(exp_dict["T_steps"])
        T_start = int(exp_dict["T_start"])
        src_guidance_scale = float(exp_dict["src_guidance_scale"])
        tar_guidance_scale = float(exp_dict["tar_guidance_scale"])
        mvg = float(exp_dict["mvg"])
        seed = int(exp_dict["seed"])

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for sample_id, data_dict in tqdm(dataset_dict.items(), desc=f"OAVC | {model_type}", ncols=100):
            src_prompt = data_dict["original_prompt"]
            tar_prompt = data_dict["editing_prompt"]
            image_rel = data_dict["image_path"]

            save_dir = os.path.join(args.save, os.path.dirname(image_rel))
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, os.path.basename(image_rel))
            if args.skip_existing and os.path.exists(out_path):
                continue

            # load image (crop to a multiple of 16)
            image = Image.open(os.path.join(args.pie_img_root, image_rel)).convert("RGB")
            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
            orig_w, orig_h = image.size

            # encode
            image_src = pipe.image_processor.preprocess(image).to(device).half()
            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x0_src = x0_src.to(device)
            _, _, H_lat, W_lat = x0_src.shape

            # cached object support
            edit_mask = None
            entry = mask_cache.get(sample_id)
            if entry is not None and entry.get("mask_path"):
                mask_path = entry["mask_path"]
                if args.mask_cache_root is not None:
                    mask_path = os.path.join(args.mask_cache_root, "masks", os.path.basename(mask_path))
                if os.path.exists(mask_path):
                    edit_mask = torch.load(mask_path, map_location="cpu").float().to(device)
            if edit_mask is None and args.fallback_full_mask:
                edit_mask = torch.ones((1, 1, H_lat, W_lat), device=device, dtype=torch.float32)

            common = dict(
                x_src=x0_src,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt,
                negative_prompt="",
                T_steps=T_steps,
                src_guidance_scale=src_guidance_scale,
                tar_guidance_scale=tar_guidance_scale,
                T_start=T_start,
                mvg=mvg,
                edit_mask=edit_mask,
                fg_dilate=args.fg_dilate,
                bd_dilate=args.bd_dilate,
                beta_max=args.beta_max,
                beta_pow=args.beta_pow,
                stage1_lambda_fg=args.stage1_lambda_fg,
                stage1_proj_bg=args.stage1_proj_bg,
                use_projected_injection=args.use_projected_injection,
                use_conf_gate=args.use_conf_gate,
                gate_decay=args.gate_decay,
                gate_k=args.gate_k,
                gate_temp=args.gate_temp,
                ring_highpass=args.ring_highpass,
                hp_kernel=args.hp_kernel,
            )
            if model_type == "SD3":
                x0_tar = OAVC_SD3(pipe, scheduler, stage1_bg_dilate=args.stage1_bg_dilate, **common)
            else:
                x0_tar = OAVC_FLUX(pipe, scheduler, mask_thresh=args.mask_thresh, **common)

            # decode & save
            x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                img = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            pil = pipe.image_processor.postprocess(img)[0]
            pil = pil.resize((orig_w, orig_h), Image.BILINEAR)
            pil.save(out_path)

    print("Done.")


if __name__ == "__main__":
    main()
