import argparse
import json
import os
import random
import re
from typing import Dict, Any, List

from tqdm import tqdm

import numpy as np
import torch
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam3"))

from sam3_helper import SAM3Helper


# -------------------------------
# Prompt helpers
# -------------------------------

EDIT_STOPWORDS = {
    "a", "an", "the",
    "in", "on", "at", "with", "of", "to", "for", "and",
}

def extract_object_concept_from_prompt(prompt: str):
    m = re.search(r"(.*)\[(.*?)\](.*)", prompt)
    if not m:
        return None
    left, inside, right = m.groups()
    inside = inside.strip()

    inside_tokens = [w for w in inside.split() if w]
    inside_core = [w for w in inside_tokens if w.lower() not in EDIT_STOPWORDS]
    if inside_core:
        return inside_core[-1]

    right_tokens = [w for w in right.strip().split() if w]
    if len(right_tokens) >= 1:
        return " ".join(right_tokens[:2]) if len(right_tokens) >= 2 else right_tokens[0]

    left_tokens = [w for w in left.strip().split() if w]
    if left_tokens:
        return left_tokens[-1]
    return None

def extract_edit_concept(src_prompt: str, tar_prompt: str) -> str:
    concept = extract_object_concept_from_prompt(src_prompt)
    if concept:
        return concept
    concept = extract_object_concept_from_prompt(tar_prompt)
    if concept:
        return concept
    return src_prompt

def build_sam3_prompt_list(src_prompt: str, tar_prompt: str):
    """
    Candidate text prompts for SAM3, tried in order until one gives a mask.
    """
    src_plain = src_prompt.replace("[", "").replace("]", "").strip()
    tar_plain = tar_prompt.replace("[", "").replace("]", "").strip()

    lower_src = src_plain.lower()
    lower_tar = tar_plain.lower()

    edit_concept = extract_edit_concept(src_prompt, tar_prompt)

    m_src = re.search(r"\[(.*?)\]", src_prompt)
    m_tar = re.search(r"\[(.*?)\]", tar_prompt)
    inside_src = m_src.group(1).strip() if m_src else ""
    inside_tar = m_tar.group(1).strip() if m_tar else ""

    prompts = []

    # --- cake shape special case (do not treat round/square as the object) ---
    is_shape_cake_case = (
        ("[round]" in src_prompt.lower() or "[square]" in src_prompt.lower() or
         "[round]" in tar_prompt.lower() or "[square]" in tar_prompt.lower())
        and (" cake" in lower_src or " cake" in lower_tar)
    )
    if is_shape_cake_case:
        prompts.extend([
            "cake", "a cake",
            "round cake", "a round cake",
            "square cake", "a square cake",
            src_plain, tar_plain,
        ])
    else:
        # --- latte art special case ---
        if (("coffee" in lower_src or "latte" in lower_src or
             "coffee" in lower_tar or "latte" in lower_tar) and edit_concept):
            prompts.append(f"{edit_concept} latte art")
            prompts.append(f"latte art {edit_concept}")

        # (1) bracket inside phrase first
        if inside_src:
            prompts += [inside_src, f"a {inside_src}"]
        if inside_tar:
            prompts += [inside_tar, f"a {inside_tar}"]

        # (2) single concept word
        if edit_concept:
            prompts += [edit_concept, f"a {edit_concept}"]

        # (3) small-object context
        if inside_src:
            prompts += [f"{inside_src} in hand", f"holding {inside_src}"]
        if inside_tar:
            prompts += [f"{inside_tar} in hand", f"holding {inside_tar}"]
        prompts += ["object in hand", "hand"]

        # (4) fallback: whole prompts
        if src_plain:
            prompts.append(src_plain)
        if tar_plain and tar_plain != src_plain:
            prompts.append(tar_plain)

    # uniq
    seen, uniq = set(), []
    for p in prompts:
        p = (p or "").strip()
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


# -------------------------------
# New: bracket list -> multi prompts
# -------------------------------

# -------------------------------
# Utils
# -------------------------------

def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_number", type=int, default=0)

    # PIE
    parser.add_argument("--mapping_json", type=str, required=True,
                        help="PIE mapping json (id -> dict), e.g. mapping_file.json / long_mapping_file.json")
    parser.add_argument("--pie_img_root", type=str, required=True,
                        help="PIE/annotation_images root")

    # SAM3
    parser.add_argument("--sam3_ckpt", type=str, required=True)
    parser.add_argument("--sam3_thresh", type=float, default=0.25)
    parser.add_argument("--sam3_dilate", type=int, default=2)
    parser.add_argument("--sam3_score_thresh", type=float, default=0.3)
    parser.add_argument("--sam3_area_min_ratio", type=float, default=0.01)
    parser.add_argument("--sam3_max_try", type=int, default=20)

    # cache output
    parser.add_argument("--mask_cache_dir", type=str, required=True)

    # skip existing
    parser.add_argument("--skip_existing", action="store_true")

    # fallback full-image mask if fail
    parser.add_argument("--fallback_full_mask", action="store_true",
                        help="if no mask found -> use full-image mask")
    parser.add_argument("--full_mask_value", type=float, default=1.0)

    # latent size rule
    parser.add_argument("--latent_downsample", type=int, default=8,
                        help="latent size = image_size // latent_downsample, default 8 for SD/FLUX VAE")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # optional: only process a comma-separated subset of sample ids (for quick tests)
    parser.add_argument("--only_ids", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # cache dirs
    masks_dir = os.path.join(args.mask_cache_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    cache_json_path = os.path.join(args.mask_cache_dir, "mask_cache.json")

    # load PIE mapping
    with open(args.mapping_json, "r", encoding="utf-8") as f:
        dataset_dict: Dict[str, Dict[str, Any]] = json.load(f)
    if args.only_ids:
        keep = set(s.strip() for s in args.only_ids.split(",") if s.strip())
        dataset_dict = {k: v for k, v in dataset_dict.items() if k in keep}

    # load SAM3 once
    sam3_helper = SAM3Helper(checkpoint_path=args.sam3_ckpt, device=device)

    cache: Dict[str, Any] = {}
    if os.path.exists(cache_json_path):
        with open(cache_json_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    updated = False

    for sample_id, data_dict in tqdm(dataset_dict.items(), desc="SAM3 Precompute", ncols=100):
        image_rel = data_dict["image_path"]
        src_prompt = data_dict["original_prompt"]
        tar_prompt = data_dict["editing_prompt"]

        out_pt = os.path.join(masks_dir, f"{sample_id}.pt")

        if args.skip_existing and os.path.exists(out_pt):
            # still ensure json has entry
            if sample_id not in cache:
                cache[sample_id] = {
                    "image_path": image_rel,
                    "mask_path": out_pt,
                    "note": "pt exists, json filled later",
                }
                updated = True
            continue

        image_path = os.path.join(args.pie_img_root, image_rel)
        if not os.path.exists(image_path):
            cache[sample_id] = {
                "image_path": image_rel,
                "error": f"image not found: {image_path}",
                "mask_path": None,
            }
            updated = True
            continue

        # load image + crop (same as your pipeline)
        image = Image.open(image_path).convert("RGB")
        image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
        H_img, W_img = image.height, image.width

        # latent size
        H_lat = H_img // args.latent_downsample
        W_lat = W_img // args.latent_downsample

        sam3_prompts = build_sam3_prompt_list(src_prompt, tar_prompt)

        edit_mask = None
        chosen_prompt = None

        for p in sam3_prompts[: args.sam3_max_try]:
            m = sam3_helper.get_mask_for_latent(
                pil_image=image,
                text_prompt=p,
                H_lat=H_lat,
                W_lat=W_lat,
                thresh=args.sam3_thresh,
                dilate_iter=args.sam3_dilate,
                score_thresh=args.sam3_score_thresh,
                area_min_ratio=args.sam3_area_min_ratio,
            )
            if m is not None and float(m.max()) > 0.05:
                edit_mask = m
                chosen_prompt = p
                break

        # fallback
        fallback_used = False
        if edit_mask is None and args.fallback_full_mask:
            edit_mask = torch.full((1, 1, H_lat, W_lat), float(args.full_mask_value),
                                   device=device, dtype=torch.float32)
            chosen_prompt = "__FULL_IMAGE_FALLBACK__"
            fallback_used = True

        # save
        if edit_mask is not None:
            # save to CPU for portability
            torch.save(edit_mask.detach().float().cpu(), out_pt)

        entry = {
            "image_path": image_rel,
            "src_prompt": src_prompt,
            "tar_prompt": tar_prompt,
            "mask_path": out_pt if edit_mask is not None else None,
            "chosen_sam3_prompt": chosen_prompt,
            "sam3_prompts_tried": sam3_prompts[: args.sam3_max_try],
            "mask_found": edit_mask is not None and (not fallback_used),
            "fallback_full_mask_used": bool(fallback_used),
            "H_lat": int(H_lat),
            "W_lat": int(W_lat),
        }

        if edit_mask is not None:
            mm = edit_mask.detach().float()
            entry["mask_stats"] = {
                "max": float(mm.max().item()),
                "mean": float(mm.mean().item()),
                "area_ratio_(>0.5)": float((mm > 0.5).float().mean().item()),
            }

        cache[sample_id] = entry
        updated = True

    if updated:
        save_json(cache_json_path, cache)

    print(f"[Done] mask cache saved to: {cache_json_path}")
    print(f"[Done] mask tensors saved to: {masks_dir}")


if __name__ == "__main__":
    main()
