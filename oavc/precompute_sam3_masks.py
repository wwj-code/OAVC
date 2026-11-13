import argparse
import json
import os
import random
import re
from typing import Dict, Any, List

import numpy as np
import torch
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam3"))

from sam3_helper import SAM3Helper


# -------------------------------
# SAME prompt helpers (保持你现在逻辑不变)
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
    ✅ 完全沿用你现在 run_script_dnaedit.py 的逻辑
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

    # --- cake shape 特判（避免 round/square 被当 object） ---
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
        # --- latte art 特判 ---
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

def split_bracket_items(s: str) -> List[str]:
    """
    支持:
      [field,path] / [field, path] / [cat|dog] / [cat/dog]
    返回: ["field","path"] 这种
    """
    if not s:
        return []
    parts = re.split(r"[,|/]", s.strip())
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


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
    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--pie_img_root", type=str, required=True)

    # choose ONE sample
    parser.add_argument("--only_id", type=str, default=None,
                        help="process only this sample id (key in mapping_json)")
    parser.add_argument("--only_image_rel", type=str, default=None,
                        help="process only the sample whose image_path == this rel path")

    # SAM3
    parser.add_argument("--sam3_ckpt", type=str, required=True)
    parser.add_argument("--sam3_thresh", type=float, default=0.25)
    parser.add_argument("--sam3_dilate", type=int, default=2)
    parser.add_argument("--sam3_score_thresh", type=float, default=0.5)
    parser.add_argument("--sam3_area_min_ratio", type=float, default=0.01)
    parser.add_argument("--sam3_max_try", type=int, default=20)

    # 多实例 union（同一 prompt 内）
    parser.add_argument("--multi_instance", action="store_true", default=True)
    parser.add_argument("--max_instances", type=int, default=1)
    parser.add_argument("--max_union_area_ratio", type=float, default=0.25)

    # cache output
    parser.add_argument("--mask_cache_dir", type=str, required=True)

    # fallback full-image mask if fail
    parser.add_argument("--fallback_full_mask", action="store_true")
    parser.add_argument("--full_mask_value", type=float, default=1.0)

    # latent size rule
    parser.add_argument("--latent_downsample", type=int, default=8)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if (args.only_id is None) and (args.only_image_rel is None):
        raise ValueError("You must provide either --only_id or --only_image_rel to process ONE sample.")

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

    # locate sample
    sample_id = None
    data_dict = None

    if args.only_id is not None:
        sid = str(args.only_id)
        if sid not in dataset_dict:
            raise KeyError(f"--only_id {sid} not found in mapping_json keys.")
        sample_id = sid
        data_dict = dataset_dict[sid]
    else:
        target_rel = args.only_image_rel.strip()
        for sid, dd in dataset_dict.items():
            if str(dd.get("image_path", "")).strip() == target_rel:
                sample_id = str(sid)
                data_dict = dd
                break
        if sample_id is None:
            raise KeyError(f"--only_image_rel '{target_rel}' not found in mapping_json.")

    # load SAM3 once
    sam3_helper = SAM3Helper(checkpoint_path=args.sam3_ckpt, device=device)

    # load existing cache json (optional)
    cache: Dict[str, Any] = {}
    if os.path.exists(cache_json_path):
        with open(cache_json_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # run one
    image_rel = data_dict["image_path"]
    src_prompt = data_dict["original_prompt"]
    tar_prompt = data_dict["editing_prompt"]

    out_pt = os.path.join(masks_dir, f"{sample_id}.pt")

    image_path = os.path.join(args.pie_img_root, image_rel)
    # image_path = image_rel
    if not os.path.exists(image_path):
        entry = {
            "image_path": image_rel,
            "error": f"image not found: {image_path}",
            "mask_path": None,
        }
        cache[sample_id] = entry
        save_json(cache_json_path, cache)
        raise FileNotFoundError(entry["error"])

    # load image + crop (same as your pipeline)
    image = Image.open(image_path).convert("RGB")
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    H_img, W_img = image.height, image.width

    # latent size
    H_lat = H_img // args.latent_downsample
    W_lat = W_img // args.latent_downsample

    sam3_prompts = build_sam3_prompt_list(src_prompt, tar_prompt)

    # -------------------------
    # NEW LOGIC:
    # 1) 如果 bracket 里有多个 items（field,path / cat,dog），就先做多 prompt union
    # 2) 否则 fallback 到原先逐 prompt 试
    # -------------------------
    edit_mask = None
    chosen_prompt = None
    sam3_prompts_tried: List[str] = []

    # extract bracket contents
    m_src = re.search(r"\[(.*?)\]", src_prompt)
    m_tar = re.search(r"\[(.*?)\]", tar_prompt)
    inside_src = m_src.group(1).strip() if m_src else ""
    inside_tar = m_tar.group(1).strip() if m_tar else ""

    items = split_bracket_items(inside_src)
    if len(items) < 2:
        items = split_bracket_items(inside_tar)

    # (1) try union of bracket items first
    if len(items) >= 2:
        sam3_prompts_tried.extend(items)
        edit_mask = sam3_helper.get_union_mask_for_latent(
            pil_image=image,
            text_prompts=items,
            H_lat=H_lat,
            W_lat=W_lat,
            thresh=args.sam3_thresh,
            dilate_iter=args.sam3_dilate,
            score_thresh=args.sam3_score_thresh,
            area_min_ratio=args.sam3_area_min_ratio,
            multi_instance=args.multi_instance,
            max_instances=args.max_instances,
            rank_mode="score",
            max_union_area_ratio=None,     # 多 prompt union 这里建议先别卡太死
            final_dilate_iter=0,
        )
        chosen_prompt = f"__UNION_BRACKET_ITEMS__:{items}"

    # (2) fallback: try your original prompt list
    if edit_mask is None:
        for p in sam3_prompts[: args.sam3_max_try]:
            sam3_prompts_tried.append(p)
            m = sam3_helper.get_mask_for_latent(
                pil_image=image,
                text_prompt=p,
                H_lat=H_lat,
                W_lat=W_lat,
                thresh=args.sam3_thresh,
                dilate_iter=args.sam3_dilate,
                score_thresh=args.sam3_score_thresh,
                area_min_ratio=args.sam3_area_min_ratio,
                multi_instance=args.multi_instance,
                max_instances=args.max_instances,
                rank_mode="score",
                max_union_area_ratio=None,
                # score_range=(0.3, 0.42)
            )
            if m is not None and float(m.max()) > 0.05:
                edit_mask = m
                chosen_prompt = p
                break

    # fallback full image
    fallback_used = False
    if edit_mask is None and args.fallback_full_mask:
        edit_mask = torch.full((1, 1, H_lat, W_lat), float(args.full_mask_value),
                               device=device, dtype=torch.float32)
        chosen_prompt = "__FULL_IMAGE_FALLBACK__"
        fallback_used = True
        sam3_prompts_tried.append("__FULL_IMAGE_FALLBACK__")

    # save
    if edit_mask is not None:
        torch.save(edit_mask.detach().float().cpu(), out_pt)

    entry = {
        "image_path": image_rel,
        "src_prompt": src_prompt,
        "tar_prompt": tar_prompt,
        "mask_path": out_pt if edit_mask is not None else None,
        "chosen_sam3_prompt": chosen_prompt,
        "sam3_prompts_tried": sam3_prompts_tried,
        "mask_found": bool(edit_mask is not None and (not fallback_used)),
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
    save_json(cache_json_path, cache)

    print("[Done] saved pt:", out_pt)
    print("[Done] updated json:", cache_json_path)
    print("[Info] chosen_sam3_prompt:", chosen_prompt)
    print("[Info] mask_found:", entry["mask_found"], "fallback_full_mask_used:", entry["fallback_full_mask_used"])


if __name__ == "__main__":
    main()
