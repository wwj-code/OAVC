import os
import json
import math
import csv
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Optional imports
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import lpips  # pip install lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SequenceMetrics:
    sequence: str
    num_frames: int
    num_pairs: int

    bg_psnr: float
    bg_l1: float
    bg_ssim: float

    twarp_bg_l1: float
    twarp_bg_psnr: float
    twarp_bg_ssim: float

    bg_lpips: float = float("nan")
    twarp_bg_lpips: float = float("nan")


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def is_frame_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for x in os.listdir(path):
        if is_image_file(x):
            return True
    return False


def list_frame_map(frame_dir: str) -> Dict[str, str]:
    """
    Returns: {stem: full_path}
    """
    out = {}
    for fn in os.listdir(frame_dir):
        if is_image_file(fn):
            stem = os.path.splitext(fn)[0]
            out[stem] = os.path.join(frame_dir, fn)
    return out


def list_frame_paths_sorted(frame_dir: str) -> List[str]:
    paths = []
    for fn in sorted(os.listdir(frame_dir)):
        if is_image_file(fn):
            paths.append(os.path.join(frame_dir, fn))
    return paths


def discover_sequences(root: str) -> List[str]:
    """
    If root itself is a frame dir -> returns [basename(root)]
    Else returns all subdirs that are frame dirs.
    """
    if is_frame_dir(root):
        return [os.path.basename(os.path.normpath(root))]
    seqs = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if is_frame_dir(p):
            seqs.append(name)
    return seqs


def resolve_seq_dir(root: str, seq_name: str) -> str:
    """
    Supports:
      - root itself is a frame dir
      - root/seq_name is a frame dir
    """
    if is_frame_dir(root):
        return root
    cand = os.path.join(root, seq_name)
    if is_frame_dir(cand):
        return cand
    raise FileNotFoundError(f"Cannot resolve sequence dir for seq='{seq_name}' under root='{root}'")


def parse_seq_map(seq_map_str: str) -> Dict[str, str]:
    """
    Example:
      "car-roundabout=car_to_taxi,paragliding=paraglider"
    Returns:
      {"car-roundabout": "car_to_taxi", "paragliding": "paraglider"}
    """
    out = {}
    if not seq_map_str.strip():
        return out
    items = [x.strip() for x in seq_map_str.split(",") if x.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid seq_map item: {item}, expected src_seq=method_seq")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def resolve_seq_dir_with_alias(root: str, seq_name: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    """
    Supports:
      - root itself is a frame dir
      - root/seq_name is a frame dir
      - root/alias_map[seq_name] is a frame dir
    """
    if is_frame_dir(root):
        return root

    cand = os.path.join(root, seq_name)
    if is_frame_dir(cand):
        return cand

    if alias_map is not None and seq_name in alias_map:
        cand2 = os.path.join(root, alias_map[seq_name])
        if is_frame_dir(cand2):
            return cand2

    raise FileNotFoundError(
        f"Cannot resolve sequence dir for seq='{seq_name}' under root='{root}'. "
        f"Also checked alias={alias_map.get(seq_name) if alias_map else None}"
    )


def read_rgb(path: str, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Returns float32 RGB image in [0,1], shape [H,W,3]
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if target_hw is not None and (arr.shape[0] != target_hw[0] or arr.shape[1] != target_hw[1]):
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32) / 255.0


def read_mask_davis(path: str, obj_id: int = -1, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Reads DAVIS-style annotation mask.
    Supports palette/grayscale or RGB masks.
    Returns binary object mask float32 in {0,1}, shape [H,W].
      obj_id = -1 : merge all non-zero ids
    """
    im = Image.open(path)

    if im.mode in ["P", "L", "I", "I;16"]:
        arr = np.array(im)
        if target_hw is not None and (arr.shape[0] != target_hw[0] or arr.shape[1] != target_hw[1]):
            arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        if obj_id == -1:
            m = (arr > 0).astype(np.float32)
        else:
            m = (arr == obj_id).astype(np.float32)
        return m

    arr = np.array(im.convert("RGB"))
    if target_hw is not None and (arr.shape[0] != target_hw[0] or arr.shape[1] != target_hw[1]):
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if obj_id == -1:
        m = (gray > 0).astype(np.float32)
    else:
        m = (gray == obj_id).astype(np.float32)
    return m


def masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if valid.sum() == 0:
        return float("nan")
    diff = np.abs(a - b).mean(axis=2)
    return float(diff[valid].mean())


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if valid.sum() == 0:
        return float("nan")
    diff = ((a - b) ** 2).mean(axis=2)
    return float(diff[valid].mean())


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    mse = masked_mse(a, b, mask)
    if math.isnan(mse):
        return float("nan")
    if mse < eps:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    Approximate masked SSIM:
    compute SSIM map per channel, then average over mask.
    Requires skimage.
    """
    if not _HAS_SKIMAGE:
        return float("nan")

    valid = mask > 0.5
    if valid.sum() == 0:
        return float("nan")

    vals = []
    for c in range(3):
        score, ssim_map = ssim_fn(
            a[..., c], b[..., c],
            data_range=1.0,
            full=True
        )
        vals.append(float(ssim_map[valid].mean()))
    return float(np.mean(vals))


class LPIPSWrapper:
    def __init__(self, net: str = "alex", device: str = "cuda:0"):
        if not _HAS_TORCH or not _HAS_LPIPS:
            raise RuntimeError("LPIPS requested but torch/lpips is not available.")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def masked_lpips(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        m = mask[..., None].astype(np.float32)
        a_m = a * m
        b_m = b * m

        ta = torch.from_numpy(a_m).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        tb = torch.from_numpy(b_m).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0

        ta = ta.to(self.device)
        tb = tb.to(self.device)

        with torch.no_grad():
            val = self.model(ta, tb).item()
        return float(val)


def compute_backward_flow(next_rgb: np.ndarray, prev_rgb: np.ndarray) -> np.ndarray:
    """
    Returns backward flow from frame_{t+1} -> frame_t
    so that prev can be warped into next coordinates.
    """
    next_gray = cv2.cvtColor((next_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor((prev_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        next_gray, prev_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow.astype(np.float32)


def warp_with_flow(src: np.ndarray, flow_bwd: np.ndarray, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    h, w = flow_bwd.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_bwd[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_bwd[..., 1]).astype(np.float32)
    return cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def intersect_common_stems(src_dir: str, edit_dir: str, mask_dir: str) -> List[str]:
    src_map = list_frame_map(src_dir)
    edt_map = list_frame_map(edit_dir)
    msk_map = list_frame_map(mask_dir)
    stems = sorted(set(src_map.keys()) & set(edt_map.keys()) & set(msk_map.keys()))
    return stems


def evaluate_sequence(
    seq_name: str,
    src_dir: str,
    edit_dir: str,
    mask_dir: str,
    obj_id: int = -1,
    use_lpips: bool = False,
    lpips_model: Optional[LPIPSWrapper] = None,
    match_mode: str = "order",   # "order" or "stem"
) -> SequenceMetrics:
    """
    match_mode:
      - "stem": source/edit/mask by same filename stem
      - "order": source+mask by same stem, edit frames by sorted order
    """

    if match_mode not in ["order", "stem"]:
        raise ValueError(f"Unknown match_mode={match_mode}")

    if match_mode == "stem":
        stems = intersect_common_stems(src_dir, edit_dir, mask_dir)
        if len(stems) < 2:
            raise RuntimeError(f"[{seq_name}] Need at least 2 aligned frames, got {len(stems)}")

        src_map = list_frame_map(src_dir)
        edt_map = list_frame_map(edit_dir)
        msk_map = list_frame_map(mask_dir)

        triplets = [(src_map[s], edt_map[s], msk_map[s]) for s in stems]

    else:
        # source and mask aligned by DAVIS stem;
        # edit frames aligned by sorted order
        src_map = list_frame_map(src_dir)
        msk_map = list_frame_map(mask_dir)
        common_stems = sorted(set(src_map.keys()) & set(msk_map.keys()))
        edt_paths = list_frame_paths_sorted(edit_dir)

        n = min(len(common_stems), len(edt_paths))
        if n < 2:
            raise RuntimeError(
                f"[{seq_name}] Need at least 2 aligned frames, got source/mask={len(common_stems)}, edit={len(edt_paths)}"
            )

        triplets = []
        for i in range(n):
            stem = common_stems[i]
            triplets.append((src_map[stem], edt_paths[i], msk_map[stem]))

    bg_psnr_list, bg_l1_list, bg_ssim_list, bg_lpips_list = [], [], [], []
    tw_l1_list, tw_psnr_list, tw_ssim_list, tw_lpips_list = [], [], [], []

    prev_src = prev_edit = prev_bg = None

    for i, (src_path, edt_path, msk_path) in enumerate(triplets):
        # Read edited first, then resize source/mask to edited size
        edt = read_rgb(edt_path)
        h, w = edt.shape[:2]

        src = read_rgb(src_path, target_hw=(h, w))
        obj_mask = read_mask_davis(msk_path, obj_id=obj_id, target_hw=(h, w))
        bg_mask = 1.0 - obj_mask

        # Per-frame background preservation
        bg_psnr_list.append(masked_psnr(edt, src, bg_mask))
        bg_l1_list.append(masked_l1(edt, src, bg_mask))
        bg_ssim_list.append(masked_ssim(edt, src, bg_mask))
        if use_lpips and lpips_model is not None:
            bg_lpips_list.append(lpips_model.masked_lpips(edt, src, bg_mask))

        if i > 0:
            # Temporal consistency on background
            flow_bwd = compute_backward_flow(src, prev_src)

            warped_prev_edit = warp_with_flow(prev_edit, flow_bwd, interp=cv2.INTER_LINEAR)
            warped_prev_bg = warp_with_flow(prev_bg.astype(np.float32), flow_bwd, interp=cv2.INTER_NEAREST)

            valid = ((warped_prev_bg > 0.5).astype(np.float32) * bg_mask).astype(np.float32)

            tw_l1_list.append(masked_l1(warped_prev_edit, edt, valid))
            tw_psnr_list.append(masked_psnr(warped_prev_edit, edt, valid))
            tw_ssim_list.append(masked_ssim(warped_prev_edit, edt, valid))
            if use_lpips and lpips_model is not None:
                tw_lpips_list.append(lpips_model.masked_lpips(warped_prev_edit, edt, valid))

        prev_src = src
        prev_edit = edt
        prev_bg = bg_mask

    return SequenceMetrics(
        sequence=seq_name,
        num_frames=len(triplets),
        num_pairs=max(0, len(triplets) - 1),

        bg_psnr=safe_mean(bg_psnr_list),
        bg_l1=safe_mean(bg_l1_list),
        bg_ssim=safe_mean(bg_ssim_list),

        twarp_bg_l1=safe_mean(tw_l1_list),
        twarp_bg_psnr=safe_mean(tw_psnr_list),
        twarp_bg_ssim=safe_mean(tw_ssim_list),

        bg_lpips=safe_mean(bg_lpips_list),
        twarp_bg_lpips=safe_mean(tw_lpips_list),
    )


def aggregate_metrics(metrics_list: List[SequenceMetrics]) -> Dict[str, float]:
    keys = [
        "bg_psnr", "bg_l1", "bg_ssim",
        "twarp_bg_l1", "twarp_bg_psnr", "twarp_bg_ssim",
        "bg_lpips", "twarp_bg_lpips",
    ]
    out = {}
    for k in keys:
        vals = [getattr(m, k) for m in metrics_list]
        out[k] = safe_mean(vals)
    out["num_sequences"] = len(metrics_list)
    out["num_frames_total"] = int(sum(m.num_frames for m in metrics_list))
    out["num_pairs_total"] = int(sum(m.num_pairs for m in metrics_list))
    return out


def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pretty_print_summary(summary: Dict[str, Dict[str, float]]):
    print("\n================ Overall Summary ================")
    header = (
        f"{'Method':<20} "
        f"{'BG_PSNR':>10} {'BG_L1':>10} {'BG_SSIM':>10} "
        f"{'TW_BG_L1':>12} {'TW_BG_PSNR':>12} {'TW_BG_SSIM':>12}"
    )
    print(header)
    print("-" * len(header))
    for method, stats in summary.items():
        print(
            f"{method:<20} "
            f"{stats['bg_psnr']:>10.3f} {stats['bg_l1']:>10.5f} {stats['bg_ssim']:>10.4f} "
            f"{stats['twarp_bg_l1']:>12.5f} {stats['twarp_bg_psnr']:>12.3f} {stats['twarp_bg_ssim']:>12.4f}"
        )
    print("=================================================\n")


def parse_method_args(method_args: List[str]) -> Dict[str, str]:
    """
    method_args expected like:
      --method DNAEdit /path/to/root --method OAVC /path/to/root
    argparse gives list like [name1, path1, name2, path2, ...]
    """
    if len(method_args) % 2 != 0:
        raise ValueError("--method must be provided as NAME PATH pairs.")
    out = {}
    for i in range(0, len(method_args), 2):
        name = method_args[i]
        path = method_args[i + 1]
        out[name] = path
    return out


def main():
    parser = argparse.ArgumentParser("DAVIS temporal/background evaluation for edited videos")
    parser.add_argument("--src_root", type=str, required=True,
                        help="DAVIS source frames root (e.g. JPEGImages/480p) or a single sequence frame dir")
    parser.add_argument("--mask_root", type=str, required=True,
                        help="DAVIS mask root (e.g. Annotations/480p) or a single sequence mask dir")

    parser.add_argument(
        "--method",
        nargs="+",
        required=True,
        help="Repeated NAME PATH pairs. Example: --method DNAEdit /path1 OAVC /path2"
    )

    parser.add_argument("--obj_id", type=int, default=-1,
                        help="DAVIS object id. -1 means merge all non-zero labels")
    parser.add_argument("--seq_subset", type=str, default="",
                        help="Comma-separated sequence names. Empty = use all discovered sequences")
    parser.add_argument("--max_seqs", type=int, default=-1,
                        help="If >0, only evaluate the first N sequences after filtering")

    parser.add_argument("--match_mode", type=str, default="order", choices=["order", "stem"],
                        help="Frame matching mode. 'order' is recommended when edit frames were saved as edt_0000.png")
    parser.add_argument("--seq_map", type=str, default="",
                        help="Optional sequence alias mapping, e.g. 'car-roundabout=car_to_taxi,paragliding=paraglider'")

    parser.add_argument("--use_lpips", action="store_true",
                        help="Enable LPIPS (requires torch + lpips)")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--out_json", type=str, default="eval_results.json")
    parser.add_argument("--out_csv", type=str, default="eval_per_sequence.csv")
    args = parser.parse_args()

    methods = parse_method_args(args.method)
    seq_alias_map = parse_seq_map(args.seq_map)

    lpips_model = None
    if args.use_lpips:
        if not (_HAS_TORCH and _HAS_LPIPS):
            raise RuntimeError("Requested --use_lpips, but torch/lpips is not installed.")
        lpips_model = LPIPSWrapper(net=args.lpips_net, device=args.device)

    all_src_seqs = discover_sequences(args.src_root)

    if args.seq_subset.strip():
        keep = [x.strip() for x in args.seq_subset.split(",") if x.strip()]
        seqs = [s for s in all_src_seqs if s in keep]
    else:
        seqs = list(all_src_seqs)

    if args.max_seqs > 0:
        seqs = seqs[:args.max_seqs]

    if len(seqs) == 0:
        raise RuntimeError("No sequences found to evaluate.")

    all_rows = []
    summary = {}

    for method_name, method_root in methods.items():
        print(f"\n[INFO] Evaluating method: {method_name}")
        seq_metrics_all = []

        for seq in seqs:
            try:
                src_dir = resolve_seq_dir(args.src_root, seq)
                mask_dir = resolve_seq_dir(args.mask_root, seq)
                edit_dir = resolve_seq_dir_with_alias(method_root, seq, alias_map=seq_alias_map)

                metrics = evaluate_sequence(
                    seq_name=seq,
                    src_dir=src_dir,
                    edit_dir=edit_dir,
                    mask_dir=mask_dir,
                    obj_id=args.obj_id,
                    use_lpips=args.use_lpips,
                    lpips_model=lpips_model,
                    match_mode=args.match_mode,
                )
                seq_metrics_all.append(metrics)

                row = asdict(metrics)
                row["method"] = method_name
                all_rows.append(row)

                print(
                    f"[{method_name:>10}] {seq:<20} "
                    f"BG_PSNR={metrics.bg_psnr:.3f} "
                    f"BG_L1={metrics.bg_l1:.5f} "
                    f"TW_BG_L1={metrics.twarp_bg_l1:.5f} "
                    f"TW_BG_PSNR={metrics.twarp_bg_psnr:.3f}"
                )

            except Exception as e:
                print(f"[WARN] Skip seq={seq}, method={method_name}: {e}")

        if len(seq_metrics_all) == 0:
            print(f"[WARN] No valid sequences for method={method_name}")
            continue

        summary[method_name] = aggregate_metrics(seq_metrics_all)

    if len(all_rows) > 0:
        write_csv(args.out_csv, all_rows)
        print(f"[INFO] Per-sequence CSV saved to: {args.out_csv}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary JSON saved to: {args.out_json}")

    pretty_print_summary(summary)


if __name__ == "__main__":
    main()