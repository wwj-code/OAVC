import torch
import torch.nn.functional as F
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Helper:
    """
    - text prompt segmentation
    - 支持 single-instance / multi-instance union (同一个 prompt 下的多实例)
    - 支持多 prompt union（cat + dog / field + path 这种“不同类别”合并）
    - 输出 latent 尺度 mask: [1,1,H_lat,W_lat] in [0,1]

    关键修正：
    - multi_instance=True 时默认不做 area_min_ratio 过滤（避免镜子里小目标被误杀）
    - multi_instance=True 时建议 rank_mode="score"
    - 支持显式设置 Sam3Processor 的 confidence_threshold
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cuda"),
        load_from_HF: bool = False,
        confidence_threshold: float = 0.05,   # ✅ 对齐你交互端的关键阈值
        resolution: int = 1008,
    ):
        self.device = device
        self.model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
        )
        self.model.to(self.device)
        self.model.eval()

        # ✅ 显式传入 confidence_threshold，避免默认 0.5 造成离线“什么都出不来”
        self.processor = Sam3Processor(
            self.model,
            resolution=resolution,
            device=str(device) if isinstance(device, str) else device.type,
            confidence_threshold=confidence_threshold,
        )

    @staticmethod
    def _compute_rank(areas: torch.Tensor, scores: torch.Tensor, rank_mode: str) -> torch.Tensor:
        rm = (rank_mode or "area_score").lower()
        if rm == "score":
            return scores
        if rm == "area":
            return areas
        return areas * scores  # area_score

    @staticmethod
    def _select_instance_mask(
        masks: torch.Tensor,   # [N,H,W]  (bool or float)
        scores: torch.Tensor,  # [N]      (float)
        score_thresh: float = 0.3,         # ✅ 你离线端的二次过滤阈值（不等于 processor 的 confidence_threshold）
        area_min_ratio: float = 0.01,      # ✅ 只建议用于 single-instance
        multi_instance: bool = False,
        max_instances: int = 1,            # None/-1 => all
        rank_mode: str = None,             # None => auto: single 用 area_score, multi 用 score
        max_union_area_ratio: float = None,
        disable_area_filter_in_multi: bool = True,  # ✅ 关键：multi 时不做 area filter
        score_range=None,
    ):
        if masks.dim() != 3:
            raise ValueError(f"expect masks as [N,H,W], got {masks.shape}")

        N, H, W = masks.shape
        if N == 0:
            return None

        scores = scores.float().view(-1)[:N]
        probs = masks.float()

        # 1) score filter（你的离线阈值）
        # keep = scores > float(score_thresh)
        
        # 1) score filter (支持区间)
        # score_range: (low, high) 其中任何一个为 None 表示不限制
        if score_range is not None:
            low, high = score_range
            keep = torch.ones_like(scores, dtype=torch.bool)
            if low is not None:
                keep = keep & (scores >= float(low))
            if high is not None:
                keep = keep & (scores <= float(high))
        else:
            keep = scores > float(score_thresh)

        if keep.sum() == 0:
            keep = torch.ones_like(scores, dtype=torch.bool)

        if keep.sum() == 0:
            keep = torch.ones_like(scores, dtype=torch.bool)

        probs = probs[keep]
        scores_kept = scores[keep]

        if probs.numel() == 0:
            return None

        # 2) area filter（⚠️ multi_instance 时默认关闭）
        areas = probs.flatten(1).sum(dim=1)

        do_area_filter = True
        if multi_instance and disable_area_filter_in_multi:
            do_area_filter = False

        if do_area_filter:
            area_thresh = float(area_min_ratio) * float(H * W)
            keep2 = areas > area_thresh
            if keep2.sum() == 0:
                keep2 = torch.ones_like(areas, dtype=torch.bool)
            probs = probs[keep2]
            areas = areas[keep2]
            scores_final = scores_kept[keep2]
        else:
            scores_final = scores_kept  # 保留全部（只按 score_thresh 过滤）

        K = probs.shape[0]
        if K == 0:
            return None

        # 3) rank_mode 默认策略
        if rank_mode is None:
            # single-instance：用 area_score 稳一点；multi：用 score 更像交互端（小目标不吃亏）
            rank_mode = "score" if multi_instance else "area_score"

        rank = SAM3Helper._compute_rank(areas, scores_final, rank_mode)
        order = torch.argsort(rank, descending=True)

        # --- single behavior ---
        if not multi_instance:
            best_idx = int(order[0].item())
            best_mask = probs[best_idx]  # [H,W]
            return best_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # --- multi behavior: union topK ---
        if max_instances is None or int(max_instances) < 0:
            topk = K
        else:
            topk = max(1, min(int(max_instances), K))

        sel = order[:topk]
        union = probs[sel].sum(dim=0).clamp(0.0, 1.0)  # [H,W]

        # optional: union 面积上限（建议先别开，容易把镜子猫挤掉）
        if max_union_area_ratio is not None and float(max_union_area_ratio) > 0:
            union_area = float(union.sum().item())
            max_area = float(max_union_area_ratio) * float(H * W)

            if union_area > max_area:
                for kk in range(topk - 1, 0, -1):
                    sel2 = order[:kk]
                    union2 = probs[sel2].sum(dim=0).clamp(0.0, 1.0)
                    if float(union2.sum().item()) <= max_area:
                        union = union2
                        break
                if float(union.sum().item()) > max_area:
                    union = probs[int(order[0].item())]

        return union.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def get_mask_for_latent(
        self,
        pil_image: Image.Image,
        text_prompt: str,
        H_lat: int,
        W_lat: int,
        thresh: float = 0.3,
        dilate_iter: int = 1,
        score_thresh: float = 0.3,
        area_min_ratio: float = 0.01,
        multi_instance: bool = False,
        max_instances: int = 1,
        rank_mode: str = None,
        max_union_area_ratio: float = None,
        disable_area_filter_in_multi: bool = True,
        score_range: str = None,
    ):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(prompt=text_prompt, state=state)

        masks = output.get("masks", None)
        scores = output.get("scores", None)
        if masks is None or scores is None:
            return None

        # unify to [N,H,W]
        if masks.dim() == 4:
            Hm, Wm = masks.shape[-2], masks.shape[-1]
            masks_flat = masks.view(-1, Hm, Wm)
        elif masks.dim() == 3:
            masks_flat = masks
        elif masks.dim() == 2:
            masks_flat = masks.unsqueeze(0)
        else:
            return None

        best_mask = self._select_instance_mask(
            masks_flat,
            scores.view(-1),
            score_thresh=score_thresh,
            area_min_ratio=area_min_ratio,
            multi_instance=multi_instance,
            max_instances=max_instances,
            rank_mode=rank_mode,
            max_union_area_ratio=max_union_area_ratio,
            disable_area_filter_in_multi=disable_area_filter_in_multi,
            score_range=score_range,
        )
        if best_mask is None:
            return None

        mask = best_mask.float()  # [1,1,H,W] at image resolution

        if thresh is not None:
            mask = (mask > float(thresh)).float()

        if dilate_iter and int(dilate_iter) > 0:
            for _ in range(int(dilate_iter)):
                mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

        mask_latent = F.interpolate(
            mask,
            size=(H_lat, W_lat),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        return mask_latent

    @torch.no_grad()
    def get_union_mask_for_latent(
        self,
        pil_image: Image.Image,
        text_prompts,                 # List[str]
        H_lat: int,
        W_lat: int,
        thresh: float = 0.3,
        dilate_iter: int = 1,
        score_thresh: float = 0.3,
        area_min_ratio: float = 0.01,
        multi_instance: bool = True,
        max_instances: int = -1,
        rank_mode: str = None,
        max_union_area_ratio: float = None,
        final_dilate_iter: int = 0,
        disable_area_filter_in_multi: bool = True,
    ):
        """
        多 prompt union：例如 ["cat","dog"] 或 ["field","path"]
        返回：union 后的 [1,1,H_lat,W_lat]
        """
        if not isinstance(text_prompts, (list, tuple)) or len(text_prompts) == 0:
            return None

        union = None
        for p in text_prompts:
            p = (p or "").strip()
            if not p:
                continue
            m = self.get_mask_for_latent(
                pil_image=pil_image,
                text_prompt=p,
                H_lat=H_lat,
                W_lat=W_lat,
                thresh=thresh,
                dilate_iter=dilate_iter,
                score_thresh=score_thresh,
                area_min_ratio=area_min_ratio,
                multi_instance=multi_instance,
                max_instances=max_instances,
                rank_mode=rank_mode,
                max_union_area_ratio=max_union_area_ratio,
                disable_area_filter_in_multi=disable_area_filter_in_multi,
            )
            if m is None:
                continue
            union = m if union is None else (union + m).clamp(0.0, 1.0)

        if union is None:
            return None

        if final_dilate_iter and int(final_dilate_iter) > 0:
            for _ in range(int(final_dilate_iter)):
                union = F.max_pool2d(union, kernel_size=3, stride=1, padding=1)

        return union.clamp(0.0, 1.0)
