import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import cv2
import numpy as np
warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[args.task], \
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

    # sanity: if using RGB selection, mask_obj_id is ignored
    if args.mask_rgb is not None and args.mask_obj_id is not None:
        logging.info("[Mask] --mask_rgb is set, will ignore --mask_obj_id.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")

    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()),
                        help="The area (width*height) of the generated video. For I2V, aspect ratio follows input image.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames to sample. The number should be 4n+1")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU after each forward to save VRAM.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism size in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism size in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Place T5 on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Use FSDP for DiT.")
    parser.add_argument("--save_file", type=str, default=None, help="File to save generated image/video.")
    parser.add_argument("--prompt", type=str, default=None, help="Source prompt.")
    parser.add_argument("--tgt_prompt", type=str, default=None, help="Target prompt.")

    parser.add_argument("--use_prompt_extend", action="store_true", default=False, help="Use prompt extend.")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--prompt_extend_target_lang", type=str, default="ch", choices=["ch", "en"])

    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--image", type=str, default=None, help="Input image for i2v.")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=1.0)

    parser.add_argument("--video_path", type=str)
    parser.add_argument("--jmp", type=int, default=12)
    parser.add_argument("--tgt_guide_scale", type=float, default=5.0)

    # -------- mask input (DAVIS / external) --------
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path to DAVIS mask folder (png frames) OR a folder of mask frames aligned with video frames.")
    parser.add_argument("--mask_obj_id", type=int, default=1,
                        help="DAVIS instance id to edit for ID-masks. Use -1 to merge all >0.")
    parser.add_argument("--mask_size", type=str, default=None,
                        help="Optional mask resize W*H, default uses --size. e.g. 832*480")

    # -------- NEW: select-by-color for colored masks --------
    parser.add_argument("--mask_rgb", type=str, default=None,
                        help='Select object by RGB color from colored masks, e.g. "255,0,0". If set, ignores --mask_obj_id.')
    parser.add_argument("--mask_rgb_tol", type=float, default=12.0,
                        help="Color distance tolerance for --mask_rgb (in RGB L2 norm).")

    # -------- mask knobs --------
    parser.add_argument("--mask_thresh", type=float, default=0.3)
    parser.add_argument("--fg_dilate", type=int, default=2)
    parser.add_argument("--bd_dilate", type=int, default=4)
    parser.add_argument("--beta_max", type=float, default=0.45)
    parser.add_argument("--beta_pow", type=float, default=2.0)

    # -------- Stage1 knobs --------
    parser.add_argument("--stage1_lambda_fg", type=float, default=0.7)
    parser.add_argument("--stage1_bg_dilate", type=int, default=1)
    parser.add_argument("--stage1_proj_bg", type=float, default=0.1)

    # -------- Stage2 knobs --------
    parser.add_argument("--use_projected_injection", action="store_true", default=True)
    parser.add_argument("--use_conf_gate", action="store_true", default=True)
    parser.add_argument("--gate_decay", type=float, default=0.95)
    parser.add_argument("--gate_k", type=float, default=2.0)
    parser.add_argument("--gate_temp", type=float, default=0.5)
    parser.add_argument("--ring_highpass", action="store_true", default=True)
    parser.add_argument("--hp_kernel", type=int, default=3)

    # -------- other --------
    parser.add_argument("--hard_bg_blend", action="store_true", default=False)

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)s: %(message)s",
                            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def load_frames_path(video_path=None, num_frames=41, target_size=(832,480)):
    frames = []
    if video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = (frame.astype(np.float32)-127.5) / 127.5
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        cap.release()
    else:
        frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for i in range(min(num_frames, len(frame_files))):
            frame_path = os.path.join(video_path, frame_files[i])
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Cannot read image: {frame_path}")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frame = (frame.astype(np.float32)-127.5) / 127.5
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

    if len(frames) == 0:
        raise ValueError("No frames loaded.")
    frames_tensor = torch.tensor(np.array(frames)).float()  # [T,C,H,W]
    frames_tensor = frames_tensor.permute(1,0,2,3).unsqueeze(0)  # [1,C,T,H,W]
    return frames_tensor


def _parse_rgb(rgb_str: str):
    # "255,0,0" -> (255,0,0)
    parts = [p.strip() for p in rgb_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--mask_rgb must be like 'R,G,B', got: {rgb_str}")
    r, g, b = [int(x) for x in parts]
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return (r, g, b)


def _read_mask_as_id_or_rgb(path: str):
    """
    Returns:
      arr: np.ndarray
        - if ID mask (palette/gray): shape [H,W], dtype uint8/uint16
        - if RGB mask: shape [H,W,3], dtype uint8 (RGB)
      mode: "id" or "rgb"
    """
    ext = os.path.splitext(path)[1].lower()
    # Prefer PIL for png (palette-safe)
    if ext == ".png":
        im = Image.open(path)
        # Palette/gray => treat as ID map
        if im.mode in ["P", "L", "I;16", "I"]:
            arr = np.array(im)
            return arr, "id"
        # RGB/RGBA => treat as RGB
        if im.mode in ["RGB", "RGBA"]:
            im = im.convert("RGB")
            arr = np.array(im)
            return arr, "rgb"
        # fallback
        im = im.convert("RGB")
        arr = np.array(im)
        return arr, "rgb"
    else:
        # jpg etc.
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise ValueError(f"Cannot read mask: {path}")
        if m.ndim == 2:
            return m, "id"
        # cv2 gives BGR, convert to RGB for consistent --mask_rgb
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
            return m, "rgb"
        raise ValueError(f"Unsupported mask shape: {m.shape}")


def load_mask_frames_path(mask_path: str,
                          num_frames: int,
                          target_size=(832, 480),
                          obj_id: int = 1,
                          mask_rgb: str = None,
                          mask_rgb_tol: float = 12.0):
    """
    mask_path: folder containing per-frame masks aligned with video frames order.
    Supports:
      (A) DAVIS palette/ID masks: PNG mode P/L where pixel value = instance id
      (B) colored visualization masks: RGB images where different objects are different colors
    Returns:
      mask tensor [1,1,T,H,W] in {0,1}
    """
    if mask_path is None:
        return None
    assert os.path.isdir(mask_path), f"mask_path must be a folder, got {mask_path}"

    files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(files) == 0:
        raise ValueError(f"No mask frames found in: {mask_path}")

    rgb_target = None
    if mask_rgb is not None:
        rgb_target = np.array(_parse_rgb(mask_rgb), dtype=np.float32)  # [3]

    frames = []
    for i in range(min(num_frames, len(files))):
        p = os.path.join(mask_path, files[i])

        arr, mode = _read_mask_as_id_or_rgb(p)

        # resize
        if mode == "id":
            # nearest for labels
            arr_rs = cv2.resize(arr, target_size, interpolation=cv2.INTER_NEAREST)
            if rgb_target is not None:
                # user asked rgb but file is id => cannot match by rgb
                # fall back to obj_id behavior
                if obj_id == -1:
                    m01 = (arr_rs > 0).astype(np.float32)
                else:
                    m01 = (arr_rs == obj_id).astype(np.float32)
            else:
                if obj_id == -1:
                    m01 = (arr_rs > 0).astype(np.float32)
                else:
                    m01 = (arr_rs == obj_id).astype(np.float32)

        elif mode == "rgb":
            arr_rs = cv2.resize(arr, target_size, interpolation=cv2.INTER_NEAREST)  # keep hard edges
            if rgb_target is not None:
                # select by color distance
                diff = arr_rs.astype(np.float32) - rgb_target[None, None, :]
                dist = np.sqrt(np.sum(diff * diff, axis=-1))  # [H,W]
                m01 = (dist <= float(mask_rgb_tol)).astype(np.float32)
            else:
                # No rgb specified: try to fall back to grayscale threshold (NOT recommended)
                # This is kept for backward compatibility.
                gray = cv2.cvtColor(arr_rs, cv2.COLOR_RGB2GRAY)
                if obj_id == -1:
                    m01 = (gray > 0).astype(np.float32)
                else:
                    # NOTE: (gray==obj_id) is generally meaningless for colored masks
                    m01 = (gray == obj_id).astype(np.float32)
        else:
            raise ValueError(f"Unknown mask mode: {mode}")

        frames.append(m01)

    mask = np.stack(frames, axis=0)  # [T,H,W]
    mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # [1,1,T,H,W]
    return mask


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    video_input = load_frames_path(args.video_path, num_frames=args.frame_num)

    # choose size for mask resize
    if args.mask_size is not None:
        mw, mh = map(int, args.mask_size.split("*"))
        mask_target_size = (mw, mh)
    else:
        vw, vh = map(int, args.size.split("*"))
        mask_target_size = (vw, vh)

    mask_video = None
    if args.mask_path is not None:
        mask_video = load_mask_frames_path(
            args.mask_path,
            num_frames=args.frame_num,
            target_size=mask_target_size,
            obj_id=args.mask_obj_id,
            mask_rgb=args.mask_rgb,
            mask_rgb_tol=args.mask_rgb_tol,
        )

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp/dit_fsdp not supported in non-distributed."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "context parallel not supported in non-distributed."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import (initialize_model_parallel, init_distributed_environment)
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(),
                                  ring_degree=args.ring_size,
                                  ulysses_degree=args.ulysses_size)

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task, device=rank)
        else:
            raise NotImplementedError(f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ---- T2V/T2I ----
    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")

        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed)
                input_prompt = args.prompt if not prompt_output.status else prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        import psutil, time
        process = psutil.Process(os.getpid())
        start_time = time.time()

        video = wan_t2v.edit_DNAEdit_weighted(
            video_input,
            args.prompt,
            args.tgt_prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            jmp=args.jmp,
            tgt_guide_scale=args.tgt_guide_scale,

            # -------- pass mask --------
            edit_mask_video=mask_video,

            # -------- pass knobs --------
            mask_thresh=args.mask_thresh,
            fg_dilate=args.fg_dilate,
            bd_dilate=args.bd_dilate,
            beta_max=args.beta_max,
            beta_pow=args.beta_pow,

            stage1_lambda_fg=args.stage1_lambda_fg,
            stage1_bg_dilate=args.stage1_bg_dilate,
            stage1_proj_bg=args.stage1_proj_bg,

            use_projected_injection=args.use_projected_injection,
            use_conf_gate=args.use_conf_gate,
            gate_decay=args.gate_decay,
            gate_k=args.gate_k,
            gate_temp=args.gate_temp,
            ring_highpass=args.ring_highpass,
            hp_kernel=args.hp_kernel,

            hard_bg_blend=args.hard_bg_blend,
        )

        running_time = time.time() - start_time
        max_cpu_memory = process.memory_info().rss / (1024 ** 2)
        peak_gpu_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 2)
        print(f"Max CPU Memory Usage: {max_cpu_memory:.2f} MB")
        print(f"Peak GPU Memory Usage: {peak_gpu_memory:.2f} MB")
        print(f"Running Time: {running_time:.2f} seconds")

    # ---- I2V ----
    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, image=img, seed=args.base_seed)
                input_prompt = args.prompt if not prompt_output.status else prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    # ---- save ----
    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}{suffix}"

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(tensor=video.squeeze(1)[None], save_file=args.save_file, nrow=1, normalize=True, value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(tensor=video[None], save_file=args.save_file, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
