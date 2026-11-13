#!/usr/bin/env bash
# OAVC on DAVIS "bear" (Wan2.1-T2V-1.3B, first 41 frames, GT instance mask as support)
mkdir -p outputs/video
CUDA_VISIBLE_DEVICES=0 python video/edit.py --task t2v-1.3B \
  --size 832*480 --frame_num 41 --ckpt_dir checkpoints/Wan2.1-T2V-1.3B \
  --prompt "A large brown bear is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the bear's deliberate movments." \
  --tgt_prompt "A large panda is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the panda's deliberate movments." \
  --video_path "data/DAVIS/JPEGImages/480p/bear" \
  --mask_path "data/DAVIS/Annotations/480p/bear" --mask_obj_id -1 \
  --sample_steps 50 --sample_shift 5.0 \
  --sample_guide_scale 1.0 --tgt_guide_scale 3.5 --jmp 12 \
  --fg_dilate 1 --bd_dilate 4 --beta_max 0.45 --beta_pow 2.0 \
  --stage1_lambda_fg 0.1 --stage1_bg_dilate 1 --stage1_proj_bg 0.1 \
  --use_projected_injection --use_conf_gate \
  --gate_decay 0.95 --gate_k 2.0 --gate_temp 0.5 \
  --ring_highpass --hp_kernel 3 \
  --hard_bg_blend \
  --save_file outputs/video/bear.mp4
python video/extract_frames.py --video outputs/video/bear.mp4 --out_dir outputs/video_frames/bear
