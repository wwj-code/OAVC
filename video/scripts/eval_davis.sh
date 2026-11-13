#!/usr/bin/env bash
# Background / temporal metrics on the four DAVIS sequences (run the four run_*.sh scripts first).
python video/eval_davis_temporal_metrics.py \
  --src_root data/DAVIS/JPEGImages/480p \
  --mask_root data/DAVIS/Annotations/480p \
  --method OAVC outputs/video_frames \
  --obj_id -1 \
  --seq_subset car-roundabout,paragliding,bear,hike \
  --out_csv outputs/davis_eval.csv \
  --out_json outputs/davis_eval.json
