#!/usr/bin/env bash
# Compute SAM3 object supports for all PIE-Bench samples.
python oavc/precompute_sam3_masks.py \
  --device_number 0 \
  --mapping_json data/PIE/mapping_file.json \
  --pie_img_root data/PIE/annotation_images \
  --sam3_ckpt checkpoints/sam3/sam3.pt \
  --mask_cache_dir data/PIE/sam3_mask_cache \
  --sam3_max_try 20 \
  --fallback_full_mask
