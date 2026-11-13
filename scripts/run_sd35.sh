#!/usr/bin/env bash
# OAVC on PIE-Bench (sd35). Extra arguments are passed through, e.g. --only_ids 000000000000
python oavc/run_pie_bench.py \
  --device_number 0 \
  --exp_yaml configs/sd35.yaml \
  --save outputs/oavc_sd35 \
  --mapping_json data/PIE/mapping_file.json \
  --pie_img_root data/PIE/annotation_images \
  --mask_cache_json masks/pie_sam3/mask_cache.json \
  --fallback_full_mask "$@"
