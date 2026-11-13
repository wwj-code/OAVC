#!/usr/bin/env bash
# Usage: bash scripts/evaluate.sh <output_folder> [result.csv]
OUT=${1:?output folder}; CSV=${2:-${OUT%/}_metrics.csv}
python -m evaluation.evaluate \
  --annotation_mapping_file data/PIE/mapping_file.json \
  --src_image_folder data/PIE/annotation_images \
  --tgt_methods "$OUT" \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part \
            clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part \
  --result_path "$CSV" \
  --edit_category_list 0 1 2 3 4 5 6 7 8 9
