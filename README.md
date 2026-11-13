# OAVC: Object-Aware Background-Controlled Editing via Weighted Velocity Guidance

<p align="center">
  <a href="https://arxiv.org/abs/2609.06288"><img src="https://img.shields.io/badge/arXiv-2609.06288-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

Official implementation of **Object-Aware Velocity Control (OAVC)**, a training-free method for text-guided image and video editing with rectified-flow models (Stable Diffusion 3.5, FLUX.1-dev, Wan2.1).

> **Object-Aware Background-Controlled Editing via Weighted Velocity Guidance**<br>
> Wuji Wang, Yue Wu, Chenhao Yi, Shuhui Wang<br>
> [arXiv:2609.06288](https://arxiv.org/abs/2609.06288)

<p align="center">
  <img src="assets/teaser.jpg" width="100%" alt="OAVC teaser: image and video editing">
</p>

OAVC edits the target object while keeping the rest of the image or video unchanged. Global velocity editors (e.g. DNAEdit) let the background drift, distort reflections and flicker over time; OAVC keeps them stable.

## Method

<p align="center">
  <img src="assets/pipeline.jpg" width="100%" alt="Overview of OAVC">
</p>

Velocity-based editors apply the prompt-induced velocity change to the whole latent, which makes non-target regions drift along the sampling trajectory **(a)**. OAVC controls **where** and **how** the edit is injected:

- **Stage 1** builds a background-anchored reference under the source prompt, relaxing foreground alignment and suppressing background residuals **(b)**.
- **Stage 2** injects the target-prompt velocity change only inside an object support (from SAM3), with an interior weight, a time-varying boundary ring, a projected (constrained) injection, a confidence gate and a boundary high-pass **(c)**.

## Results

**Image editing on PIE-Bench** (SD3.5 and FLUX.1-dev)

<p align="center">
  <img src="assets/pie_comparison.jpg" width="100%" alt="Qualitative comparison on PIE-Bench">
</p>

**Video editing on DAVIS** (Wan2.1-T2V-1.3B)

<p align="center">
  <img src="assets/videos/video_bear.gif" width="80%" alt="DAVIS bear: brown bear to panda"><br>
  <em>"A large brown bear ..." → "A large panda ..."</em>
</p>

<p align="center">
  <img src="assets/videos/video_car.gif" width="80%" alt="DAVIS car-roundabout: dark gray car to yellow taxi"><br>
  <em>"A dark gray car ..." → "A yellow taxi ..."</em>
</p>

<p align="center">
  <img src="assets/videos/video_hike.gif" width="80%" alt="DAVIS hike: man to woman"><br>
  <em>"A man is hiking on the rock." → "A woman is hiking on the rock"</em>
</p>

Full-quality MP4s: [bear](assets/videos/video_bear.mp4) · [car](assets/videos/video_car.mp4) · [hike](assets/videos/video_hike.mp4)

## Installation

```bash
conda create -n oavc python=3.10 -y
conda activate oavc
pip install -r requirements.txt
```

Install [SAM3](https://github.com/facebookresearch/sam3) and download `sam3.pt`.

Expected layout (all commands are run from the repository root):

```
checkpoints/
├── stable-diffusion-3.5-medium/
├── FLUX.1-dev/
├── sam3/sam3.pt
└── Wan2.1-T2V-1.3B/              # video only
data/
├── PIE/                          # PIE-Bench: annotation_images/, mapping_file.json
└── DAVIS/                        # DAVIS 2017 480p: JPEGImages/, Annotations/ (video only)
```

## Image editing on PIE-Bench

**1. Compute SAM3 object supports** (once per image, cached)

```bash
bash scripts/precompute_masks.sh
```

**2. Run OAVC**

```bash
bash scripts/run_sd35.sh     # Stable Diffusion 3.5
bash scripts/run_flux.sh     # FLUX.1-dev
```

All hyper-parameters default to the values in the paper (λ_fg = 0.6, ρ_bg = 0.1, r_fg = 2, r_bd = 4, β_max = 0.45). Sampler settings (steps, start step, CFG, MVG) are in `configs/`. Use `--only_ids 000000000000,000000000001` to run a few samples.

**3. Evaluate**

```bash
bash scripts/evaluate.sh outputs/oavc_sd35
```

This computes Structure Distance, background PSNR / LPIPS / MSE / SSIM and CLIP similarity following the PIE-Bench protocol.

## Video editing on DAVIS (Wan2.1)

```bash
bash video/scripts/run_bear_to_panda.sh
```

Scripts for the four DAVIS sequences are in `video/scripts/`. The object support comes from the DAVIS ground-truth masks. Background metrics (BG-PSNR, BG-L1 and their flow-warped versions) are computed with `video/eval_davis_temporal_metrics.py`.

## Acknowledgements

OAVC is built on [DNAEdit](https://github.com/xiechenxi99/DNAEdit_code). The video code is built on [Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache-2.0, see `video/wan/LICENSE.txt`). Object supports use [SAM3](https://github.com/facebookresearch/sam3), and evaluation follows [PIE-Bench](https://github.com/cure-lab/PnPInversion). We thank the authors for releasing their code.

## License

This project is released under the [MIT License](LICENSE). Code adapted from third-party projects keeps its original terms: the Wan2.1 code in `video/wan/` is under Apache-2.0 (`video/wan/LICENSE.txt`), and parts derived from DNAEdit remain credited to its authors.

## Citation

```bibtex
@article{wang2026oavc,
  title   = {Object-Aware Background-Controlled Editing via Weighted Velocity Guidance},
  author  = {Wang, Wuji and Wu, Yue and Yi, Chenhao and Wang, Shuhui},
  journal = {arXiv preprint arXiv:2609.06288},
  year    = {2026}
}
```
