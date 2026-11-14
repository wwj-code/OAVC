---
title: OAVC Interactive Editing
emoji: 🖌️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: mit
short_description: Training-free object-aware image editing with SD3.5
---

# OAVC: Object-Aware Velocity Control — interactive demo

Training-free, object-aware image editing on Stable Diffusion 3.5 (medium).
Upload an image, paint the object you want to change, enter a source and a target prompt, and click **Run OAVC**.
Only the painted object is edited; the background stays anchored to the source image.

- Paper: [Object-Aware Background-Controlled Editing via Weighted Velocity Guidance](https://arxiv.org/abs/2609.06288)
- Code: [github.com/wwj-code/OAVC](https://github.com/wwj-code/OAVC)

All sampler and OAVC hyper-parameters default to the paper settings (SD3.5: 40 steps, start step 13, CFG 3.5).
