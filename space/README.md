---
title: PlanSpot
emoji: 📐
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: One-shot symbol spotting + wall segmentation with DINOv2
---

# PlanSpot: results viewer

Static results page for [PlanSpot](https://github.com/dzakybd/planspot): one-shot symbol spotting and thin-structure (wall) segmentation on CAD drawings with DINOv2.

- Interactive viewer: precomputed detections for example drawings; the threshold slider and method toggle re-score them in the browser.
- Benchmarks: phase 1 (DINOv2 feature matching vs pixel NCC) and phase 2 (U-Net vs frozen DINOv2 vs LoRA vs LoRA + clDice).
- To run the live Gradio demo, clone the GitHub repo and run `python app.py`.

Data: FloorPlanCAD (Fan et al., ICCV 2021), CC BY-NC 4.0, via the Voxel51 export. Non-commercial research demo.
