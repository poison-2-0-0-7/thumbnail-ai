# Benchmark Summary Report — Phase 1 Renderer V2

**Run Timestamp:** {timestamp}
**Target Device:** {device}
**Hardware Ceiling:** {vram_ceiling_gb} GB VRAM

---

## 1. Model Alternative Sweeps

| Detection & Segmentation Model | Peak VRAM (GB) | Latency (s) | Mask IoU |
|---|---|---|---|
| GroundingDINO + SAM2.1 | {dino_sam2_vram} | {dino_sam2_latency} | {dino_sam2_iou} |
| SAM3 (Unified) | {sam3_vram} | {sam3_latency} | {sam3_iou} |

| Matting Refinement Model | Peak VRAM (GB) | Latency (s) | Matte SAD |
|---|---|---|---|
| BiRefNet-lite | {birefnet_vram} | {birefnet_latency} | {birefnet_sad} |
| GuidedFilter (Fallback) | {guided_vram} | {guided_latency} | {guided_sad} |

| Background Inpaint Model | Peak VRAM (GB) | Latency (s) | Outside PSNR (dB) |
|---|---|---|---|
| SDXL + BrushNet | {brushnet_vram} | {brushnet_latency} | {brushnet_psnr} |
| Classical Synthetic Inpaint | {classical_vram} | {classical_latency} | {classical_psnr} |

---

## 2. Recommendation & Validation Verdict

- **VRAM Budget Compliant:** {vram_compliant}
- **Selected Primary Stack:** GroundingDINO + SAM2.1 / BiRefNet-lite / Depth-Anything V2 / SDXL+BrushNet
- **Selected Fallback Stack:** SAM3 / GuidedFilter / Synthetic Depth / Classical Inpaint

