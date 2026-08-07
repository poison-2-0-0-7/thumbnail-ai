# Phase 1 Model Benchmark Report

**Run ID:** `{run_id}`
**Timestamp:** {timestamp}
**Device:** {device}
**Fixture count:** {fixture_count}

---

## Detection + Segmentation

| Model | Image | Peak VRAM (GB) | Latency (s) | Mask IoU |
|---|---|---|---|---|
{detection_rows}

### Detection Summary
- **Recommended:** {detection_recommendation}
- **Rationale:** {detection_rationale}

---

## Matting Refinement

| Model | Image | Peak VRAM (GB) | Latency (s) | Matte SAD |
|---|---|---|---|---|
{matting_rows}

### Matting Summary
- **Recommended:** {matting_recommendation}
- **Rationale:** {matting_rationale}

---

## Inpainting (Background Synthesis)

| Model | Image | Peak VRAM (GB) | Latency (s) | PSNR (outside mask) | LPIPS (outside mask) |
|---|---|---|---|---|---|
{inpainting_rows}

### Inpainting Summary
- **Recommended:** {inpainting_recommendation}
- **Rationale:** {inpainting_rationale}

---

## Conclusions

{conclusions}
