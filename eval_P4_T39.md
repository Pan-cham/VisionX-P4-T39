1RUA24CSE0001,1RUA24CSE0011,1RUA24CSE0061

# Evaluation - P4_T39 (VisionX) - Edge-Preserving Depth Map Upsampling and Refinement

- **Problem ID:** P4
- **Team:** VisionX (T39)
- **Team members:** AAKANKSH N (1RUA24CSE0001); ADARSH E (1RUA24CSE0011); ANOOP SURAJ K (1RUA24CSE0061)
- **Dataset:** NYU Depth V2 validation split
- **Evaluation samples:** 100
- **Degradation settings:** noise sigma = 0.01 m; hole fraction approximately 0.10
- **Depth unit:** metres
- **PSNR peak:** 10 m
- **Model:** Guided Residual U-Net; 452,401 parameters; checkpoint size 1.85 MB
- **Evaluation runtime:** CPU; process RSS at end of evaluation = 833 MB

## 1. Depth accuracy, image quality and boundary metrics

### 1.1 Upsampling factor: 2×

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Latency (ms/image) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nearest | 0.0622 | 0.0232 | 45.7652 | 0.9862 | 0.1279 | 0.8750 | 0.6280 |
| Bilinear | 0.0549 | 0.0206 | 46.9247 | 0.9896 | 0.1140 | 0.8078 | 0.5416 |
| Bicubic | 0.0528 | 0.0207 | 47.1889 | 0.9900 | 0.1089 | 0.8727 | 0.6038 |
| Classical (JBF+GF) | 0.0526 | 0.0209 | 47.2210 | 0.9903 | 0.1087 | 0.8186 | 21.7394 |
| Deep Learning | 0.0528 | 0.0209 | 47.1805 | 0.9900 | 0.1087 | 0.8739 | 281.8881 |
| Hybrid | 0.0526 | 0.0201 | 47.2711 | 0.9902 | 0.1087 | 0.8739 | 286.0340 |

### 1.2 Upsampling factor: 4×

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Latency (ms/image) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nearest | 0.0998 | 0.0402 | 41.7213 | 0.9684 | 0.2011 | 0.7810 | 0.4018 |
| Bilinear | 0.0876 | 0.0365 | 42.9063 | 0.9777 | 0.1781 | 0.5519 | 0.3150 |
| Bicubic | 0.0845 | 0.0355 | 43.1954 | 0.9782 | 0.1698 | 0.6109 | 0.4110 |
| Classical (JBF+GF) | 0.0845 | 0.0370 | 43.1809 | 0.9794 | 0.1711 | 0.6429 | 63.6681 |
| Deep Learning | 0.0842 | 0.0355 | 43.2219 | 0.9783 | 0.1692 | 0.6127 | 283.0212 |
| Hybrid | 0.0841 | 0.0352 | 43.2544 | 0.9785 | 0.1693 | 0.6127 | 287.2881 |

### 1.3 Upsampling factor: 8×

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Latency (ms/image) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nearest | 0.1528 | 0.0727 | 38.0156 | 0.9377 | 0.2918 | 0.4186 | 0.3559 |
| Bilinear | 0.1348 | 0.0655 | 39.1508 | 0.9610 | 0.2585 | 0.2702 | 0.2585 |
| Bicubic | 0.1300 | 0.0627 | 39.4691 | 0.9609 | 0.2461 | 0.3249 | 0.4550 |
| Classical (JBF+GF) | 0.1284 | 0.0640 | 39.5454 | 0.9653 | 0.2444 | 0.4282 | 129.3352 |
| Deep Learning | 0.1297 | 0.0625 | 39.4920 | 0.9610 | 0.2456 | 0.3256 | 280.1577 |
| Hybrid | 0.1297 | 0.0626 | 39.4997 | 0.9614 | 0.2462 | 0.3256 | 284.3674 |

## 2. Performance across upsampling factors

RMSE in metres; lower is better.

| Method | 2× | 4× | 8× |
|---|---:|---:|---:|
| Nearest | 0.0622 | 0.0998 | 0.1528 |
| Bilinear | 0.0549 | 0.0876 | 0.1348 |
| Bicubic | 0.0528 | 0.0845 | 0.1300 |
| Classical (JBF+GF) | 0.0526 | 0.0845 | 0.1284 |
| Deep Learning | 0.0528 | 0.0842 | 0.1297 |
| Hybrid | 0.0526 | 0.0841 | 0.1297 |

## 3. Efficiency

- Model parameters: 452,401
- Model size: 1.85 MB
- Process RSS at end of evaluation (CPU): 833 MB
- Latency values above are CPU evaluation latency per image at the evaluated resolution.
- Separate Render timings observed during the live demo were approximately 12.4 s for Deep Learning and 12.6 s for Hybrid on one tested image. These are single-request deployment observations, not the CPU benchmark values above.

## 4. Ablation study (4×)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid (full) | 0.0841 | 0.0352 | 43.2544 | 0.9785 | 0.1693 | 0.6127 |
| Hybrid without edge gate | 0.0840 | 0.0351 | 43.2737 | 0.9787 | 0.1692 | 0.6069 |
| Hybrid without post-filter (= DL) | 0.0842 | 0.0355 | 43.2219 | 0.9783 | 0.1692 | 0.6127 |
| Hybrid without RGB guidance | 0.0843 | 0.0352 | 43.2303 | 0.9784 | 0.1697 | 0.6108 |
| Classical (JBF+GF) | 0.0845 | 0.0370 | 43.1809 | 0.9794 | 0.1711 | 0.6429 |
| Bicubic | 0.0845 | 0.0355 | 43.1954 | 0.9782 | 0.1698 | 0.6109 |

**Interpretation:** On this 4× validation evaluation, the full hybrid has the lowest RMSE among the listed methods, but the improvement over bicubic is small (0.0004 m). The ablation results show that individual hybrid components have small and mixed effects; the edge gate does not improve RMSE in this run. These results should not be presented as evidence that every component consistently improves every metric.

## 5. Failure cases

The three worst Hybrid cases at 4×, identified by the evaluation script, are:

| Validation sample | Hybrid RMSE (m) | Evidence |
|---|---:|---|
| val_016 | 0.279 | Failure-case panel should be inspected |
| val_015 | 0.276 | Failure-case panel should be inspected |
| val_036 | 0.267 | Failure-case panel should be inspected |

**Failure-cause analysis is not yet verified.** Inspect the corresponding RGB, ground-truth, Hybrid and error panels in `outputs/failure_cases/` before attributing the errors to thin structures, reflective surfaces, large missing regions or other causes. Do not treat those possible causes as confirmed without inspecting the panels.

## 6. Limitations and interpretation

- The aggregate evaluation uses 100 samples from the NYU Depth V2 validation split; results do not establish performance on every indoor/outdoor domain or on KITTI/Middlebury.
- The hybrid advantage in aggregate RMSE is small, so claims of substantial improvement over interpolation are not supported by these measurements.
- At 8×, Classical (JBF+GF) has lower RMSE than Deep Learning and Hybrid.
- CPU inference latency for the learned methods is materially higher than interpolation and classical filtering.
- The live Render latency is a separate deployment measurement and should not be mixed with the CPU evaluation benchmark.
- Failure-case causes require visual inspection and confirmation.

## 7. Reproducibility details

- Dataset split: NYU Depth V2 validation split prepared for this project.
- Number of validation samples: 100.
- Upsampling factors: 2×, 4× and 8×.
- Noise sigma: 0.01 m.
- Hole fraction: approximately 0.10.
- Checkpoint evaluated: corrected V2 Guided Residual U-Net (`hybrid_unet_v2.pt`, deployed under the expected filename `hybrid_unet.pt`).
- Code and deployment: https://github.com/Pan-cham/VisionX-P4-T39
- Live application: https://visionx-p4-t39.onrender.com
