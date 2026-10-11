# Image Steganography Project — Experiment Record
## 1. Project overview
This project uses deep learning to hide a secret RGB image inside a cover
RGB image and reconstruct the secret from the generated stego image.
- Dataset: EuroSAT RGB
- Image size: 64 x 64
- Training images: 18,900
- Validation images: 4,050
- Test images: 4,050
- Batch size: 32
- Device used for training: CPU
## 2. Experiment inventory
### Experiment 1 — Baseline MSE
- Checkpoint: `checkpoints/baseline_mse_best_model.pth`
- Training history: `results/baseline_training_history.json`
- Visualizations: `results/baseline_epoch_visualizations/`
- Purpose: Establish a baseline using mean squared error (MSE).
### Experiment 2 — MSE + SSIM
- Checkpoint: `checkpoints/experiment2_ssim_best_model.pth`
- Training history: `results/experiment2_training_history.json`
- Visualizations: `results/experiment2_epoch_visualizations/`
- Purpose: Include structural similarity (SSIM) in the loss design.
### Experiment 3 — Weighted Loss
- Checkpoint: `checkpoints/experiment3_weighted_best_model.pth`
- Additional checkpoint: `checkpoints/exp3_weighted.pth`
- Training history: `results/experiment3_training_history.json`
- Model file: `src/models/steganography_model_exp3.py`
- Purpose: Investigate a weighted loss design.
- Note: The provenance of the additional checkpoint has not yet been verified.
### Experiment 4 — Larger CNN
- Checkpoint: `checkpoints/experiment4_larger_model_best.pth`
- Final checkpoint copy: `checkpoints/final_model_exp4.pth`
- Model architecture: `src/models/steganography_model.py`
- Purpose: Use a larger encoder-decoder CNN with residual blocks.
- Status: Selected as the final model based on the overall balance
  between cover-image quality and secret-image reconstruction.
Recorded test metrics on 4,050 images:
| Metric | Cover to stego | Secret to recovered |
|---|---:|---:|
| MSE | 0.000920 | 0.000891 |
| PSNR | 30.3633 dB | 30.5019 dB |
| SSIM | 0.911380 | 0.983020 |
- Note: A separately identified Experiment 4 training-history JSON file
  was not found in the current results inventory.
### Experiment 5 — Edge-Aware Loss
- Checkpoint: `checkpoints/experiment5_best_model.pth`
- Training history: `results/experiment5_training_history.json`
- Visualizations: `results/experiment5_visualizations/`
- Purpose: Investigate edge-aware loss to preserve image details.
Recorded test metrics on 4,050 images:
| Metric | Cover to stego | Secret to recovered |
|---|---:|---:|
| MSE | 0.000876 | 0.001195 |
| PSNR | 30.5731 dB | 29.2273 dB |
| SSIM | 0.908909 | 0.982983 |
### Experiment 6 — Residual Embedding
- Checkpoint: `checkpoints/experiment6_residual_best.pth`
- Training script: `src/training/train_experiment6.py`
- Model: `src/models/steganography_model_exp6.py`
- Training history: `results/experiment6_training_history.json`
- Visualizations: `results/experiment6_visualizations/`
- Comparison script: `src/evaluation/compare_experiment4_experiment6.py`
- Comparison results: `results/experiment4_vs_experiment6_comparison.json`
- Purpose: Investigate residual embedding initialized from Experiment 4.
Recorded test metrics on 4,050 images:
| Metric | Cover to stego | Secret to recovered |
|---|---:|---:|
| MSE | 0.000712 | 0.001006 |
| PSNR | 31.4733 dB | 29.9746 dB |
| SSIM | 0.884496 | 0.980610 |
## 3. Final model selection
Experiment 4 is the selected final model.
Experiment 6 improved cover MSE and PSNR, but its cover SSIM and all
reported secret-reconstruction metrics were worse than Experiment 4.
Experiment 5 had slightly better cover MSE and PSNR, while Experiment 4
had better cover SSIM and secret-reconstruction metrics.
The selection therefore prioritizes a balanced result rather than claiming
that Experiment 4 is best on every individual metric.
## 4. Final checkpoint verification
The final checkpoint was copied without overwriting the original:
- Original: `checkpoints/experiment4_larger_model_best.pth`
- Final copy: `checkpoints/final_model_exp4.pth`
The SHA256 hashes of both files matched during verification.
The final copy also loaded successfully into `SteganographyModel`, with
373,734 parameters, and was set to evaluation mode.
## 5. Default evaluation and visualization
The following scripts are configured to use the final Experiment 4 checkpoint:
- `src/evaluation/evaluate.py`
- `src/evaluation/visualize_results.py`
The Experiment 4 vs Experiment 6 comparison script intentionally retains
separate checkpoint references for comparing the two models.
## 6. Recordkeeping notes
- `results/training_history.json` duplicates
  `results/experiment5_training_history.json`; it must not be presented
  as Experiment 4 training history.
- Missing or uncertain historical records must not be reconstructed by
  inventing metrics or relabeling another experiment's files.
- Existing experiment checkpoints and results are retained.
- No further experiments are planned; the project experiment set ends at
  Experiment 6.
