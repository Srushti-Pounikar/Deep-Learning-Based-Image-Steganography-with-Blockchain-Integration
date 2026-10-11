# Deep Learning-Based Image Steganography with Blockchain Integration



## Project Overview



This project explores deep learning-based image steganography. A secret RGB image is hidden inside a cover RGB image and reconstructed from the generated stego image.



The project uses the EuroSAT RGB dataset and a CNN-based encoder-decoder architecture. Six experiments investigate different model architectures and loss functions.



## Dataset



\- Dataset: EuroSAT RGB

\- Image dimensions: 64 × 64

\- Training images: 18,900

\- Validation images: 4,050

\- Test images: 4,050



## Experiments



Experiment 4 is the selected final model because it provides a balanced result for cover-image quality and secret-image reconstruction.



See \[Experiment Records](docs/EXPERIMENTS.md) for experiment details, available metrics, checkpoints, and recordkeeping notes.



## Final Model



\- Original checkpoint: `checkpoints/experiment4\_larger\_model\_best.pth`

\- Final checkpoint: `checkpoints/final\_model\_exp4.pth`

\- Architecture: `src/models/steganography\_model.py`



The final checkpoint is an exact copy of the original Experiment 4 checkpoint, verified using matching SHA256 hashes.



## Evaluation



Run these commands from the repository root with the virtual environment activated:



```powershell

python -m src.evaluation.evaluate

python -m src.evaluation.visualize\_results

```



Both scripts are configured to load `checkpoints/final\_model\_exp4.pth`.



## Repository Structure



\- `src/data/` — Dataset loading and preparation

\- `src/models/` — Model architectures and loss functions

\- `src/training/` — Training scripts

\- `src/evaluation/` — Evaluation and visualization scripts

\- `checkpoints/` — Saved model checkpoints

\- `results/` — Training histories, metrics, and visualizations

\- `docs/` — Experiment documentation



## Recordkeeping



Existing checkpoints and available results are retained. Missing historical records are identified in the experiment documentation rather than reconstructed or attributed to another experiment.

