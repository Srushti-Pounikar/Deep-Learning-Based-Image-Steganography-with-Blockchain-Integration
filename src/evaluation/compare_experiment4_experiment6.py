
import os
import json
import math

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from src.config import (
    DEVICE,
    SPLIT_FILE,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)

from src.data.steganography_dataset import SteganographyDataset
from src.models.steganography_model import (
    SteganographyModel as Experiment4Model,
)
from src.models.steganography_model_exp6 import (
    SteganographyModel as Experiment6Model,
)


# --------------------------------------------------
# Checkpoints and output
# --------------------------------------------------

EXPERIMENT4_CHECKPOINT = (
    "checkpoints/experiment4_larger_model_best.pth"
)

EXPERIMENT6_CHECKPOINT = (
    "checkpoints/experiment6_residual_best.pth"
)

OUTPUT_PATH = (
    "results/experiment4_vs_experiment6_comparison.json"
)


# --------------------------------------------------
# Metric helpers
# --------------------------------------------------

def calculate_psnr(mse):
    if mse == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse)


def calculate_ssim_batch(original, reconstructed):
    original = original.detach().cpu()
    reconstructed = reconstructed.detach().cpu()

    total_ssim = 0.0

    for index in range(original.shape[0]):
        image1 = (
            original[index]
            .permute(1, 2, 0)
            .numpy()
        )

        image2 = (
            reconstructed[index]
            .permute(1, 2, 0)
            .numpy()
        )

        total_ssim += structural_similarity(
            image1,
            image2,
            channel_axis=2,
            data_range=1.0,
        )

    return total_ssim


# --------------------------------------------------
# Checkpoint loading
# --------------------------------------------------

def load_checkpoint(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print("Loaded:", checkpoint_path)

    return model


# --------------------------------------------------
# Evaluate one model
# --------------------------------------------------

def evaluate_model(model, test_loader, model_name):
    totals = {
        "cover_mse": 0.0,
        "secret_mse": 0.0,
        "cover_ssim": 0.0,
        "secret_ssim": 0.0,
    }

    total_images = 0

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            cover = batch["cover"].to(DEVICE)
            secret = batch["secret"].to(DEVICE)

            stego, recovered = model(cover, secret)

            # Calculate per-image MSE, then sum
            cover_mse = (
                (cover - stego) ** 2
            ).mean(dim=(1, 2, 3))

            secret_mse = (
                (secret - recovered) ** 2
            ).mean(dim=(1, 2, 3))

            totals["cover_mse"] += cover_mse.sum().item()
            totals["secret_mse"] += secret_mse.sum().item()

            totals["cover_ssim"] += calculate_ssim_batch(
                cover, stego
            )

            totals["secret_ssim"] += calculate_ssim_batch(
                secret, recovered
            )

            total_images += cover.shape[0]

    results = {
        key: value / total_images
        for key, value in totals.items()
    }

    results["cover_psnr"] = calculate_psnr(
        results["cover_mse"]
    )

    results["secret_psnr"] = calculate_psnr(
        results["secret_mse"]
    )

    print(f"\n{model_name} results")
    print("-" * 42)
    print(f"Test images    : {total_images}")
    print(f"Cover MSE      : {results['cover_mse']:.6f}")
    print(f"Cover PSNR     : {results['cover_psnr']:.4f} dB")
    print(f"Cover SSIM     : {results['cover_ssim']:.6f}")
    print(f"Secret MSE     : {results['secret_mse']:.6f}")
    print(f"Secret PSNR    : {results['secret_psnr']:.4f} dB")
    print(f"Secret SSIM    : {results['secret_ssim']:.6f}")

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 55)
    print("EXPERIMENT 4 vs EXPERIMENT 6")
    print("=" * 55)
    print("Device:", DEVICE)

    test_dataset = SteganographyDataset(
        split_file=SPLIT_FILE,
        split="test",
        image_size=IMAGE_SIZE,
        seed=RANDOM_SEED,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Test images:", len(test_dataset))
    print("Batch size:", BATCH_SIZE)

    # Each model uses its own architecture.
    model4 = Experiment4Model()
    model4 = load_checkpoint(
        model4,
        EXPERIMENT4_CHECKPOINT,
    )

    model6 = Experiment6Model(residual_scale=0.1)
    model6 = load_checkpoint(
        model6,
        EXPERIMENT6_CHECKPOINT,
    )

    # Both models use the same test dataset and image pairs.
    results4 = evaluate_model(
        model4,
        test_loader,
        "EXPERIMENT 4",
    )

    results6 = evaluate_model(
        model6,
        test_loader,
        "EXPERIMENT 6",
    )

    comparison = {
        "test_images": len(test_dataset),
        "experiment4": results4,
        "experiment6": results6,
        "improvement": {},
    }

    # Positive values mean improvement.
    # Error metrics should decrease; quality metrics should increase.
    lower_is_better = ["cover_mse", "secret_mse"]
    higher_is_better = [
        "cover_psnr",
        "secret_psnr",
        "cover_ssim",
        "secret_ssim",
    ]

    for metric in lower_is_better:
        baseline = results4[metric]
        new_value = results6[metric]

        comparison["improvement"][metric] = (
            (baseline - new_value) / baseline * 100
            if baseline != 0
            else None
        )

    for metric in higher_is_better:
        baseline = results4[metric]
        new_value = results6[metric]

        comparison["improvement"][metric] = (
            (new_value - baseline) / abs(baseline) * 100
            if baseline != 0 and math.isfinite(baseline)
            and math.isfinite(new_value)
            else None
        )

    print("\n" + "=" * 55)
    print("COMPARISON: POSITIVE IMPROVEMENT IS BETTER")
    print("=" * 55)

    for metric, improvement in comparison["improvement"].items():
        if improvement is None:
            print(f"{metric:15s}: not available")
        else:
            print(f"{metric:15s}: {improvement:+.2f}%")

    os.makedirs("results", exist_ok=True)

    def convert_to_json_safe(value):
        if isinstance(value, dict):
            return {
                key: convert_to_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                convert_to_json_safe(item)
                for item in value
            ]

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float) and not math.isfinite(value):
            return None

        return value


    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            convert_to_json_safe(comparison),
            file,
            indent=4,
            allow_nan=False
        )

    print("\nComparison saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
