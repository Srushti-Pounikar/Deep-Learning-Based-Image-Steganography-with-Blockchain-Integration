import math

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from src.config import (
    DEVICE,
    SPLIT_FILE,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED
)

from src.data.steganography_dataset import SteganographyDataset
from src.models.steganography_model import SteganographyModel
from src.models.losses import SteganographyLoss


CHECKPOINT_PATH = "checkpoints/best_model.pth"


def calculate_psnr(mse):
    """
    Calculate PSNR from mean squared error.

    Images are assumed to be normalized to [0, 1].
    """

    if mse == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse)


def calculate_ssim_batch(
    original_images,
    reconstructed_images
):
    """
    Calculate average SSIM for a batch.
    """

    original_images = original_images.detach().cpu()
    reconstructed_images = reconstructed_images.detach().cpu()

    total_ssim = 0.0

    batch_size = original_images.shape[0]

    for index in range(batch_size):

        original = original_images[index]
        reconstructed = reconstructed_images[index]

        # Convert from CHW to HWC
        original = original.permute(1, 2, 0).numpy()
        reconstructed = reconstructed.permute(1, 2, 0).numpy()

        score = structural_similarity(
            original,
            reconstructed,
            channel_axis=2,
            data_range=1.0
        )

        total_ssim += score

    return total_ssim


def main():

    print("Using device:", DEVICE)

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_dataset = SteganographyDataset(
        split_file=SPLIT_FILE,
        split="test",
        image_size=IMAGE_SIZE,
        seed=RANDOM_SEED
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Test images:", len(test_dataset))
    print("Batch size :", BATCH_SIZE)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SteganographyModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total_cover_mse = 0.0
    total_secret_mse = 0.0

    total_cover_ssim = 0.0
    total_secret_ssim = 0.0

    total_images = 0

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    with torch.no_grad():

        for batch in test_loader:

            cover = batch["cover"].to(DEVICE)
            secret = batch["secret"].to(DEVICE)

            stego, recovered_secret = model(
                cover,
                secret
            )

            # MSE
            cover_mse = torch.mean(
                (cover - stego) ** 2,
                dim=(1, 2, 3)
            )

            secret_mse = torch.mean(
                (secret - recovered_secret) ** 2,
                dim=(1, 2, 3)
            )

            # Accumulate MSE
            total_cover_mse += cover_mse.sum().item()
            total_secret_mse += secret_mse.sum().item()

            # SSIM
            total_cover_ssim += calculate_ssim_batch(
                cover,
                stego
            )

            total_secret_ssim += calculate_ssim_batch(
                secret,
                recovered_secret
            )

            total_images += cover.shape[0]

    # --------------------------------------------------
    # Average metrics
    # --------------------------------------------------

    average_cover_mse = (
        total_cover_mse / total_images
    )

    average_secret_mse = (
        total_secret_mse / total_images
    )

    average_cover_psnr = calculate_psnr(
        average_cover_mse
    )

    average_secret_psnr = calculate_psnr(
        average_secret_mse
    )

    average_cover_ssim = (
        total_cover_ssim / total_images
    )

    average_secret_ssim = (
        total_secret_ssim / total_images
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n======================================")
    print("TEST SET RESULTS")
    print("======================================")

    print("\nCover → Stego")
    print("--------------------------------------")
    print(
        f"MSE  : {average_cover_mse:.6f}"
    )
    print(
        f"PSNR : {average_cover_psnr:.4f} dB"
    )
    print(
        f"SSIM : {average_cover_ssim:.6f}"
    )

    print("\nSecret → Recovered Secret")
    print("--------------------------------------")
    print(
        f"MSE  : {average_secret_mse:.6f}"
    )
    print(
        f"PSNR : {average_secret_psnr:.4f} dB"
    )
    print(
        f"SSIM : {average_secret_ssim:.6f}"
    )

    print("\nTotal test images:", total_images)


if __name__ == "__main__":
    main()