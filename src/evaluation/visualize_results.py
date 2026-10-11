import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    SPLIT_FILE,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED
)

from src.data.steganography_dataset import SteganographyDataset
from src.models.steganography_model import SteganographyModel


CHECKPOINT_PATH = "checkpoints/final_model_exp4.pth"

OUTPUT_DIR = "results/visualizations"


def main():

    print("Using device:", DEVICE)

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

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

    # --------------------------------------------------
    # Load model
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

    print("Loaded:", CHECKPOINT_PATH)

    # --------------------------------------------------
    # Get one test batch
    # --------------------------------------------------

    with torch.no_grad():

        batch = next(iter(test_loader))

        cover = batch["cover"].to(DEVICE)
        secret = batch["secret"].to(DEVICE)

        stego, recovered_secret = model(
            cover,
            secret
        )

    # --------------------------------------------------
    # Save first 5 examples
    # --------------------------------------------------

    number_of_examples = 5

    for index in range(number_of_examples):

        cover_image = cover[index].cpu().permute(1, 2, 0).numpy()
        secret_image = secret[index].cpu().permute(1, 2, 0).numpy()
        stego_image = stego[index].cpu().permute(1, 2, 0).numpy()
        recovered_image = recovered_secret[index].cpu().permute(1, 2, 0).numpy()

        # ------------------------------
        # Create figure
        # ------------------------------

        figure, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        axes[0].imshow(cover_image)
        axes[0].set_title("Cover")

        axes[1].imshow(secret_image)
        axes[1].set_title("Secret")

        axes[2].imshow(stego_image)
        axes[2].set_title("Stego")

        axes[3].imshow(recovered_image)
        axes[3].set_title("Recovered Secret")

        for axis in axes:
            axis.axis("off")

        figure.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"comparison_{index + 1}.png"
        )

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(figure)

        print("Saved:", output_path)

    print("\nVisualization completed!")


if __name__ == "__main__":
    main()
