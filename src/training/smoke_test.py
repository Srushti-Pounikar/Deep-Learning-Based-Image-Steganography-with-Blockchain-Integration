import torch
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    SPLIT_FILE,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    SECRET_LOSS_WEIGHT,
    RANDOM_SEED
)

from src.data.steganography_dataset import SteganographyDataset
from src.models.steganography_model import SteganographyModel
from src.models.losses import SteganographyLoss


def set_seed(seed):
    torch.manual_seed(seed)


def main():

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed(RANDOM_SEED)

    print("Using device:", DEVICE)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = SteganographyDataset(
        split_file=SPLIT_FILE,
        split="train",
        image_size=IMAGE_SIZE,
        seed=RANDOM_SEED
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SteganographyModel().to(DEVICE)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    loss_function = SteganographyLoss(
        secret_weight=SECRET_LOSS_WEIGHT
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------
    # Smoke test
    # --------------------------------------------------

    number_of_batches = 10

    total_loss = 0.0
    total_cover_loss = 0.0
    total_secret_loss = 0.0

    print()
    print("======================================")
    print("EXPERIMENT 4 SMOKE TEST")
    print("======================================")
    print("Dataset images :", len(dataset))
    print("Batch size     :", BATCH_SIZE)
    print("Test batches   :", number_of_batches)
    print("Test images    :", number_of_batches * BATCH_SIZE)
    print()
    print("Starting smoke test...")
    print()

    model.train()

    for batch_number, batch in enumerate(data_loader):

        if batch_number >= number_of_batches:
            break

        cover = batch["cover"].to(DEVICE)
        secret = batch["secret"].to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        stego, recovered_secret = model(
            cover,
            secret
        )

        # Calculate loss
        loss, cover_loss, secret_loss = loss_function(
            cover,
            stego,
            secret,
            recovered_secret
        )

        # Check for invalid values
        if not torch.isfinite(loss):
            print("ERROR: Loss became NaN or infinite.")
            return

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        total_cover_loss += cover_loss.item()
        total_secret_loss += secret_loss.item()

        print(
            f"Batch {batch_number + 1:2d}/10"
            f" | Loss: {loss.item():.6f}"
            f" | Cover: {cover_loss.item():.6f}"
            f" | Secret: {secret_loss.item():.6f}"
        )

    # --------------------------------------------------
    # Average results
    # --------------------------------------------------

    average_loss = total_loss / number_of_batches
    average_cover_loss = total_cover_loss / number_of_batches
    average_secret_loss = total_secret_loss / number_of_batches

    print()
    print("Smoke test completed!")
    print()
    print("Results:")
    print(
        f"Average total loss : {average_loss:.6f}"
    )
    print(
        f"Average cover loss : {average_cover_loss:.6f}"
    )
    print(
        f"Average secret loss: {average_secret_loss:.6f}"
    )


if __name__ == "__main__":
    main()