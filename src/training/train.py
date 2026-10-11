import os
import json

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    SPLIT_FILE,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    SECRET_LOSS_WEIGHT,
    NUM_EPOCHS,
    RANDOM_SEED,
    CHECKPOINT_DIR
)

from src.data.steganography_dataset import SteganographyDataset
from src.models.steganography_model import SteganographyModel
from src.models.losses import SteganographyLoss


# --------------------------------------------------
# Output directories
# --------------------------------------------------

RESULTS_DIR = "results"

VISUALIZATION_DIR = os.path.join(
    RESULTS_DIR,
    "epoch_visualizations"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)


# --------------------------------------------------
# Train one epoch
# --------------------------------------------------

def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_cover_loss = 0.0
    total_secret_loss = 0.0

    for batch in data_loader:

        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        stego, recovered_secret = model(
            cover,
            secret
        )

        # Calculate losses
        loss, cover_loss, secret_loss = loss_function(
            cover,
            stego,
            secret,
            recovered_secret
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()
        total_cover_loss += cover_loss.item()
        total_secret_loss += secret_loss.item()

    number_of_batches = len(data_loader)

    average_loss = total_loss / number_of_batches
    average_cover_loss = total_cover_loss / number_of_batches
    average_secret_loss = total_secret_loss / number_of_batches

    return (
        average_loss,
        average_cover_loss,
        average_secret_loss
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(
    model,
    data_loader,
    loss_function,
    device
):

    model.eval()

    total_loss = 0.0
    total_cover_loss = 0.0
    total_secret_loss = 0.0

    with torch.no_grad():

        for batch in data_loader:

            cover = batch["cover"].to(device)
            secret = batch["secret"].to(device)

            stego, recovered_secret = model(
                cover,
                secret
            )

            loss, cover_loss, secret_loss = loss_function(
                cover,
                stego,
                secret,
                recovered_secret
            )

            total_loss += loss.item()
            total_cover_loss += cover_loss.item()
            total_secret_loss += secret_loss.item()

    number_of_batches = len(data_loader)

    average_loss = total_loss / number_of_batches
    average_cover_loss = total_cover_loss / number_of_batches
    average_secret_loss = total_secret_loss / number_of_batches

    return (
        average_loss,
        average_cover_loss,
        average_secret_loss
    )


# --------------------------------------------------
# Save visual comparison
# --------------------------------------------------

def save_epoch_visualization(
    model,
    data_loader,
    device,
    epoch
):

    model.eval()

    with torch.no_grad():

        batch = next(iter(data_loader))

        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)

        stego, recovered_secret = model(
            cover,
            secret
        )

    # Use the first image in the batch
    cover_image = (
        cover[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    secret_image = (
        secret[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    stego_image = (
        stego[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    recovered_image = (
        recovered_secret[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    # Create figure
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

    figure.suptitle(
        f"Epoch {epoch}"
    )

    figure.tight_layout()

    output_path = os.path.join(
        VISUALIZATION_DIR,
        f"epoch_{epoch}.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        "Visualization saved:",
        output_path
    )


# --------------------------------------------------
# Save training history
# --------------------------------------------------

def save_training_history(history):

    with open(
        HISTORY_PATH,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed(RANDOM_SEED)

    print("Using device:", DEVICE)

    # --------------------------------------------------
    # Create directories
    # --------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        VISUALIZATION_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # Training dataset
    # --------------------------------------------------

    train_dataset = SteganographyDataset(
        split_file=SPLIT_FILE,
        split="train",
        image_size=IMAGE_SIZE,
        seed=RANDOM_SEED
    )

    # --------------------------------------------------
    # Validation dataset
    # --------------------------------------------------

    validation_dataset = SteganographyDataset(
        split_file=SPLIT_FILE,
        split="validation",
        image_size=IMAGE_SIZE,
        seed=RANDOM_SEED
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Training images   :", len(train_dataset))
    print("Validation images :", len(validation_dataset))
    print("Batch size        :", BATCH_SIZE)
    print("Epochs            :", NUM_EPOCHS)

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
    # Best model tracking
    # --------------------------------------------------

    best_validation_loss = float("inf")

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    history = {
        "epoch": [],
        "train_loss": [],
        "train_cover_loss": [],
        "train_secret_loss": [],
        "validation_loss": [],
        "validation_cover_loss": [],
        "validation_secret_loss": []
    }

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):

        current_epoch = epoch + 1

        print(
            f"Epoch {current_epoch}/{NUM_EPOCHS}"
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        (
            train_loss,
            train_cover_loss,
            train_secret_loss
        ) = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            DEVICE
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        (
            validation_loss,
            validation_cover_loss,
            validation_secret_loss
        ) = validate(
            model,
            validation_loader,
            loss_function,
            DEVICE
        )

        # --------------------------------------------------
        # Store history
        # --------------------------------------------------

        history["epoch"].append(
            current_epoch
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_cover_loss"].append(
            train_cover_loss
        )

        history["train_secret_loss"].append(
            train_secret_loss
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_cover_loss"].append(
            validation_cover_loss
        )

        history["validation_secret_loss"].append(
            validation_secret_loss
        )

        save_training_history(history)

        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print(
            f"Train Loss       : {train_loss:.6f}"
        )

        print(
            f"Train Cover Loss : {train_cover_loss:.6f}"
        )

        print(
            f"Train Secret Loss: {train_secret_loss:.6f}"
        )

        print(
            f"Val Loss         : {validation_loss:.6f}"
        )

        print(
            f"Val Cover Loss   : {validation_cover_loss:.6f}"
        )

        print(
            f"Val Secret Loss  : {validation_secret_loss:.6f}"
        )

        # --------------------------------------------------
        # Save visualization
        # --------------------------------------------------

        save_epoch_visualization(
            model,
            validation_loader,
            DEVICE,
            current_epoch
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )

            torch.save(
                {
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss
                },
                checkpoint_path
            )

            print(
                "Best model saved to:",
                checkpoint_path
            )

        print()

    print("Training completed!")

    print(
        "\nTraining history saved to:",
        HISTORY_PATH
    )


if __name__ == "__main__":
    main()