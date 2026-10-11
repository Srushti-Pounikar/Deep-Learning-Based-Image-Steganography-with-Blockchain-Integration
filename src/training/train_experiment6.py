
import os
import json
import random

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    SPLIT_FILE,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    CHECKPOINT_DIR,
)

from src.data.steganography_dataset import SteganographyDataset
from src.models.steganography_model_exp6 import SteganographyModel
from src.models.losses import SteganographyLoss


# --------------------------------------------------
# Experiment 6 settings
# --------------------------------------------------

EXPERIMENT_NAME = "Experiment 6 - Residual Embedding"

INITIAL_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "experiment4_larger_model_best.pth"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "experiment6_residual_best.pth"
)

RESULTS_DIR = "results"
VISUALIZATION_DIR = os.path.join(
    RESULTS_DIR,
    "experiment6_visualizations"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "experiment6_training_history.json"
)

NUM_EPOCHS = 3

# A smaller learning rate is used when fine-tuning
# an existing checkpoint with a modified encoder.
LEARNING_RATE = 0.0001

RESIDUAL_SCALE = 0.1


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Load Experiment 4 weights
# --------------------------------------------------

def load_initial_checkpoint(model):
    if not os.path.isfile(INITIAL_CHECKPOINT):
        raise FileNotFoundError(
            f"Experiment 4 checkpoint not found: "
            f"{INITIAL_CHECKPOINT}"
        )

    checkpoint = torch.load(
        INITIAL_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    print("Loaded Experiment 4 checkpoint successfully.")
    print("Starting from pretrained model weights.")


# --------------------------------------------------
# Train one epoch
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    loss_function,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    total_cover_loss = 0.0
    total_secret_loss = 0.0

    for batch in loader:
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)

        optimizer.zero_grad(set_to_none=True)

        stego, recovered_secret = model(cover, secret)

        loss, cover_loss, secret_loss = loss_function(
            cover,
            stego,
            secret,
            recovered_secret
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cover_loss += cover_loss.item()
        total_secret_loss += secret_loss.item()

    number_of_batches = len(loader)

    return (
        total_loss / number_of_batches,
        total_cover_loss / number_of_batches,
        total_secret_loss / number_of_batches
    )


# --------------------------------------------------
# Validate
# --------------------------------------------------

def validate(
    model,
    loader,
    loss_function,
    device
):
    model.eval()

    total_loss = 0.0
    total_cover_loss = 0.0
    total_secret_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            cover = batch["cover"].to(device)
            secret = batch["secret"].to(device)

            stego, recovered_secret = model(cover, secret)

            loss, cover_loss, secret_loss = loss_function(
                cover,
                stego,
                secret,
                recovered_secret
            )

            total_loss += loss.item()
            total_cover_loss += cover_loss.item()
            total_secret_loss += secret_loss.item()

    number_of_batches = len(loader)

    return (
        total_loss / number_of_batches,
        total_cover_loss / number_of_batches,
        total_secret_loss / number_of_batches
    )


# --------------------------------------------------
# Save image comparison
# --------------------------------------------------

def save_visualization(model, loader, device, epoch):
    model.eval()

    batch = next(iter(loader))

    cover = batch["cover"].to(device)
    secret = batch["secret"].to(device)

    with torch.no_grad():
        stego, recovered = model(cover, secret)

    images = [
        cover[0],
        secret[0],
        stego[0],
        recovered[0]
    ]

    titles = [
        "Cover",
        "Secret",
        "Stego",
        "Recovered Secret"
    ]

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))

    for axis, image, title in zip(axes, images, titles):
        image = image.detach().cpu().permute(1, 2, 0).numpy()
        axis.imshow(image.clip(0, 1))
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(f"Experiment 6 - Epoch {epoch}")
    figure.tight_layout()

    output_path = os.path.join(
        VISUALIZATION_DIR,
        f"epoch_{epoch}.png"
    )

    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("Visualization saved:", output_path)


# --------------------------------------------------
# Save history
# --------------------------------------------------

def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    set_seed(RANDOM_SEED)

    if not os.path.isfile(SPLIT_FILE):
        raise FileNotFoundError(
            f"Dataset split file not found: {SPLIT_FILE}"
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    print("=" * 55)
    print(EXPERIMENT_NAME)
    print("=" * 55)
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Residual scale:", RESIDUAL_SCALE)
    print("Initial checkpoint:", INITIAL_CHECKPOINT)

    train_dataset = SteganographyDataset(
        split_file=SPLIT_FILE,
        split="train",
        image_size=IMAGE_SIZE,
        seed=RANDOM_SEED
    )

    validation_dataset = SteganographyDataset(
        split_file=SPLIT_FILE,
        split="validation",
        image_size=IMAGE_SIZE,
        seed=RANDOM_SEED
    )

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

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))

    model = SteganographyModel(
        residual_scale=RESIDUAL_SCALE
    )

    # Load weights before moving the model to the device.
    load_initial_checkpoint(model)
    model = model.to(DEVICE)

    loss_function = SteganographyLoss(
        cover_weight=1.5,
        secret_weight=1.0,
        ssim_weight=0.2,
        edge_weight=0.1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float("inf")

    history = {
        "experiment": EXPERIMENT_NAME,
        "initial_checkpoint": INITIAL_CHECKPOINT,
        "residual_scale": RESIDUAL_SCALE,
        "learning_rate": LEARNING_RATE,
        "epoch": [],
        "train_loss": [],
        "train_cover_loss": [],
        "train_secret_loss": [],
        "validation_loss": [],
        "validation_cover_loss": [],
        "validation_secret_loss": []
    }

    print("\nStarting Experiment 6 training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_cover, train_secret = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            DEVICE
        )

        val_loss, val_cover, val_secret = validate(
            model,
            validation_loader,
            loss_function,
            DEVICE
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_cover_loss"].append(train_cover)
        history["train_secret_loss"].append(train_secret)
        history["validation_loss"].append(val_loss)
        history["validation_cover_loss"].append(val_cover)
        history["validation_secret_loss"].append(val_secret)

        save_history(history)

        print(f"Train Loss       : {train_loss:.6f}")
        print(f"Train Cover Loss : {train_cover:.6f}")
        print(f"Train Secret Loss: {train_secret:.6f}")
        print(f"Val Loss         : {val_loss:.6f}")
        print(f"Val Cover Loss   : {val_cover:.6f}")
        print(f"Val Secret Loss  : {val_secret:.6f}")

        save_visualization(
            model,
            validation_loader,
            DEVICE,
            epoch
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss

            torch.save(
                {
                    "experiment": EXPERIMENT_NAME,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss,
                    "validation_cover_loss": val_cover,
                    "validation_secret_loss": val_secret,
                    "residual_scale": RESIDUAL_SCALE
                },
                BEST_CHECKPOINT
            )

            print("Best Experiment 6 checkpoint saved:", BEST_CHECKPOINT)

        print()

    print("Experiment 6 training completed.")
    print("History:", HISTORY_PATH)
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("Visualizations:", VISUALIZATION_DIR)


if __name__ == "__main__":
    main()
