import torch
from torch.utils.data import DataLoader

from src.data.steganography_dataset import SteganographyDataset


SPLIT_FILE = "data/processed/splits/dataset_split.csv"

BATCH_SIZE = 32


# Create dataset
train_dataset = SteganographyDataset(
    split_file=SPLIT_FILE,
    split="train",
    image_size=64,
    seed=42
)


# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# Get one batch
batch = next(iter(train_loader))

cover_images = batch["cover"]
secret_images = batch["secret"]


print("Number of training images:", len(train_dataset))
print("Batch size:", BATCH_SIZE)

print("Cover batch shape:", cover_images.shape)
print("Secret batch shape:", secret_images.shape)

print(
    "Cover data type:",
    cover_images.dtype
)

print(
    "Secret data type:",
    secret_images.dtype
)

print(
    "Cover range:",
    cover_images.min().item(),
    "to",
    cover_images.max().item()
)

print(
    "Secret range:",
    secret_images.min().item(),
    "to",
    secret_images.max().item()
)