from pathlib import Path
import random
import pandas as pd

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SteganographyDataset(Dataset):
    """
    Dataset for image-to-image steganography.

    Each sample contains:
        - cover image
        - secret image

    Both images come from the same dataset split.
    """

    def __init__(
        self,
        split_file,
        split,
        image_size=64,
        seed=42
    ):
        self.split_file = Path(split_file)
        self.split = split
        self.image_size = image_size
        self.seed = seed

        # Load split metadata
        self.data = pd.read_csv(self.split_file)

        # Keep only the requested split
        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(
                f"No images found for split: {split}"
            )

        # Store image paths
        self.image_paths = self.data["image_path"].tolist()

        # Transformation applied to every image
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor()
        ])

        # Create deterministic random generator
        self.random_generator = random.Random(seed)

    def __len__(self):
        """
        Return the number of available images.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Return one cover-secret image pair.
        """

        # Select cover image
        cover_path = self.image_paths[index]

        # Select secret image
        secret_index = self.random_generator.randrange(
            len(self.image_paths)
        )

        # Prevent the same image being used as
        # both cover and secret
        while secret_index == index:
            secret_index = self.random_generator.randrange(
                len(self.image_paths)
            )

        secret_path = self.image_paths[secret_index]

        # Load images
        cover_image = Image.open(
            cover_path
        ).convert("RGB")

        secret_image = Image.open(
            secret_path
        ).convert("RGB")

        # Convert images to tensors
        cover_tensor = self.transform(
            cover_image
        )

        secret_tensor = self.transform(
            secret_image
        )

        return {
            "cover": cover_tensor,
            "secret": secret_tensor,
            "cover_path": cover_path,
            "secret_path": secret_path
        }