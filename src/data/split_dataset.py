from pathlib import Path
import random
import csv


# -----------------------------
# Configuration
# -----------------------------

DATASET_DIR = Path("data/raw/EuroSAT/EuroSAT_RGB")
OUTPUT_DIR = Path("data/processed/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# -----------------------------
# Validate configuration
# -----------------------------

if TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0:
    raise ValueError("Train, validation and test ratios must sum to 1.0.")


# -----------------------------
# Find all images
# -----------------------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )


random.seed(RANDOM_SEED)

image_records = []

class_directories = sorted(
    directory
    for directory in DATASET_DIR.iterdir()
    if directory.is_dir()
)


for class_directory in class_directories:

    class_name = class_directory.name

    image_files = sorted(
        file
        for file in class_directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    random.shuffle(image_files)

    total_images = len(image_files)

    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)

    train_files = image_files[:train_count]

    val_files = image_files[
        train_count:train_count + val_count
    ]

    test_files = image_files[
        train_count + val_count:
    ]

    for image_file in train_files:
        image_records.append(
            [str(image_file), class_name, "train"]
        )

    for image_file in val_files:
        image_records.append(
            [str(image_file), class_name, "validation"]
        )

    for image_file in test_files:
        image_records.append(
            [str(image_file), class_name, "test"]
        )


# -----------------------------
# Create output directory
# -----------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Save split metadata
# -----------------------------

output_file = OUTPUT_DIR / "dataset_split.csv"

with output_file.open(
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["image_path", "class_name", "split"]
    )

    writer.writerows(image_records)


# -----------------------------
# Print summary
# -----------------------------

train_total = sum(
    1 for record in image_records
    if record[2] == "train"
)

validation_total = sum(
    1 for record in image_records
    if record[2] == "validation"
)

test_total = sum(
    1 for record in image_records
    if record[2] == "test"
)


print("\nDataset split completed successfully!")

print(f"Total images      : {len(image_records)}")
print(f"Training images   : {train_total}")
print(f"Validation images : {validation_total}")
print(f"Test images       : {test_total}")

print(f"\nSplit file saved to:")
print(output_file)