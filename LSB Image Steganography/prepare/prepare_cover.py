from PIL import Image
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_IMAGE = Path("C:\\Users\\HP\\Desktop\\Anish 10th books\\LSB Image Steganography\\images\\cover\\satellite.jpg")

OUTPUT_IMAGE = Path("C:\\Users\\HP\\Desktop\\Anish 10th books\\LSB Image Steganography\\images\\cover\\satellite_256.png")

TARGET_SIZE = (256, 256)


# ============================================================
# PREPARE COVER IMAGE
# ============================================================

def prepare_cover():
    """
    Prepare the satellite cover image.

    Steps:
    1. Check that the input image exists.
    2. Open the image.
    3. Convert it to RGB.
    4. Resize it to 256 × 256.
    5. Save it as PNG.
    """

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Input image not found: {INPUT_IMAGE}"
        )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(INPUT_IMAGE)

    print("Original cover image:")
    print("Path :", INPUT_IMAGE)
    print("Size :", image.size)
    print("Mode :", image.mode)

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize(TARGET_SIZE)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save as PNG
    # --------------------------------------------------------

    image.save(
        OUTPUT_IMAGE,
        "PNG"
    )

    print("\nPrepared cover image:")
    print("Size :", image.size)
    print("Mode :", image.mode)
    print("Saved:", OUTPUT_IMAGE)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prepare_cover()