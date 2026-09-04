from PIL import Image
from pathlib import Path
import struct


# ============================================================
# CONFIGURATION
# ============================================================

COVER_IMAGE = Path("C:\\Users\\HP\\Desktop\\Anish 10th books\\LSB Image Steganography\\images\\cover\\satellite_256.png")

SECRET_IMAGE = Path("C:\\Users\\HP\\Desktop\\Anish 10th books\\LSB Image Steganography\\images\\secret\\secret_80.png")

OUTPUT_IMAGE = Path("C:\\Users\\HP\\Desktop\\Anish 10th books\\LSB Image Steganography\\images\\output\\stego.png")


# ============================================================
# HEADER CONFIGURATION
# ============================================================

# 4-byte identifier
MAGIC = b"STEG"

# Header layout:
#
# MAGIC       = 4 bytes
# WIDTH       = 2 bytes
# HEIGHT      = 2 bytes
# CHANNELS    = 1 byte
# DATA LENGTH = 4 bytes
#
# Total = 13 bytes
HEADER_FORMAT = ">4sHHBI"

HEADER_SIZE = struct.calcsize(
    HEADER_FORMAT
)


# ============================================================
# CREATE HEADER
# ============================================================

def create_header(
    width: int,
    height: int,
    channels: int,
    data_length: int
) -> bytes:
    """
    Create the binary header for the hidden image.
    """

    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        width,
        height,
        channels,
        data_length
    )


# ============================================================
# IMAGE -> BYTES
# ============================================================

def image_to_bytes(
    image: Image.Image
) -> bytes:
    """
    Convert RGB image pixels into raw bytes.
    """

    return image.tobytes()


# ============================================================
# BYTES -> BITS
# ============================================================

def bytes_to_bits(
    data: bytes
) -> list[int]:
    """
    Convert bytes into a list of individual bits.
    """

    bits = []

    for byte in data:

        for bit_position in range(7, -1, -1):

            bit = (
                byte >> bit_position
            ) & 1

            bits.append(bit)

    return bits


# ============================================================
# ENCODER
# ============================================================

def encode_image():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not COVER_IMAGE.exists():
        raise FileNotFoundError(
            f"Cover image not found:\n{COVER_IMAGE}"
        )

    if not SECRET_IMAGE.exists():
        raise FileNotFoundError(
            f"Secret image not found:\n{SECRET_IMAGE}"
        )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    cover = Image.open(
        COVER_IMAGE
    ).convert("RGB")

    secret = Image.open(
        SECRET_IMAGE
    ).convert("RGB")

    print("========================================")
    print("LSB IMAGE STEGANOGRAPHY - ENCODER")
    print("========================================")

    print("\nCover image:")
    print("Size :", cover.size)
    print("Mode :", cover.mode)

    print("\nSecret image:")
    print("Size :", secret.size)
    print("Mode :", secret.mode)

    # --------------------------------------------------------
    # Secret image information
    # --------------------------------------------------------

    secret_width, secret_height = secret.size
    secret_channels = 3

    secret_data = image_to_bytes(
        secret
    )

    secret_data_length = len(
        secret_data
    )

    print("\nSecret image data:")
    print(
        "Size :",
        secret_data_length,
        "bytes"
    )

    print(
        "Bits :",
        secret_data_length * 8
    )

    # --------------------------------------------------------
    # Create header
    # --------------------------------------------------------

    header = create_header(
        secret_width,
        secret_height,
        secret_channels,
        secret_data_length
    )

    print("\nHeader:")
    print("Size:", HEADER_SIZE, "bytes")

    # --------------------------------------------------------
    # Combine header + secret image data
    # --------------------------------------------------------

    payload = (
        header +
        secret_data
    )

    payload_bits = bytes_to_bits(
        payload
    )

    total_payload_bits = len(
        payload_bits
    )

    # --------------------------------------------------------
    # Calculate cover capacity
    # --------------------------------------------------------

    cover_width, cover_height = cover.size

    # One LSB from each RGB channel
    capacity_bits = (
        cover_width
        * cover_height
        * 3
    )

    capacity_bytes = (
        capacity_bits // 8
    )

    print("\nCover capacity:")
    print(
        capacity_bits,
        "bits"
    )
    print(
        capacity_bytes,
        "bytes"
    )

    print("\nPayload:")
    print(
        total_payload_bits,
        "bits"
    )
    print(
        total_payload_bits // 8,
        "bytes"
    )

    # --------------------------------------------------------
    # Capacity check
    # --------------------------------------------------------

    if total_payload_bits > capacity_bits:

        raise ValueError(
            "Secret image is too large for "
            "the selected cover image."
        )

    # --------------------------------------------------------
    # Read cover pixels
    # --------------------------------------------------------

    pixels = list(
        cover.getdata()
    )

    new_pixels = []

    bit_index = 0

    # --------------------------------------------------------
    # Embed payload using LSB
    # --------------------------------------------------------

    for pixel in pixels:

        r, g, b = pixel

        channels = [
            r,
            g,
            b
        ]

        for channel_index in range(3):

            if bit_index < total_payload_bits:

                secret_bit = payload_bits[
                    bit_index
                ]

                # Clear LSB
                channels[channel_index] &= 254

                # Insert secret bit
                channels[channel_index] |= secret_bit

                bit_index += 1

        new_pixels.append(
            tuple(channels)
        )

    # --------------------------------------------------------
    # Create stego image
    # --------------------------------------------------------

    stego = Image.new(
        "RGB",
        cover.size
    )

    stego.putdata(
        new_pixels
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save stego image
    # --------------------------------------------------------

    stego.save(
        OUTPUT_IMAGE,
        "PNG"
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n========================================")
    print("ENCODING SUCCESSFUL")
    print("========================================")

    print(
        "Stego image:",
        OUTPUT_IMAGE
    )

    print(
        "Bits embedded:",
        bit_index
    )

    print(
        "Payload utilization:",
        f"{(total_payload_bits / capacity_bits) * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    encode_image()