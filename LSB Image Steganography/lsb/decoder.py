from PIL import Image
from pathlib import Path
import struct


# ============================================================
# CONFIGURATION
# ============================================================

STEGO_IMAGE = Path("C:\\Users\\HP\\Desktop\\Anish 10th books\\LSB Image Steganography\\images\\output\\stego.png")

OUTPUT_IMAGE = Path("C:\\Users\\HP\\Desktop\\Anish 10th books\\LSB Image Steganography\\images\\output\\decoded_secret.png")


# ============================================================
# HEADER CONFIGURATION
# ============================================================

MAGIC = b"STEG"

HEADER_FORMAT = ">4sHHBI"

HEADER_SIZE = struct.calcsize(
    HEADER_FORMAT
)


# ============================================================
# BITS -> BYTES
# ============================================================

def bits_to_bytes(
    bits: list[int]
) -> bytes:
    """
    Convert a list of bits into bytes.
    """

    if len(bits) % 8 != 0:

        raise ValueError(
            "Number of bits must be divisible by 8."
        )

    output = bytearray()

    for i in range(
        0,
        len(bits),
        8
    ):

        byte_value = 0

        for bit in bits[i:i + 8]:

            byte_value = (
                byte_value << 1
            ) | bit

        output.append(
            byte_value
        )

    return bytes(output)


# ============================================================
# READ HEADER
# ============================================================

def read_header(
    header_data: bytes
):
    """
    Read and validate the hidden-image header.
    """

    if len(header_data) != HEADER_SIZE:

        raise ValueError(
            "Invalid header size."
        )

    (
        magic,
        width,
        height,
        channels,
        data_length
    ) = struct.unpack(
        HEADER_FORMAT,
        header_data
    )

    # Check magic signature
    if magic != MAGIC:

        raise ValueError(
            "Invalid STEG signature. "
            "This image does not appear to "
            "contain our hidden image."
        )

    return (
        width,
        height,
        channels,
        data_length
    )


# ============================================================
# DECODER
# ============================================================

def decode_image():

    # --------------------------------------------------------
    # Check stego image
    # --------------------------------------------------------

    if not STEGO_IMAGE.exists():

        raise FileNotFoundError(
            f"Stego image not found:\n{STEGO_IMAGE}"
        )

    # --------------------------------------------------------
    # Load stego image
    # --------------------------------------------------------

    stego = Image.open(
        STEGO_IMAGE
    ).convert("RGB")

    print("========================================")
    print("LSB IMAGE STEGANOGRAPHY - DECODER")
    print("========================================")

    print("\nStego image:")
    print("Size :", stego.size)
    print("Mode :", stego.mode)

    # --------------------------------------------------------
    # Read LSBs
    # --------------------------------------------------------

    pixels = list(
        stego.getdata()
    )

    extracted_bits = []

    # We need to read enough for the header first.
    header_bits_required = (
        HEADER_SIZE * 8
    )

    for pixel in pixels:

        r, g, b = pixel

        for value in (r, g, b):

            extracted_bits.append(
                value & 1
            )

            if len(extracted_bits) == header_bits_required:
                break

        if len(extracted_bits) == header_bits_required:
            break

    # --------------------------------------------------------
    # Convert header bits to bytes
    # --------------------------------------------------------

    header_data = bits_to_bytes(
        extracted_bits
    )

    # --------------------------------------------------------
    # Decode header
    # --------------------------------------------------------

    (
        secret_width,
        secret_height,
        secret_channels,
        data_length
    ) = read_header(
        header_data
    )

    print("\nHeader information:")
    print(
        "Secret width    :",
        secret_width
    )

    print(
        "Secret height   :",
        secret_height
    )

    print(
        "Channels        :",
        secret_channels
    )

    print(
        "Data length     :",
        data_length,
        "bytes"
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if secret_width <= 0:
        raise ValueError(
            "Invalid secret image width."
        )

    if secret_height <= 0:
        raise ValueError(
            "Invalid secret image height."
        )

    if secret_channels != 3:
        raise ValueError(
            "Only RGB secret images are supported."
        )

    expected_data_length = (
        secret_width
        * secret_height
        * secret_channels
    )

    if data_length != expected_data_length:

        raise ValueError(
            "Secret image header contains "
            "an invalid data length."
        )

    # --------------------------------------------------------
    # Read complete payload
    # --------------------------------------------------------

    total_bits_needed = (
        HEADER_SIZE * 8
        +
        data_length * 8
    )

    all_bits = []

    for pixel in pixels:

        r, g, b = pixel

        for value in (r, g, b):

            all_bits.append(
                value & 1
            )

            if len(all_bits) == total_bits_needed:
                break

        if len(all_bits) == total_bits_needed:
            break

    if len(all_bits) < total_bits_needed:

        raise ValueError(
            "Stego image does not contain "
            "enough embedded data."
        )

    # --------------------------------------------------------
    # Remove header
    # --------------------------------------------------------

    secret_bits = all_bits[
        HEADER_SIZE * 8:
    ]

    # --------------------------------------------------------
    # Convert secret bits to bytes
    # --------------------------------------------------------

    secret_data = bits_to_bytes(
        secret_bits
    )

    # --------------------------------------------------------
    # Reconstruct secret image
    # --------------------------------------------------------

    decoded_secret = Image.frombytes(
        "RGB",
        (
            secret_width,
            secret_height
        ),
        secret_data
    )

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save decoded image
    # --------------------------------------------------------

    decoded_secret.save(
        OUTPUT_IMAGE,
        "PNG"
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n========================================")
    print("DECODING SUCCESSFUL")
    print("========================================")

    print(
        "Decoded secret image:",
        OUTPUT_IMAGE
    )

    print(
        "Decoded size:",
        decoded_secret.size
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    decode_image()