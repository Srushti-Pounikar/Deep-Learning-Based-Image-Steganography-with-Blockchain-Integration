# LSB Image Steganography

A local Python project that hides a secret image inside a cover image using **LSB (Least Significant Bit) steganography**.

## Project Idea

The project hides a smaller RGB image inside a larger RGB satellite cover image.

```text
Cover Image (256×256 RGB)
          +
Secret Image (80×80 RGB)
          ↓
     LSB Encoder
          ↓
      Stego Image
          ↓
     LSB Decoder
          ↓
  Recovered Secret
```

The current implementation uses only **RGB images**. Multispectral imagery and deep learning are not part of the current version.

## Folder Structure

```text
LSB Image Steganography/
│
├── images/
│   ├── cover/
│   │   ├── satellite.jpg
│   │   └── satellite_256.png
│   │
│   ├── secret/
│   │   ├── secret.jpg
│   │   └── secret_80.png
│   │
│   └── output/
│       ├── stego.png
│       └── decoded_secret.png
│
├── lsb/
│   ├── encoder.py
│   └── decoder.py
│
└── prepare/
    ├── prepare_cover.py
    └── prepare_secret.py
```

## Working Process

### 1. Prepare the Cover

`prepare/prepare_cover.py` loads the original satellite image, converts it to RGB, resizes it to **256×256**, and saves `satellite_256.png`. fileciteturn1file3L20-L29

### 2. Prepare the Secret

`prepare/prepare_secret.py` loads the secret image, converts it to RGB, resizes it to **80×80**, and saves `secret_80.png`. fileciteturn1file2L20-L29

### 3. Encode

`lsb/encoder.py`:

1. Loads the cover and secret images.
2. Converts the secret pixels into bytes and then bits.
3. Creates a small header containing the secret's size, channels, and data length.
4. Combines the header and secret data.
5. Stores the payload in the **least significant bit of each RGB channel**.
6. Saves the result as `images/output/stego.png`. fileciteturn1file1L149-L172 fileciteturn1file1L193-L199 fileciteturn1file1L267-L296

### 4. Decode

`lsb/decoder.py`:

1. Loads `stego.png`.
2. Reads the LSBs from its RGB channels.
3. Reads and validates the `STEG` header.
4. Extracts the secret data.
5. Reconstructs the RGB secret image.
6. Saves it as `images/output/decoded_secret.png`. fileciteturn1file0L146-L180 fileciteturn1file0L290-L312 fileciteturn1file0L324-L329

## Capacity

For a **256×256 RGB** cover using 1 LSB from each RGB channel:

```text
256 × 256 × 3 = 196,608 bits
                 = 24,576 bytes
```

The encoder calculates capacity from the cover dimensions and its 3 RGB channels. fileciteturn1file1L207-L229

The current **80×80 RGB secret** contains:

```text
80 × 80 × 3 = 19,200 bytes
19,200 × 8 = 153,600 bits
```

This fits within the cover capacity, leaving space for the header.

## Header

The project uses a **13-byte header**:

```text
MAGIC       = 4 bytes
WIDTH       = 2 bytes
HEIGHT      = 2 bytes
CHANNELS    = 1 byte
DATA LENGTH = 4 bytes
```

The magic signature is:

```text
STEG
```

The header lets the decoder know how to reconstruct the hidden image. fileciteturn1file1L21-L35

## Requirements

```text
Python
Pillow
```

The project also uses Python's built-in `struct` and `pathlib` modules. fileciteturn1file1L1-L3

## Basic Usage

```text
1. Put the cover image in:
   images/cover/satellite.jpg

2. Put the secret image in:
   images/secret/secret.jpg

3. Prepare the cover.
   → satellite_256.png

4. Prepare the secret.
   → secret_80.png

5. Run the encoder.
   → stego.png

6. Run the decoder.
   → decoded_secret.png
```

## Evaluation

The project can be evaluated using:

- **PSNR** – difference between the original cover and stego image.
- **SSIM** – structural similarity.
- **Secret recovery** – whether the extracted secret matches the original.
- **Payload utilization** – how much of the available capacity is used.

## Current Scope

This is a **local Python implementation** focused on RGB image-in-image LSB steganography.

Currently excluded:

- Multispectral images
- Deep learning
- Networking/deployment
- Blockchain
