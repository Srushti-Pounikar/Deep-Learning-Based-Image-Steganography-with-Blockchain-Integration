import torch
import torch.nn as nn


# ==================================================
# Residual Block
# ==================================================

class ResidualBlock(nn.Module):
    """
    Basic residual block.

    The input is processed by two convolution layers
    and then added back to the original input.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        residual = self.block(x)

        output = x + residual

        output = self.activation(output)

        return output


# ==================================================
# Encoder
# ==================================================

class Encoder(nn.Module):
    """
    Encoder hides the secret image inside
    the cover image.
    """

    def __init__(self):
        super().__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        # Two residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(64),
                ResidualBlock(64)
            ]
        )

        # Convert features back to RGB image
        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, cover, secret):

        # Combine cover and secret
        combined_input = torch.cat(
            [cover, secret],
            dim=1
        )

        # Initial feature extraction
        features = self.initial(
            combined_input
        )

        # Residual processing
        for block in self.residual_blocks:
            features = block(features)

        # Generate stego image
        stego = self.output(
            features
        )

        return stego


# ==================================================
# Decoder
# ==================================================

class Decoder(nn.Module):
    """
    Decoder extracts the secret image
    from the stego image.
    """

    def __init__(self):
        super().__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        # Two residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(64),
                ResidualBlock(64)
            ]
        )

        # Reconstruct secret
        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, stego):

        # Extract features
        features = self.initial(
            stego
        )

        # Residual processing
        for block in self.residual_blocks:
            features = block(features)

        # Recover secret
        recovered_secret = self.output(
            features
        )

        return recovered_secret


# ==================================================
# Complete Steganography Model
# ==================================================

class SteganographyModel(nn.Module):
    """
    Complete image-to-image steganography model.
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, cover, secret):

        # Hide secret inside cover
        stego = self.encoder(
            cover,
            secret
        )

        # Recover secret from stego
        recovered_secret = self.decoder(
            stego
        )

        return (
            stego,
            recovered_secret
        )