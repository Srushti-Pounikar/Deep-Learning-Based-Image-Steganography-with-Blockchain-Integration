import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder hides the secret image inside the cover image.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
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
            nn.ReLU(),

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
        combined_input = torch.cat(
            [cover, secret],
            dim=1
        )

        stego = self.network(combined_input)

        return stego


class Decoder(nn.Module):
    """
    Decoder extracts the secret image from the stego image.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
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
            nn.ReLU(),

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
        recovered_secret = self.network(stego)

        return recovered_secret


class SteganographyModel(nn.Module):
    """
    Complete image-to-image steganography model.
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, cover, secret):
        stego = self.encoder(
            cover,
            secret
        )

        recovered_secret = self.decoder(
            stego
        )

        return stego, recovered_secret