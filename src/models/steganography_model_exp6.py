
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, residual_scale=0.1):
        super().__init__()

        self.residual_scale = residual_scale

        self.initial = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64),
            ResidualBlock(64)
        ])

        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, cover, secret):
        combined = torch.cat([cover, secret], dim=1)
        features = self.initial(combined)

        for block in self.residual_blocks:
            features = block(features)

        residual = self.residual_scale * torch.tanh(
            self.output(features)
        )

        stego = torch.clamp(cover + residual, 0.0, 1.0)
        return stego


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64),
            ResidualBlock(64)
        ])

        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, stego):
        features = self.initial(stego)

        for block in self.residual_blocks:
            features = block(features)

        return self.output(features)


class SteganographyModel(nn.Module):
    def __init__(self, residual_scale=0.1):
        super().__init__()

        self.encoder = Encoder(residual_scale)
        self.decoder = Decoder()

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        recovered_secret = self.decoder(stego)

        return stego, recovered_secret
