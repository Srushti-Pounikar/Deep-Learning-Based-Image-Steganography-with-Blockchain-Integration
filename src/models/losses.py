import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# SSIM
# ==================================================

def calculate_ssim(
    image1,
    image2,
    window_size=7
):
    """
    Calculate differentiable SSIM between two images.

    Expected image range: [0, 1]
    """

    padding = window_size // 2

    # Local means
    mean1 = F.avg_pool2d(
        image1,
        kernel_size=window_size,
        stride=1,
        padding=padding
    )

    mean2 = F.avg_pool2d(
        image2,
        kernel_size=window_size,
        stride=1,
        padding=padding
    )

    # Squared means
    mean1_squared = mean1 * mean1
    mean2_squared = mean2 * mean2
    mean12 = mean1 * mean2

    # Local variances
    variance1 = (
        F.avg_pool2d(
            image1 * image1,
            kernel_size=window_size,
            stride=1,
            padding=padding
        )
        - mean1_squared
    )

    variance2 = (
        F.avg_pool2d(
            image2 * image2,
            kernel_size=window_size,
            stride=1,
            padding=padding
        )
        - mean2_squared
    )

    # Local covariance
    covariance = (
        F.avg_pool2d(
            image1 * image2,
            kernel_size=window_size,
            stride=1,
            padding=padding
        )
        - mean12
    )

    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (
        (2 * mean12 + C1)
        * (2 * covariance + C2)
    )

    denominator = (
        (mean1_squared + mean2_squared + C1)
        * (variance1 + variance2 + C2)
    )

    ssim_map = numerator / (
        denominator + 1e-8
    )

    return ssim_map.mean()


# ==================================================
# Edge Loss
# ==================================================

def calculate_edge_loss(
    image1,
    image2
):
    """
    Calculate edge-aware loss using
    horizontal and vertical image gradients.

    This encourages the recovered secret
    to preserve edges and fine details.
    """

    # Horizontal gradients
    image1_x = image1[:, :, :, 1:] - image1[:, :, :, :-1]
    image2_x = image2[:, :, :, 1:] - image2[:, :, :, :-1]

    # Vertical gradients
    image1_y = image1[:, :, 1:, :] - image1[:, :, :-1, :]
    image2_y = image2[:, :, 1:, :] - image2[:, :, :-1, :]

    # Compare horizontal edges
    horizontal_loss = F.l1_loss(
        image1_x,
        image2_x
    )

    # Compare vertical edges
    vertical_loss = F.l1_loss(
        image1_y,
        image2_y
    )

    # Total edge loss
    edge_loss = (
        horizontal_loss
        + vertical_loss
    )

    return edge_loss


# ==================================================
# Steganography Loss
# ==================================================

class SteganographyLoss(nn.Module):

    def __init__(
        self,
        cover_weight=1.5,
        secret_weight=1.0,
        ssim_weight=0.2,
        edge_weight=0.1
    ):
        super().__init__()

        self.cover_weight = cover_weight
        self.secret_weight = secret_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight

        self.mse = nn.MSELoss()

    def forward(
        self,
        cover,
        stego,
        secret,
        recovered_secret
    ):

        # --------------------------------------------------
        # Cover reconstruction loss
        # --------------------------------------------------

        cover_loss = self.mse(
            stego,
            cover
        )

        # --------------------------------------------------
        # Secret MSE loss
        # --------------------------------------------------

        secret_mse = self.mse(
            recovered_secret,
            secret
        )

        # --------------------------------------------------
        # Secret SSIM loss
        # --------------------------------------------------

        secret_ssim = calculate_ssim(
            recovered_secret,
            secret
        )

        secret_ssim_loss = (
            1.0 - secret_ssim
        )

        # --------------------------------------------------
        # Secret edge loss
        # --------------------------------------------------

        secret_edge_loss = calculate_edge_loss(
            recovered_secret,
            secret
        )

        # --------------------------------------------------
        # Combined secret loss
        # --------------------------------------------------

        secret_loss = (
            (1.0 - self.ssim_weight - self.edge_weight)
            * secret_mse
            + self.ssim_weight
            * secret_ssim_loss
            + self.edge_weight
            * secret_edge_loss
        )

        # --------------------------------------------------
        # Total loss
        # --------------------------------------------------

        total_loss = (
            self.cover_weight
            * cover_loss
            + self.secret_weight
            * secret_loss
        )

        return (
            total_loss,
            cover_loss,
            secret_loss
        )