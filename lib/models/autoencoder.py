"""Convolutional autoencoder for anomaly detection via reconstruction error."""

from __future__ import annotations

import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AnomalyAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder.

    The model learns to reconstruct *normal* images. At inference time, anomalous
    regions produce higher reconstruction error which serves as an anomaly score.

    Parameters
    ----------
    in_channels:
        Number of input image channels (default 3 for RGB).
    base_channels:
        Number of channels in the first encoder block. Each subsequent block
        doubles the channel count.
    depth:
        Number of encoder/decoder blocks. Controls the compression ratio.
    latent_dim:
        Dimensionality of the bottleneck fully-connected layer.
    image_size:
        Expected spatial input size (must be divisible by ``2**depth``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        latent_dim: int = 256,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.depth = depth

        # --- Encoder ---
        channels = [in_channels] + [base_channels * (2**i) for i in range(depth)]
        self.encoder = nn.Sequential(
            *[_EncoderBlock(channels[i], channels[i + 1]) for i in range(depth)]
        )

        # Spatial size after encoding
        self._enc_spatial = image_size // (2**depth)
        self._enc_channels = channels[-1]
        flat_dim = self._enc_channels * self._enc_spatial * self._enc_spatial

        # --- Bottleneck ---
        self.fc_encode = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        # --- Decoder ---
        dec_channels = list(reversed(channels))
        self.decoder = nn.Sequential(
            *[_DecoderBlock(dec_channels[i], dec_channels[i + 1]) for i in range(depth)]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_encode(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self._enc_channels, self._enc_spatial, self._enc_spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, latent)``."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
