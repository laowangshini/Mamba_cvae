"""
编码器：根据 block_type 动态选择 CNNBlock / VSSBlock_1D / SS2DBlock。
"""
import torch
import torch.nn as nn
from .mamba_blocks import CNNBlock, VSSBlock_1D, SS2DBlock


class MambaEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, block_type="ss2d"):
        super().__init__()

        if block_type == "cnn":
            Block = CNNBlock
        elif block_type == "mamba_1d":
            Block = VSSBlock_1D
        elif block_type == "ss2d":
            Block = SS2DBlock
        else:
            raise ValueError(f"Unsupported block_type: {block_type}")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),
            nn.GELU(),
            nn.BatchNorm2d(64),
        )

        self.layer1 = Block(64)
        self.down1 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.layer2 = Block(128)
        self.down2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)

        self.layer3 = Block(256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)

        x = self.layer2(x)
        x = self.down2(x)

        x = self.layer3(x)

        x = self.global_pool(x).flatten(1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
