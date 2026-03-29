"""
解码器：根据 block_type 动态选择 CNNBlock / VSSBlock_1D / SS2DBlock。
可选：cond_embed_dim 时在 Mamba 块（非 CNN）首层 Norm 后注入 AdaLN（与 1.txt Phase2 一致）。
"""
import torch
import torch.nn as nn
from .mamba_blocks import CNNBlock, VSSBlock_1D, SS2DBlock


def _make_block(Block, dim, cond_embed_dim):
    if Block is CNNBlock:
        return Block(dim)
    return Block(dim, cond_embed_dim=cond_embed_dim)


class MambaDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        out_channels=3,
        block_type="ss2d",
        cond_embed_dim=None,
    ):
        super().__init__()

        if block_type == "cnn":
            Block = CNNBlock
        elif block_type == "mamba_1d":
            Block = VSSBlock_1D
        elif block_type == "ss2d":
            Block = SS2DBlock
        else:
            raise ValueError(f"Unsupported block_type: {block_type}")

        self.map_size = 4
        self.embed_dim = 256
        self.cond_embed_dim = cond_embed_dim

        self.fc_in = nn.Linear(latent_dim, self.embed_dim * self.map_size * self.map_size)

        self.layer1 = _make_block(Block, 256, cond_embed_dim)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.layer2 = _make_block(Block, 128, cond_embed_dim)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.layer3 = _make_block(Block, 64, cond_embed_dim)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )
        self.tanh = nn.Tanh()

    def forward(self, z, cond_emb=None):
        x = self.fc_in(z)
        x = x.view(-1, self.embed_dim, self.map_size, self.map_size)
        x = self.layer1(x, cond_emb)
        x = self.up1(x)

        x = self.layer2(x, cond_emb)
        x = self.up2(x)

        x = self.layer3(x, cond_emb)
        x = self.up3(x)

        x = self.final_conv(x)
        return self.tanh(x)
