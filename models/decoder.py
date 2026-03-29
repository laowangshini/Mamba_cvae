"""
解码器（Phase2 / 1.txt）：根据 block_type 选择 CNN / 1D Mamba / SS2D；
当 cond_dim>0 时，在解码器内持有 cond_embedding，并将连续条件 c_emb 传入各 Mamba 块（AdaLN）。
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
        cond_dim=0,
        cond_embed_dim=256,
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
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_embed_dim

        # 1.txt：将离散 CelebA 属性映射到连续空间，供各「条件化」Mamba 块使用
        if cond_dim and cond_dim > 0:
            self.cond_embedding = nn.Sequential(
                nn.Linear(cond_dim, cond_embed_dim),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim),
            )
            ced = cond_embed_dim
        else:
            self.cond_embedding = None
            ced = None

        self.fc_in = nn.Linear(latent_dim, self.embed_dim * self.map_size * self.map_size)

        self.layer1 = _make_block(Block, 256, ced)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.layer2 = _make_block(Block, 128, ced)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.layer3 = _make_block(Block, 64, ced)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )
        self.tanh = nn.Tanh()

    def forward(self, z, cond=None):
        cond_emb = None
        if self.cond_embedding is not None:
            if cond is None:
                raise ValueError("decoder 已启用条件分支时必须提供 cond (B, cond_dim)。")
            cond_emb = self.cond_embedding(cond)

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
