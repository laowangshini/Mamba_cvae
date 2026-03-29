import torch
import torch.nn as nn
from .encoder import MambaEncoder
from .decoder import MambaDecoder


class MambaCVAE(nn.Module):
    """
    CVAE（Phase2 / 1.txt）：Encoder 仅编码图像；Decoder 内可选 cond_embedding + 各层 AdaLN。
    cond_dim=0 时为无条件解码（与旧 checkpoint 结构一致，仅缺少 decoder.cond_embedding）。
    """

    def __init__(
        self,
        latent_dim=128,
        block_type="ss2d",
        cond_dim=0,
        cond_embed_dim=256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_embed_dim

        self.encoder = MambaEncoder(latent_dim=latent_dim, block_type=block_type)
        self.decoder = MambaDecoder(
            latent_dim=latent_dim,
            block_type=block_type,
            cond_dim=cond_dim,
            cond_embed_dim=cond_embed_dim,
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, cond=None):
        return self.decoder(z, cond)

    def forward(self, x, cond=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, cond)
        return recon_x, mu, logvar
