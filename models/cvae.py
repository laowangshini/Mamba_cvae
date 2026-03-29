import torch
import torch.nn as nn
from .encoder import MambaEncoder
from .decoder import MambaDecoder


class MambaCVAE(nn.Module):
    """
    CVAE（Phase2 / Phase3 / 1.txt）：Encoder 仅编码图像；Decoder 内可选条件分支 + 各层 AdaLN。
    cond_mode=attr 且 cond_dim=0：无条件（与旧 checkpoint 一致）。
    cond_mode=clip_seq：cond 为 CLIP 文本序列 [B,L,C]，经 MambaSemanticMapper 后注入 AdaLN。
    """

    def __init__(
        self,
        latent_dim=128,
        block_type="ss2d",
        cond_dim=0,
        cond_embed_dim=256,
        cond_mode="attr",
        clip_text_dim=768,
        mapper_bidirectional=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_embed_dim
        self.cond_mode = cond_mode
        self.clip_text_dim = clip_text_dim

        self.encoder = MambaEncoder(latent_dim=latent_dim, block_type=block_type)
        self.decoder = MambaDecoder(
            latent_dim=latent_dim,
            block_type=block_type,
            cond_dim=cond_dim,
            cond_embed_dim=cond_embed_dim,
            cond_mode=cond_mode,
            clip_text_dim=clip_text_dim,
            mapper_bidirectional=mapper_bidirectional,
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
