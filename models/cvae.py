import torch
import torch.nn as nn
from .encoder import MambaEncoder
from .decoder import MambaDecoder


class MambaCVAE(nn.Module):
    """
    CVAE：编码器仅看图像；解码器可选 AdaLN，将 CelebA 40 维属性嵌入为 cond_emb 后传入各 Mamba 块。
    cond_dim=0 时退化为无条件模型（与旧 checkpoint 行为一致，decoder 不接收有效条件）。
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

        if cond_dim and cond_dim > 0:
            self.cond_embedding = nn.Sequential(
                nn.Linear(cond_dim, cond_embed_dim),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim),
            )
            self.decoder = MambaDecoder(
                latent_dim=latent_dim,
                block_type=block_type,
                cond_embed_dim=cond_embed_dim,
            )
        else:
            self.cond_embedding = None
            self.decoder = MambaDecoder(
                latent_dim=latent_dim,
                block_type=block_type,
                cond_embed_dim=None,
            )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, cond=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if self.cond_embedding is not None:
            if cond is None:
                raise ValueError("cond_dim>0 时必须在 forward 中传入 cond (B, cond_dim)。")
            c_emb = self.cond_embedding(cond)
            recon_x = self.decoder(z, c_emb)
        else:
            recon_x = self.decoder(z, None)
        return recon_x, mu, logvar
