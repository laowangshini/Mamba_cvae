"""
解码器（Phase2 / Phase3 / 1.txt）：根据 block_type 选择 CNN / 1D Mamba / SS2D；
Phase2：cond_dim>0 时用 MLP 将属性向量映射为 AdaLN 条件；
Phase3：cond_mode=clip_seq 时用 MambaSemanticMapper 处理 CLIP 文本 Token 序列 [B,L,C]。
"""
import torch
import torch.nn as nn
from .mamba_blocks import (
    AttentionSemanticMapper,
    CNNBlock,
    GatedHybridCrossAttnBlock,
    HybridCrossAttnBlock,
    LinearSemanticMapper,
    MambaSemanticMapper,
    MambaSemanticMapper_Dual,
    MambaSemanticMapper_NoPool,
    SS2DBlock,
    VSSBlock_1D,
)


class _CrossAttnInject2D(nn.Module):
    """
    将 2D 视觉特征 (B,C,H,W) 展平为序列并投影到 D，与文本序列做 Cross-Attn 后再投影回 C。
    """

    def __init__(self, in_channels, attn_dim):
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.proj_in = nn.Linear(in_channels, attn_dim)
        self.proj_out = nn.Linear(attn_dim, in_channels)

    def forward(self, x, t_seq, cross_block):
        b, c, h, w = x.shape
        assert c == self.in_channels
        v = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        v = self.proj_in(v)
        v = cross_block(v, t_seq)
        v = self.proj_out(v)
        out = v.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out


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
        cond_mode="attr",
        clip_text_dim=768,
        mapper_bidirectional=True,
        attn_heads=4,
        bottleneck_inject_stages=1,
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
        self.cond_mode = cond_mode
        self.clip_text_dim = clip_text_dim
        self.bottleneck_inject_stages = int(bottleneck_inject_stages)

        if cond_mode == "clip_seq":
            self.cond_embedding = MambaSemanticMapper(
                clip_text_dim=clip_text_dim,
                hidden_dim=cond_embed_dim,
                bidirectional=mapper_bidirectional,
            )
            ced = cond_embed_dim
            self.cross_attn_blocks = None
            self._cross_attn_injectors = None
        elif cond_mode == "clip_pooled":
            self.cond_embedding = LinearSemanticMapper(
                clip_text_dim=clip_text_dim, hidden_dim=cond_embed_dim
            )
            ced = cond_embed_dim
            self.cross_attn_blocks = None
            self._cross_attn_injectors = None
        elif cond_mode == "clip_attention":
            self.cond_embedding = AttentionSemanticMapper(
                clip_text_dim=clip_text_dim,
                hidden_dim=cond_embed_dim,
                num_heads=attn_heads,
            )
            ced = cond_embed_dim
            self.cross_attn_blocks = None
            self._cross_attn_injectors = None
        elif cond_mode == "clip_crossattn":
            # Phase 3.2：文本序列不池化 + Cross-Attention 注入
            self.cond_embedding = MambaSemanticMapper_NoPool(
                clip_text_dim=clip_text_dim,
                hidden_dim=cond_embed_dim,
                bidirectional=mapper_bidirectional,
            )
            ced = cond_embed_dim
            self.cross_attn_blocks = nn.ModuleList(
                [
                    HybridCrossAttnBlock(cond_embed_dim, num_heads=attn_heads)
                    for _ in range(3)
                ]
            )
            # 每个 stage 的视觉通道不同，需要投影到 cond_embed_dim 后做 attention，再投影回去
            self._cross_attn_injectors = nn.ModuleList(
                [
                    _CrossAttnInject2D(256, cond_embed_dim),
                    _CrossAttnInject2D(128, cond_embed_dim),
                    _CrossAttnInject2D(64, cond_embed_dim),
                ]
            )
        elif cond_mode == "clip_hybrid":
            # Phase 3.3：Dual-Injection（全局 AdaLN + gated Cross-Attn）
            self.cond_embedding = MambaSemanticMapper_Dual(
                clip_text_dim=clip_text_dim,
                hidden_dim=cond_embed_dim,
                bidirectional=mapper_bidirectional,
            )
            ced = cond_embed_dim
            # Phase 3.4：只在瓶颈层（最低分辨率 stage）做 Cross-Attn，并使用通道级 gate
            self.cross_attn_blocks = nn.ModuleList(
                [
                    GatedHybridCrossAttnBlock(cond_embed_dim, num_heads=attn_heads)
                    for _ in range(3)
                ]
            )
            self._cross_attn_injectors = nn.ModuleList(
                [
                    _CrossAttnInject2D(256, cond_embed_dim),
                    _CrossAttnInject2D(128, cond_embed_dim),
                    _CrossAttnInject2D(64, cond_embed_dim),
                ]
            )
        elif cond_dim and cond_dim > 0:
            self.cond_embedding = nn.Sequential(
                nn.Linear(cond_dim, cond_embed_dim),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim),
            )
            ced = cond_embed_dim
            self.cross_attn_blocks = None
            self._cross_attn_injectors = None
        else:
            self.cond_embedding = None
            ced = None
            self.cross_attn_blocks = None
            self._cross_attn_injectors = None

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
        cond_seq = None
        if self.cond_embedding is not None:
            if cond is None:
                if self.cond_mode in (
                    "clip_seq",
                    "clip_pooled",
                    "clip_attention",
                    "clip_crossattn",
                    "clip_hybrid",
                ):
                    raise ValueError(
                        "decoder 为 clip_* 时必须提供 cond，形状 [B, L, clip_text_dim]。"
                    )
                raise ValueError("decoder 已启用条件分支时必须提供 cond (B, cond_dim)。")
            if self.cond_mode == "clip_crossattn":
                cond_seq = self.cond_embedding(cond)  # [B, Lt, cond_embed_dim]
                cond_emb = cond_seq.mean(dim=1)  # dummy global，供 AdaLN 形状兼容
            elif self.cond_mode == "clip_hybrid":
                cond_seq, cond_emb = self.cond_embedding(cond)  # seq + global
            else:
                cond_emb = self.cond_embedding(cond)

        x = self.fc_in(z)
        x = x.view(-1, self.embed_dim, self.map_size, self.map_size)
        x = self.layer1(x, cond_emb)
        if (
            self.cond_mode == "clip_crossattn"
            and cond_seq is not None
        ) or (
            self.cond_mode == "clip_hybrid"
            and cond_seq is not None
            and self.bottleneck_inject_stages > 0
        ):
            x = self._cross_attn_injectors[0](x, cond_seq, self.cross_attn_blocks[0])
        x = self.up1(x)

        x = self.layer2(x, cond_emb)
        if (
            self.cond_mode == "clip_crossattn"
            and cond_seq is not None
        ) or (
            self.cond_mode == "clip_hybrid"
            and cond_seq is not None
            and self.bottleneck_inject_stages > 1
        ):
            x = self._cross_attn_injectors[1](x, cond_seq, self.cross_attn_blocks[1])
        x = self.up2(x)

        x = self.layer3(x, cond_emb)
        if (
            self.cond_mode == "clip_crossattn"
            and cond_seq is not None
        ) or (
            self.cond_mode == "clip_hybrid"
            and cond_seq is not None
            and self.bottleneck_inject_stages > 2
        ):
            x = self._cross_attn_injectors[2](x, cond_seq, self.cross_attn_blocks[2])
        x = self.up3(x)

        x = self.final_conv(x)
        return self.tanh(x)
