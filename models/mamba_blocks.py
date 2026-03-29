"""
模块化 Backbone：CNNBlock / VSSBlock_1D（光栅 Mamba）/ SS2DBlock（VMamba 风格 SS2D）。

对齐 1.txt 消融：block_type 可选 cnn | mamba_1d | ss2d。
Phase2：AdaLN 在 LayerNorm 后对特征做 FiLM 式调制（x*(1+gamma)+beta），避免仅输入端 Concat 的条件信号稀释。
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization（借鉴 DiT, Peebles & Xie, ICCV 2023；调制形式见 FiLM, Perez et al., AAAI 2018）。

    将条件嵌入映射为 gamma、beta，对 LayerNorm 后的 Token 特征做 x * (1 + gamma) + beta，
    使条件在解码器每一层持续起作用，而非仅在输入拼接一次。

    最后一层 Linear 采用 Zero-Init：训练初期 gamma,beta -> 0，块近似恒等映射，利于深层稳定优化。
    """

    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim * 2),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x, cond):
        # x: [B, L, feat_dim], cond: [B, cond_dim]
        emb = self.mlp(cond)
        gamma, beta = emb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


class SS2D(nn.Module):
    """
    2D Selective Scan（Cross-Scan）：(B, C, H, W) -> (B, C, H, W)。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_tfwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_tbwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.out_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        B, C, H, W = x.shape

        x_fwd = x.view(B, C, -1).transpose(1, 2).contiguous()
        x_bwd = torch.flip(x_fwd, dims=[1]).contiguous()
        x_t = x.transpose(2, 3).contiguous()
        x_tfwd = x_t.view(B, C, -1).transpose(1, 2).contiguous()
        x_tbwd = torch.flip(x_tfwd, dims=[1]).contiguous()

        y_fwd = self.mamba_fwd(x_fwd)
        y_bwd = self.mamba_bwd(x_bwd)
        y_tfwd = self.mamba_tfwd(x_tfwd)
        y_tbwd = self.mamba_tbwd(x_tbwd)

        y_bwd = torch.flip(y_bwd, dims=[1])
        y_tbwd = torch.flip(y_tbwd, dims=[1])

        y_fwd_2d = y_fwd.transpose(1, 2).view(B, C, H, W)
        y_bwd_2d = y_bwd.transpose(1, 2).view(B, C, H, W)
        y_tfwd_2d = y_tfwd.transpose(1, 2).view(B, C, W, H).transpose(2, 3).contiguous()
        y_tbwd_2d = y_tbwd.transpose(1, 2).view(B, C, W, H).transpose(2, 3).contiguous()

        y_cat = torch.cat([y_fwd_2d, y_bwd_2d, y_tfwd_2d, y_tbwd_2d], dim=1)
        y_cat = y_cat.permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(y_cat)
        return out.permute(0, 3, 1, 2).contiguous()


class RasterMamba(nn.Module):
    """1D 光栅扫描：展平 H*W 后经单个 Mamba，再还原为 (B,C,H,W)。"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.view(B, C, -1).transpose(1, 2).contiguous()
        out = self.mamba(seq)
        return out.transpose(1, 2).view(B, C, H, W)


class CNNBlock(nn.Module):
    """纯卷积残差块，作为无 Mamba 基线。"""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x, cond_emb=None):
        del cond_emb  # Baseline：不做条件注入
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return x + out


class VSSBlock_1D(nn.Module):
    """
    与 SS2DBlock 同构的双分支块，将 SS2D 替换为光栅 1D Mamba。
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, cond_embed_dim=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.adaln = AdaLN(cond_embed_dim, dim) if cond_embed_dim is not None else None
        hidden_dim = int(dim * expand)

        self.proj_main = nn.Linear(dim, hidden_dim)
        self.proj_gate = nn.Linear(dim, hidden_dim)

        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True
        )
        self.act = nn.SiLU()

        self.raster_mamba = RasterMamba(
            d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=1
        )
        self.norm_mamba = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x, cond_emb=None):
        B, C, H, W = x.shape
        assert C == self.dim, f"Input channels {C} != dim {self.dim}"

        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm)
        if self.adaln is not None and cond_emb is not None:
            seq = x_norm.reshape(B, H * W, C)
            seq = self.adaln(seq, cond_emb)
            x_norm = seq.view(B, H, W, C)

        x_main = self.proj_main(x_norm).permute(0, 3, 1, 2).contiguous()
        x_gate = self.act(self.proj_gate(x_norm))

        x_main = self.dwconv(x_main)
        x_main = self.act(x_main)
        x_main = self.raster_mamba(x_main)

        x_main = x_main.permute(0, 2, 3, 1).contiguous()
        x_main = self.norm_mamba(x_main)
        x_fused = x_main * x_gate

        out = self.out_proj(x_fused).permute(0, 3, 1, 2).contiguous()
        return out + x


class SS2DBlock(nn.Module):
    """
    VMamba 风格：DWConv + SS2D + 门控；接口 (B, C, H, W)。

    当传入 cond_embed_dim 时，在首层 LayerNorm 之后注入 AdaLN，对应 1.txt 中
    「ConditionalMambaBlock」在深度层反复注入条件的思想（此处为 2D 特征图实现）。
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, cond_embed_dim=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.adaln = AdaLN(cond_embed_dim, dim) if cond_embed_dim is not None else None
        hidden_dim = int(dim * expand)

        self.proj_main = nn.Linear(dim, hidden_dim)
        self.proj_gate = nn.Linear(dim, hidden_dim)

        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True
        )
        self.act = nn.SiLU()

        self.ss2d = SS2D(
            d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=1
        )
        self.norm_ss2d = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x, cond_emb=None):
        B, C, H, W = x.shape
        assert C == self.dim, f"Input channels {C} != dim {self.dim}"

        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm)
        if self.adaln is not None and cond_emb is not None:
            seq = x_norm.reshape(B, H * W, C)
            seq = self.adaln(seq, cond_emb)
            x_norm = seq.view(B, H, W, C)

        x_main = self.proj_main(x_norm).permute(0, 3, 1, 2).contiguous()
        x_gate = self.act(self.proj_gate(x_norm))

        x_main = self.dwconv(x_main)
        x_main = self.act(x_main)
        x_main = self.ss2d(x_main)

        x_main = x_main.permute(0, 2, 3, 1).contiguous()
        x_main = self.norm_ss2d(x_main)
        x_fused = x_main * x_gate

        out = self.out_proj(x_fused).permute(0, 3, 1, 2).contiguous()
        return out + x


# 兼容旧代码中的命名
VSSBlock = SS2DBlock


# 1.txt 中 1D 序列版 ConditionalMambaBlock 的语义，由带 cond_embed_dim 的
# SS2DBlock / VSSBlock_1D 在 (B,C,H,W) 上完成；CNNBlock 不做条件注入（消融基线）。
