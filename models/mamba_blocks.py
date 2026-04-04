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


class MambaSemanticMapper(nn.Module):
    """
    Phase 3：基于 Mamba 的序列语义映射器（Sequential Semantic Mapper）。
    将 CLIP Text Encoder 的 Token 序列 [B, L, C] 压缩为供 AdaLN 使用的全局条件向量 [B, hidden_dim]。
    双向扫描参考 Vim；序列池化参考 ZigMa 类工作。
    """

    def __init__(
        self,
        clip_text_dim=768,
        hidden_dim=256,
        bidirectional=True,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.proj_in = nn.Linear(clip_text_dim, hidden_dim)
        self.mamba_fwd = Mamba(
            d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        if self.bidirectional:
            self.mamba_rev = Mamba(
                d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand
            )
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, text_seq):
        # text_seq: [B, L, clip_text_dim]
        x = self.proj_in(text_seq)
        if self.bidirectional:
            x_fwd = self.mamba_fwd(x)
            x_rev = self.mamba_rev(x.flip(dims=[1])).flip(dims=[1])
            x = x_fwd + x_rev
        else:
            x = self.mamba_fwd(x)
        pooled = x.mean(dim=1)
        return self.proj_out(pooled)


class LinearSemanticMapper(nn.Module):
    """
    Phase 3.1 Baseline：线性/MLP 池化映射（模拟 “拿到序列后直接池化 + MLP” 的粗粒度做法）。
    """

    def __init__(self, clip_text_dim=768, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, text_seq):
        pooled = text_seq.mean(dim=1)
        return self.proj(pooled)


class AttentionSemanticMapper(nn.Module):
    """
    Phase 3.1 Baseline：自注意力映射（Transformer 标准做法），之后仍池化为全局向量以对接 AdaLN。
    """

    def __init__(self, clip_text_dim=768, hidden_dim=256, num_heads=4):
        super().__init__()
        self.proj_in = nn.Linear(clip_text_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, text_seq):
        x = self.proj_in(text_seq)
        x, _ = self.attn(x, x, x, need_weights=False)
        pooled = x.mean(dim=1)
        return self.proj_out(pooled)


class MambaSemanticMapper_NoPool(nn.Module):
    """
    Phase 3.2：不做池化，保留完整文本序列特征供 Cross-Attention 注入。
    输出形状：[B, L, hidden_dim]
    """

    def __init__(self, clip_text_dim=512, hidden_dim=256, bidirectional=True):
        super().__init__()
        self.proj_in = nn.Linear(clip_text_dim, hidden_dim)
        self.bidirectional = bidirectional

        self.mamba_fwd = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        if self.bidirectional:
            self.mamba_rev = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)

    def forward(self, text_seq):
        x = self.proj_in(text_seq)
        x_fwd = self.mamba_fwd(x)
        if self.bidirectional:
            x_rev = self.mamba_rev(x.flip(dims=[1])).flip(dims=[1])
            return x_fwd + x_rev
        return x_fwd


class HybridCrossAttnBlock(nn.Module):
    """
    文本序列 [B, Lt, D] 注入到视觉序列 [B, Lv, D] 的 Cross-Attention 块（视觉为 Query）。
    """

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, v_seq, t_seq):
        q = self.norm_q(v_seq)
        kv = self.norm_kv(t_seq)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return v_seq + attn_out


class MambaSemanticMapper_Dual(nn.Module):
    """
    Phase 3.3：同时输出序列特征与池化全局向量。
      - out_seq:   [B, L, hidden_dim] 用于 Cross-Attention
      - out_pooled:[B, hidden_dim]    用于 AdaLN（全局结构稳定）
    """

    def __init__(self, clip_text_dim=512, hidden_dim=256, bidirectional=True):
        super().__init__()
        self.proj_in = nn.Linear(clip_text_dim, hidden_dim)
        self.bidirectional = bidirectional

        self.mamba_fwd = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        if self.bidirectional:
            self.mamba_rev = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)

    def forward(self, text_seq):
        x = self.proj_in(text_seq)
        x_fwd = self.mamba_fwd(x)
        if self.bidirectional:
            x_rev = self.mamba_rev(x.flip(dims=[1])).flip(dims=[1])
            out_seq = x_fwd + x_rev
        else:
            out_seq = x_fwd
        out_pooled = out_seq.mean(dim=1)
        return out_seq, out_pooled


class GatedHybridCrossAttnBlock(nn.Module):
    """
    Phase 3.3：带零初始化 gate 的 Cross-Attention。
    初始 gate=0，使模型训练初期等效于不注入 Cross-Attn，避免破坏视觉流形（保护 LPIPS）。
    """

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        # Phase 5：通道级门控，warm-start 0.02 解决冷启动（零初始化时梯度过小导致 Cross-Attn 永远沉默）
        self.gate = nn.Parameter(torch.full((hidden_dim,), 0.02))

    def forward(self, v_seq, t_seq):
        q = self.norm_q(v_seq)
        kv = self.norm_kv(t_seq)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        # gate: [D]，通过广播作用到 [B, Lv, D]
        return v_seq + self.gate * attn_out
