"""OT-CFM U-Net for CIFAR-10/CIFAR-2 (32x32x3)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


def _gn(ch):
    g = min(32, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch)
        # DDPM-standard dropout position: after the second norm+SiLU, before conv2.
        # nn.Dropout has no parameters, so state_dicts stay compatible either way.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj_out(attn)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownBlock(nn.Module):
    """One encoder stage: num_res_blocks ResBlocks (with optional attention), then optional downsample."""

    def __init__(self, in_ch, out_ch, temb_dim, num_res_blocks, use_attn, downsample, dropout=0.0):
        super().__init__()
        self.resblocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        ch = in_ch
        for _ in range(num_res_blocks):
            self.resblocks.append(ResBlock(ch, out_ch, temb_dim, dropout=dropout))
            self.attns.append(AttentionBlock(out_ch) if use_attn else nn.Identity())
            ch = out_ch
        self.downsample = Downsample(out_ch) if downsample else None

    def forward(self, x, temb):
        skips = []
        h = x
        for res, attn in zip(self.resblocks, self.attns):
            h = res(h, temb)
            h = attn(h)
            skips.append(h)
        if self.downsample is not None:
            h = self.downsample(h)
            skips.append(h)
        return h, skips


class UpBlock(nn.Module):
    """One decoder stage: num_res_blocks+1 ResBlocks (concat skip each time), optional attention, optional upsample."""

    def __init__(self, in_ch, out_ch, skip_channels, temb_dim, num_res_blocks, use_attn, upsample, dropout=0.0):
        super().__init__()
        self.resblocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        ch = in_ch
        for i in range(num_res_blocks + 1):
            self.resblocks.append(ResBlock(ch + skip_channels[i], out_ch, temb_dim, dropout=dropout))
            self.attns.append(AttentionBlock(out_ch) if use_attn else nn.Identity())
            ch = out_ch
        self.upsample = Upsample(out_ch) if upsample else None

    def forward(self, x, temb, skips):
        h = x
        for res, attn in zip(self.resblocks, self.attns):
            h = torch.cat([h, skips.pop()], dim=1)
            h = res(h, temb)
            h = attn(h)
        if self.upsample is not None:
            h = self.upsample(h)
        return h


class UNetCFM(nn.Module):
    """U-Net for OT-CFM / DDPM. Channel progression: 128 -> 256 -> 256 -> 256.
    Attention at 16x16 resolution. ~36M params on 32x32x3 input.

    Supports optional class conditioning via learned embeddings.
    When num_classes is set, the model accepts an additional class_label argument.
    The null class index (num_classes) is used for classifier-free guidance dropout.
    """

    def __init__(self, in_ch=3, base_ch=128, ch_mults=(1, 2, 2, 2),
                 attn_resolutions=(16,), num_res_blocks=2, num_classes=None,
                 dropout=0.0):
        super().__init__()
        temb_dim = base_ch * 4
        self.time_embed = SinusoidalTimeEmbedding(base_ch)
        self.num_classes = num_classes
        if num_classes is not None:
            # +1 for null/unconditional class (index = num_classes)
            self.class_embed = nn.Embedding(num_classes + 1, temb_dim)

        channels = [base_ch * m for m in ch_mults]
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Build encoder
        self.down_blocks = nn.ModuleList()
        prev_ch = base_ch
        resolution = 32
        all_skip_channels = [base_ch]  # from conv_in

        for i, ch in enumerate(channels):
            do_down = i < len(channels) - 1
            use_attn = resolution in attn_resolutions
            self.down_blocks.append(DownBlock(prev_ch, ch, temb_dim, num_res_blocks, use_attn, do_down, dropout=dropout))
            for _ in range(num_res_blocks):
                all_skip_channels.append(ch)
            if do_down:
                all_skip_channels.append(ch)
                resolution //= 2
            prev_ch = ch

        # Bottleneck
        self.mid_block1 = ResBlock(prev_ch, prev_ch, temb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(prev_ch)
        self.mid_block2 = ResBlock(prev_ch, prev_ch, temb_dim, dropout=dropout)

        # Build decoder
        self.up_blocks = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            do_up = i < len(channels) - 1
            use_attn = resolution in attn_resolutions
            # Collect skip channels for this stage (num_res_blocks + 1 skips, popped from end)
            skip_chs = []
            for _ in range(num_res_blocks + 1):
                skip_chs.append(all_skip_channels.pop())
            self.up_blocks.append(UpBlock(prev_ch, ch, skip_chs, temb_dim, num_res_blocks, use_attn, do_up, dropout=dropout))
            prev_ch = ch
            if do_up:
                resolution *= 2

        self.norm_out = _gn(prev_ch)
        self.conv_out = nn.Conv2d(prev_ch, in_ch, 3, padding=1)

    def forward(self, x, t, class_label=None):
        """x: (B, C, H, W), t: (B,) or scalar, class_label: (B,) optional.
        Returns velocity v_θ(x,t) or noise prediction ε_θ(x,t)."""
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        temb = self.time_embed(t)
        if self.num_classes is not None and class_label is not None:
            temb = temb + self.class_embed(class_label)

        h = self.conv_in(x)
        skips = [h]

        for block in self.down_blocks:
            h, s = block(h, temb)
            skips.extend(s)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for block in self.up_blocks:
            h = block(h, temb, skips)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return h
