"""FreyaDiT: a non-autoregressive flow-matching DiT for Korean TTS.

Latent frames self-attend with rotary position embeddings and cross-attend to
character-level text features refined by a small ConvNeXt stack. Trained with
an optimal-transport rectified-flow objective on frozen VoxCPM2 VAE latents.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



# noise seed for the shipped voice. Re-pin this once a Korean SFT checkpoint
# has a chosen voice; any fixed seed gives a fixed (but arbitrary) speaker
# until then. Was LEYLA_SEED=9 (spread 2.3 Hz vs 14.9 Hz for seed 0) for the
# production seed for the Korean 063 대화체 voice-lock (chosen by ear after SFT
# stage 1: speaker 063, 대화체-only, step1500).
DEFAULT_SEED = 11


def rope_freqs(dim, length, theta=10000.0, device=None):
    """Build rotary embedding angles of shape [length, dim]."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(length, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cat([freqs, freqs], dim=-1)


def apply_rope(x, cos, sin):
    """Apply rotary position embedding to a [B, H, N, D] tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def additive_mask(mask, dtype):
    """Turn a bool key-padding mask [B, N] into an additive float mask [B, 1, 1, N].

    SDPA's memory-efficient backend wants an additive mask rather than bool.
    """
    if mask is None:
        return None
    out = torch.zeros(mask.shape[0], 1, 1, mask.shape[1], device=mask.device, dtype=dtype)
    return out.masked_fill(~mask[:, None, None, :], float("-inf"))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embed(t, dim, max_period=10000):
    """Sinusoidal embedding of a scalar diffusion time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    angles = t[:, None] * freqs[None]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class SelfAttn(nn.Module):
    """Multi-head self-attention over latent frames, with RoPE."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, mask=None):
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.h) for t in self.qkv(x).chunk(3, -1)]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask(mask, q.dtype))
        return self.o(rearrange(out, "b h n d -> b n (h d)"))


class CrossAttn(nn.Module):
    """Frames attend to character features. No positional encoding on the text side;
    alignment is learned through attention."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, ctx, ctx_mask=None):
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.h)
        k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.h) for t in self.kv(ctx).chunk(2, -1)]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask(ctx_mask, q.dtype))
        return self.o(rearrange(out, "b h n d -> b n (h d)"))


class SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """DiT block: self-attention, cross-attention, SwiGLU FFN.

    Each of the three sub-layers is gated by adaLN-zero, so a single timestep
    conditioning vector produces nine modulation signals (shift/scale/gate x3).
    Gates start at zero so the block is initialized to identity.
    """

    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.sa = SelfAttn(d, heads)
        self.nx = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.xa = CrossAttn(d, heads)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLU(d, ff)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, ctx, cos, sin, fmask=None, cmask=None):
        s1, b1, g1, sx, bx, gx, s2, b2, g2 = self.ada(c).chunk(9, -1)
        x = x + g1.unsqueeze(1) * self.sa(modulate(self.n1(x), b1, s1), cos, sin, fmask)
        x = x + gx.unsqueeze(1) * self.xa(modulate(self.nx(x), bx, sx), ctx, cmask)
        x = x + g2.unsqueeze(1) * self.ff(modulate(self.n2(x), b2, s2))
        return x


class ConvNeXt1d(nn.Module):
    """1-D ConvNeXt block used to refine character embeddings before cross-attention."""

    def __init__(self, d, mult=2):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.n = nn.LayerNorm(d)
        self.p1 = nn.Linear(d, d * mult)
        self.p2 = nn.Linear(d * mult, d)

    def forward(self, x):
        residual = x
        x = self.dw(x.transpose(1, 2)).transpose(1, 2)
        x = self.n(x)
        return residual + self.p2(F.gelu(self.p1(x)))


class FreyaDiT(nn.Module):
    """FreyaTTS acoustic model.

    Predicts the rectified-flow velocity field over 64-dim VoxCPM2 latent
    frames, conditioned on character ids. Also carries a small duration head
    that regresses log frame count from mean-pooled text features.
    """

    def __init__(self, vocab, feat=64, d=768, depth=22, heads=12, ff=2048, text_conv=4, fill_id=0):
        super().__init__()
        self.feat = feat
        self.d = d
        self.heads = heads
        self.char_emb = nn.Embedding(vocab, d)
        self.text_conv = nn.ModuleList([ConvNeXt1d(d) for _ in range(text_conv)])
        self.x_proj = nn.Linear(feat, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(depth)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_f[-1].weight)
        nn.init.zeros_(self.ada_f[-1].bias)
        self.out = nn.Linear(d, feat)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.dur = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

    def text_encode(self, text_ids):
        """Embed character ids and refine them with the ConvNeXt stack."""
        x = self.char_emb(text_ids)
        for conv in self.text_conv:
            x = conv(x)
        return x

    def forward(self, x_t, t, ctx, fmask=None, cmask=None):
        """Predict the velocity field at noisy latents x_t and time t."""
        B, T, _ = x_t.shape
        head_dim = self.d // self.heads
        base = rope_freqs(head_dim, T, device=x_t.device)
        cos = torch.cos(base)[None, None]
        sin = torch.sin(base)[None, None]

        h = self.x_proj(x_t)
        c = self.t_mlp(timestep_embed(t, self.d))
        for blk in self.blocks:
            h = blk(h, c, ctx, cos, sin, fmask, cmask)

        s, g = self.ada_f(c).chunk(2, -1)
        return self.out(modulate(self.nf(h), s, g))

    def cfm_loss(self, x1, text_ids, fmask=None, cmask=None):
        """Conditional flow-matching loss on clean latents x1 (masked mean if fmask given)."""
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        ctx = self.text_encode(text_ids)
        v = self(xt, t, ctx, fmask, cmask)
        target = x1 - x0
        if fmask is None:
            return ((v - target) ** 2).mean()
        m = fmask[..., None].float()
        return (((v - target) ** 2) * m).sum() / (m.sum() * self.feat + 1e-6)

    def dur_loss(self, text_ids, logT, cmask=None):
        """MSE loss of the duration head against log frame counts."""
        te = self.text_encode(text_ids)
        if cmask is not None:
            pooled = (te * cmask[..., None].float()).sum(1) / (cmask.sum(1, keepdim=True) + 1e-6)
        else:
            pooled = te.mean(1)
        pred = self.dur(pooled).squeeze(-1)
        return ((pred - logT) ** 2).mean(), pred

    @torch.no_grad()
    def sample(self, text_ids, T, steps=32, cmask=None, seed=DEFAULT_SEED):
        """Integrate the ODE from noise to latents with a fixed-step Euler solver."""
        B = text_ids.shape[0]
        ctx = self.text_encode(text_ids)
        device = text_ids.device
        if seed is None:
            x = torch.randn(B, T, self.feat, device=device)
        else:
            gen = torch.Generator(device=device).manual_seed(int(seed))
            x = torch.randn(B, T, self.feat, device=device, generator=gen)
        for i in range(steps):
            t = torch.full((B,), i / steps, device=x.device)
            x = x + self(x, t, ctx, None, cmask) / steps
        return x
