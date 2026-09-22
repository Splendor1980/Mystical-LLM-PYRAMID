"""
RWKV-5.2 (Eagle, "x051a-style") model ~11M params, vocab 3000.
Reference: BlinkDL/nanoRWKV (model.py) + RWKV-LM init rules (README_UNIFIED.md §7.2-7.3).

No custom CUDA kernel required: trains on any GPU/CPU.
No positional embedding (pure RNN recurrence, positions come from time-decay).
Weight-tying between token embedding and LM head.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0..1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1..~0
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd

            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))  # per-head, keep fp32 in state

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))  # time-first "u"

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd, eps=(1e-5) * 64)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        H, N = self.n_head, self.head_size
        if T % 256 == 0:
            Q = 256
        elif T % 128 == 0:
            Q = 128
        else:
            Q = 256
            if T < Q:
                raise ValueError("block_size should be >= 256 for RWKV x051a chunking")
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2)
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1)
        v = self.value(xv).view(B, T, H, N).transpose(1, 2)
        g = F.silu(self.gate(xg))

        w = torch.exp(-torch.exp(self.time_decay.float()))  # time_decay -> (H,1)
        u = self.time_faaaa.float()  # time_first (H,1)

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(Q - 1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2 * Q - 1)
        w = w[:, :, Q - 1:].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype)
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype)

        for i in range(T // Q):
            rr = r[:, :, i * Q:i * Q + Q, :]
            kk = k[:, :, :, i * Q:i * Q + Q]
            vv = v[:, :, i * Q:i * Q + Q, :]
            y[:, :, i * Q:i * Q + Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = ws * state + (kk * wk) @ vv

        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g
        y = self.dropout(self.output(y))
        return y


class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.value = nn.Linear(3 * config.n_embd, config.n_embd, bias=config.bias)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.tmix = RWKV_TimeMix(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cmix = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.cmix(self.ln_2(x))
        return x


class GPTConfig:
    def __init__(self, vocab_size=3000, block_size=512, n_layer=13, n_head=8,
                 n_embd=256, dropout=0.02, bias=False):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


class RWKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply_init()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def apply_init(self):
        L, C, V = self.config.n_layer, self.config.n_embd, self.config.vocab_size

        def _ortho(weight, gain):
            nn.init.orthogonal_(weight, gain=gain)

        with torch.no_grad():
            # head (tied with wte)
            self.lm_head.weight.normal_(std=0.02)
            _ortho(self.lm_head.weight, 0.5 * math.sqrt(V / C))
            for lid, block in enumerate(self.transformer.h):
                for name, p in block.named_parameters():
                    if name.endswith('att.output.weight'):
                        p.zero_()
                    elif name.endswith('att.receptance.weight'):
                        _ortho(p, 1.0)
                    elif name.endswith('att.key.weight'):
                        _ortho(p, 0.1)
                    elif name.endswith('att.value.weight'):
                        _ortho(p, 1.0)
                    elif name.endswith('att.gate.weight'):
                        _ortho(p, 0.1)
                    elif name.endswith('ffn.key.weight'):
                        _ortho(p, 1.0)
                    elif name.endswith('ffn.value.weight'):
                        p.zero_()
                    elif name.endswith('ffn.receptance.weight'):
                        p.zero_()
                    elif name.endswith('att.ln_x.weight'):
                        p.fill_(((1 + lid) / L) ** 0.7)
                    elif name.endswith('.bias'):
                        p.zero_()

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"block_size too small: {t} > {self.config.block_size}"
        assert t % 256 == 0 or t % 128 == 0, "block_size must be multiple of 128"

        x = self.transformer.drop(self.transformer.wte(idx))  # (B,T,C)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        out = idx
        for _ in range(max_new_tokens):
            idx_cond = out if out.size(1) <= self.config.block_size else out[:, -self.config.block_size:]
            t = idx_cond.size(1)
            if t % 128 != 0:
                aligned = (t // 128) * 128
                if aligned > 0:
                    idx_cond = idx_cond[:, -aligned:]
                else:
                    pad = 128 - t
                    idx_cond = torch.cat(
                        [torch.zeros(idx_cond.size(0), pad, dtype=idx_cond.dtype,
                                     device=idx_cond.device), idx_cond], dim=1)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, ix), dim=1)
        return out


def build_model(vocab_size=3000, block_size=512, n_layer=13, n_embd=256,
                n_head=8, dropout=0.02, bias=False):
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=dropout, bias=bias)
    return RWKV(cfg)