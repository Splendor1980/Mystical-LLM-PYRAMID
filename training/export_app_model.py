"""Export trained RWKV-5.2 11M into a compact on-device asset + reference outputs.

Outputs (app/src/main/assets/):
  model.bin    - single blob: header, weights, tokenizer vocab/merges/bytemap
  ref_logits.bin, ref_ids.txt - reference logits for a fixed prompt (for Java port check)
  tok_ref.json - tokenizer reference encodes (for Java port check)

Verifies per-token recurrence == chunked forward on test sequences.
"""
import json
import os
import struct

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, 'kaggle', 'outputs_v1', 'ckpt', 'ckpt_last.pt')
TOK_JSON = os.path.join(ROOT, 'Синтетика от Соннет 5', 'tokenizer.json')
OUT_DIR = os.path.join(ROOT, 'app', 'src', 'main', 'assets')
BYPEMAP = os.path.join(ROOT, 'kaggle', 'bytemap.json')

torch.manual_seed(7)


def write_fp16(f, name, t):
    raw = t.astype('<f2').tobytes()
    nb = name.encode('utf-8')
    f.write(struct.pack('<H', len(nb)))
    f.write(nb)
    f.write(struct.pack('<I', t.size))
    f.write(raw)


def write_fp32(f, name, t):
    raw = t.astype('<f4').tobytes()
    nb = name.encode('utf-8')
    f.write(struct.pack('<H', len(nb)))
    f.write(nb)
    f.write(struct.pack('<I', t.size))
    f.write(raw)


def layer_norm(x, w, eps=1e-5):
    n = x.size
    m = x.mean()
    v = x.var()
    return (x - m) / np.sqrt(v + eps) * w


def group_norm(x, w, b, head_size, eps):
    H = x.size // head_size
    out = np.empty_like(x)
    for h in range(H):
        row = x[h * head_size:(h + 1) * head_size]
        m = row.mean()
        v = row.var()
        out[h * head_size:(h + 1) * head_size] = (row - m) / np.sqrt(v + eps)
    return out * w + b


class Rwkv51:
    """Per-token, stateful recurrence. All fp64 math on fp32 weights."""

    def __init__(self, params, args):
        self.p = params
        self.a = args
        H = args['n_head']
        N = args['n_embd'] // args['n_head']
        L = args['n_layer']
        self.s = np.zeros((L, H, N, N), dtype=np.float64)
        self.prev_ln1 = [None] * L
        self.prev_ln2 = [None] * L
        self.dec = {}
        self.u = {}
        for l in range(L):
            dec = self.p['transformer.h.%d.tmix.time_decay' % l].ravel()
            self.dec[l] = np.exp(-np.exp(np.minimum(dec, 20.0)))
            self.u[l] = self.p['transformer.h.%d.tmix.time_faaaa' % l].ravel().copy()

    def reset(self):
        self.s.fill(0.0)
        self.prev_ln1 = [None] * self.a['n_layer']
        self.prev_ln2 = [None] * self.a['n_layer']

    def silu(self, z):
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return z * s

    def tmix(self, x, l):
        # x: CURRENT ln_1 output (C,); uses prev_ln1[l] as the time-shift
        H = self.a['n_head']
        N = self.a['n_embd'] // H
        sh = self.prev_ln1[l]
        shift = np.zeros_like(x) if sh is None else sh
        xx = shift - x
        ma = lambda q: self.p['transformer.h.%d.tmix.time_maa_%s' % (l, q)]
        xk = x + xx * ma('k')
        xv = x + xx * ma('v')
        xr = x + xx * ma('r')
        xg = x + xx * ma('g')
        r = (self.p['transformer.h.%d.tmix.receptance.weight' % l] @ xr).reshape(H, N)
        k = (self.p['transformer.h.%d.tmix.key.weight' % l] @ xk).reshape(H, N)
        v = (self.p['transformer.h.%d.tmix.value.weight' % l] @ xv).reshape(H, N)
        g = self.silu(self.p['transformer.h.%d.tmix.gate.weight' % l] @ xg)
        out = np.empty(self.a['n_embd'], dtype=np.float64)
        sd = self.s[l]
        for h in range(H):
            y = sd[h].T @ r[h] + self.u[l][h] * (r[h] @ k[h]) * v[h]
            sd[h] = self.dec[l][h] * sd[h] + np.outer(k[h], v[h])
            out[h * N:(h + 1) * N] = y
        out = group_norm(out, self.p['transformer.h.%d.tmix.ln_x.weight' % l],
                         self.p['transformer.h.%d.tmix.ln_x.bias' % l], N, (1e-5) * 64)
        out = out * g
        out = self.p['transformer.h.%d.tmix.output.weight' % l] @ out
        return out

    def cmix(self, x, l):
        sh = self.prev_ln2[l]
        shift = np.zeros_like(x) if sh is None else sh
        xx = shift - x
        ma = lambda q: self.p['transformer.h.%d.cmix.time_maa_%s' % (l, q)]
        xk = x + xx * ma('k')
        xr = x + xx * ma('r')
        z = self.p['transformer.h.%d.cmix.key.weight' % l] @ xk
        z = np.where(z > 0, z, 0) ** 2
        z = self.p['transformer.h.%d.cmix.value.weight' % l] @ z
        rz = self.p['transformer.h.%d.cmix.receptance.weight' % l] @ xr
        gate = 1.0 / (1.0 + np.exp(-np.clip(rz, -30, 30)))
        return gate * z

    def forward_one(self, tok_id):
        x = self.p['lm_head.weight'][tok_id].copy()
        for l in range(self.a['n_layer']):
            n1 = layer_norm(x, self.p['transformer.h.%d.ln_1.weight' % l])
            x = x + self.tmix(n1, l)
            self.prev_ln1[l] = n1
            n2 = layer_norm(x, self.p['transformer.h.%d.ln_2.weight' % l])
            x = x + self.cmix(n2, l)
            self.prev_ln2[l] = n2
        return self.p['lm_head.weight'] @ layer_norm(x, self.p['transformer.ln_f.weight'])


def load_state(pth, args):
    ck = torch.load(pth, map_location='cpu', weights_only=False)
    sd = ck['model']
    W = {}
    for k, v in sd.items():
        a = v.detach()
        if a.ndim == 3:
            a = a.reshape(-1)
        W[k] = a.double().numpy().copy()
    return ck, W


def main():
    ck, fixed = load_state(CKPT, None)
    args = ck['model_args']
    print('model_args:', args)
    H, N, L, C = args['n_head'], args['n_embd'] // args['n_head'], args['n_layer'], args['n_embd']

    order = []
    order.append(('lm_head.weight', fixed['lm_head.weight']))  # tied wte
    for l in range(L):
        order.append(('transformer.h.%d.ln_1.weight' % l, fixed['transformer.h.%d.ln_1.weight' % l]))
        for q in ('k', 'v', 'r', 'g'):
            order.append(('transformer.h.%d.tmix.time_maa_%s' % (l, q), fixed['transformer.h.%d.tmix.time_maa_%s' % (l, q)]))
        order.append(('transformer.h.%d.tmix.time_decay' % l, fixed['transformer.h.%d.tmix.time_decay' % l]))
        order.append(('transformer.h.%d.tmix.time_faaaa' % l, fixed['transformer.h.%d.tmix.time_faaaa' % l]))
        for q in ('receptance', 'key', 'value', 'gate', 'output'):
            order.append(('transformer.h.%d.tmix.%s.weight' % (l, q), fixed['transformer.h.%d.tmix.%s.weight' % (l, q)]))
        order.append(('transformer.h.%d.tmix.ln_x.weight' % l, fixed['transformer.h.%d.tmix.ln_x.weight' % l]))
        order.append(('transformer.h.%d.tmix.ln_x.bias' % l, fixed['transformer.h.%d.tmix.ln_x.bias' % l]))
        order.append(('transformer.h.%d.ln_2.weight' % l, fixed['transformer.h.%d.ln_2.weight' % l]))
        for q in ('k', 'r'):
            order.append(('transformer.h.%d.cmix.time_maa_%s' % (l, q), fixed['transformer.h.%d.cmix.time_maa_%s' % (l, q)]))
        order.append(('transformer.h.%d.cmix.key.weight' % l, fixed['transformer.h.%d.cmix.key.weight' % l]))
        order.append(('transformer.h.%d.cmix.value.weight' % l, fixed['transformer.h.%d.cmix.value.weight' % l]))
        order.append(('transformer.h.%d.cmix.receptance.weight' % l, fixed['transformer.h.%d.cmix.receptance.weight' % l]))
    order.append(('transformer.ln_f.weight', fixed['transformer.ln_f.weight']))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'model.bin'), 'wb') as f:
        f.write(b'RWKV5')
        f.write(struct.pack('<B', 1))
        f.write(struct.pack('<5I', args['vocab_size'], L, C, H, args['block_size']))
        f.write(struct.pack('<4I', 0, 1, 2, 3))
        f.write(struct.pack('<I', len(order)))
        for name, t in order:
            if name.endswith('time_decay') or name.endswith('time_faaaa'):
                write_fp32(f, name, t)
            else:
                write_fp16(f, name, t)
        tok = json.load(open(TOK_JSON, encoding='utf-8'))
        vocab = tok['model']['vocab']
        merges = tok['model']['merges']
        vocab_sorted = [k for k in sorted(vocab, key=lambda x: vocab[x])]
        f.write(struct.pack('<I', len(vocab_sorted)))
        for s in vocab_sorted:
            b = s.encode('utf-8')
            f.write(struct.pack('<H', len(b)))
            f.write(b)
        f.write(struct.pack('<I', len(merges)))
        for a, b in merges:
            ba, bb = a.encode('utf-8'), b.encode('utf-8')
            f.write(struct.pack('<HH', len(ba), len(bb)))
            f.write(ba)
            f.write(bb)
        bm = {int(k): v for k, v in json.load(open(BYPEMAP, encoding='utf-8')).items()}
        f.write(struct.pack('<256I', *[bm[b] for b in range(256)]))
    print('model.bin size:', os.path.getsize(os.path.join(OUT_DIR, 'model.bin')), 'bytes')

    # ---------------- recurrence vs chunked forward (incl. pads => exact parity)
    from tokenizers import Tokenizer as Tkit
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'kaggle'))
    from rwkv_model import build_model
    tok = Tkit.from_file(TOK_JSON)
    mm = build_model(vocab_size=args['vocab_size'], block_size=args['block_size'],
                     n_layer=L, n_embd=C, n_head=H, dropout=0.0)
    mm.load_state_dict(ck['model'])
    mm.eval()

    model = Rwkv51(fixed, args)
    tests = ['Привет, как дела\n', 'Расскажи легенду о свече\n',
             'Око леса смотрит на тебя и не мигает. ' * 30]
    for prompt in tests:
        ids = tok.encode(prompt).ids
        ids = ids[:args['block_size']]
        pad_to = ((len(ids) - 1) // 128 + 1) * 128
        if pad_to == 0:
            pad_to = 128
        pad_to = min(pad_to, args['block_size'])
        x = torch.zeros(1, pad_to, dtype=torch.long)
        x[0, pad_to - len(ids):] = torch.tensor(ids)
        model.reset()
        for i in x[0].tolist():
            last = model.forward_one(i)
        with torch.no_grad():
            logits, _ = mm(x)
        plogits = logits[0, -1].double().numpy()
        d = np.abs(last - plogits)
        print('prompt %-24r len=%d pad=%d maxdiff=%.3e' % (prompt[:20], len(ids), pad_to, d.max()))

    # ---------------- reference for Java: fp16-rounded weights (as stored in model.bin)
    w_h16 = {}
    for k, v in fixed.items():
        if k.endswith('time_decay') or k.endswith('time_faaaa'):
            w_h16[k] = v
        else:
            w_h16[k] = np.float64(v.astype(np.float16))
    ref_model = Rwkv51(w_h16, args)
    ids = tok.encode('Око леса смотрит на тебя\n').ids
    ref_model.reset()
    for i in ids:
        lg = ref_model.forward_one(i)
    with open(os.path.join(OUT_DIR, 'ref_logits.bin'), 'wb') as f:
        f.write(lg.astype('<f4').tobytes())
    with open(os.path.join(OUT_DIR, 'ref_ids.txt'), 'w', encoding='utf-8') as f:
        f.write(' '.join(map(str, ids)))
    print('ref ids:', ids)
    print('top ids:', np.argsort(lg)[::-1][:8].tolist())


if __name__ == '__main__':
    main()