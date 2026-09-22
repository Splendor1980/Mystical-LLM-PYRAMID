"""Local generation check for the trained Mystic RWKV-5.2 11M checkpoint.

Usage: python kaggle\gen_check.py --ckpt kaggle\outputs_v1\ckpt\ckpt_last.pt \
          --tok kaggle\outputs_v1\data\tokenizer.json \
          --prompts kaggle\outputs_v1\data\prompts.txt \
          --out kaggle\gen_check_out.txt [--temp 0.8] [--new 160]
Loads exact architecture from ckpt['model_args'], runs CPU generation.
"""
import argparse
import json
import os
import sys
import torch
import tokenizers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_model import build_model  # noqa: E402


def load_ckpt(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    ma = ck['model_args']
    model = build_model(vocab_size=ma['vocab_size'], block_size=ma['block_size'],
                        n_layer=ma['n_layer'], n_embd=ma['n_embd'], n_head=ma['n_head'])
    st = ck['model']
    unwanted = '_orig_mod.'
    for k, v in list(st.items()):
        if k.startswith(unwanted):
            st[k[len(unwanted):]] = st.pop(k)
    model.load_state_dict(st, strict=False)
    model.eval()
    meta = {k: ck.get(k) for k in ('phase', 'iter_num', 'phase_iter', 'best_val')}
    return model, meta, ck['model_args']


def main():
    ap = ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tok', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--new', type=int, default=160)
    ap.add_argument('--suffix', type=str, default='')
    args = ap.parse_args()

    model, meta, ma = load_ckpt(args.ckpt)
    print('arch:', ma)
    print('ckpt meta:', meta)
    print('params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    tok = tokenizers.Tokenizer.from_file(args.tok)
    with open(args.prompts, encoding='utf-8') as f:
        prompts = [l.strip() for l in f if l.strip()]

    lines = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p).ids[:200]
            if args.suffix:
                ids = ids + tok.encode(args.suffix).ids
            idx = torch.tensor([ids], dtype=torch.long)
            print(f'\n=== {p} ===', flush=True)
            gen = model.generate(idx, args.new, temperature=args.temp, top_k=40)
            text = tok.decode(gen[0].tolist())
            text = text.replace('\n', '\n  ')
            print('  ' + text[:400], flush=True)
            lines.append(f'PROMPT: {p}\nGEN:\n{tok.decode(gen[0].tolist())}\n{"-"*60}')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()