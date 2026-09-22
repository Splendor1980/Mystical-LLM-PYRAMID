"""
Trainer for Mystic RWKV-5.2 11M (vocab 3000) on raw uint16 .bin files.

Pipeline (README_UNIFIED.md §2):
  Phase A (pretrain):   1 epoch over mixed narr(+qa) ~227M tokens, LR 2e-4..4e-4
  Phase B (SFT):        3 epochs over qa_train.bin only,      LR 3e-5..6e-5
  Phase C (optional):   0.5 epoch over 90% narr + 10% qa mixback

Resume-safe: every checkpoint stores phase state, global/phase iter, RNG and
optimizer. Runs A -> B -> C automatically when --phase ALL; stops cleanly near
wall-clock budget (Kaggle session limit) and pushes checkpoints to a private
Kaggle dataset via CLI (container is pre-authenticated).

Works identically on local PC (resolution of data via args) and in Kaggle.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from rwkv_model import RWKV, build_model
except ImportError:
    pass  # concatenated kernel: names are already in module scope

MODEL_TARGET_M = 11.0

# ----------------------------------------------------------------------------- phases
# Each phase: source=('narr','qa','narr+qa'), epochs/weights, lr range, warmup frac, dropout.
def make_phases(qa_epochs=1.0, epochs_b=3, phase_c=False, narr_frac_c=0.9, narr_epochs_a=1.0,
                lr_a_init=3e-4, lr_a_final=3e-5, lr_b_init=5e-5, lr_b_final=1e-5,
                warmup_a=0.015, warmup_b=0.02, dropout_a=0.02, dropout_b=0.05):
    phases = {}
    phases['A'] = dict(
        sources=[('narr', 1.0, narr_epochs_a), ('qa', 1.0, qa_epochs)],
        lr_init=lr_a_init, lr_final=lr_a_final, warmup=warmup_a, dropout=dropout_a,
        name='pretrain-mixed')
    phases['B'] = dict(
        sources=[('qa', 1.0, epochs_b)],
        lr_init=lr_b_init, lr_final=lr_b_final, warmup=warmup_b, dropout=dropout_b,
        name='sft-qa')
    if phase_c:
        phases['C'] = dict(
            sources=[('narr', narr_frac_c, 0.5), ('qa', 1.0 - narr_frac_c, 0.5)],
            lr_init=lr_b_init, lr_final=lr_b_final, warmup=0.01, dropout=dropout_a,
            name='mixback')
    return phases


# ----------------------------------------------------------------------------- data
class DataMix:
    """Weighted random-window samplers over several raw uint16 .bin memmaps."""

    def __init__(self):
        self.items = []  # dict(name, mmap, n_tokens, wt, do_repeat)

    def add(self, path, name, weight=1.0, repeat=1.0):
        n_bytes = os.path.getsize(path)
        assert n_bytes % 2 == 0, f"{path}: odd byte count"
        n_tokens = n_bytes // 2
        mmap = np.memmap(path, dtype=np.uint16, mode='r')
        self.items.append(dict(name=name, mmap=mmap, n_tokens=n_tokens, wt=weight, repeat=repeat))

    def get_batch(self, batch_size, block_size, device):
        x = torch.zeros(batch_size, block_size, dtype=torch.long, device='cpu')
        y = torch.zeros(batch_size, block_size, dtype=torch.long, device='cpu')
        for b in range(batch_size):
            item = random.choices(self.items, weights=[it['wt'] * it['repeat'] for it in self.items])[0]
            hi = item['n_tokens'] - block_size - 1
            ix = random.randint(0, hi) if hi > 0 else 0
            seq = item['mmap'][ix:ix + block_size + 1].astype(np.int64)
            x[b] = torch.from_numpy(seq[:-1])
            y[b] = torch.from_numpy(seq[1:])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def est_tokens_per_epoch(self):
        return sum(it['n_tokens'] * it['repeat'] for it in self.items)


# ----------------------------------------------------------------------------- checkpoint push
class CheckpointPusher:
    def __init__(self, kaggle_user, dataset_slug, push_every_sec, push_dir, enabled=True):
        self.kaggle_user = kaggle_user
        self.slug = dataset_slug
        self.every = push_every_sec
        self.push_dir = push_dir
        self.enabled = enabled and _which('kaggle') is not None
        self.last_push = 0.0
        self.n_pushes = 0
        if not enabled:
            print("[push] auto-push disabled", flush=True)
        elif not _which('kaggle'):
            print("[push] kaggle CLI not found - auto-push unavailable", flush=True)

    def maybe_push(self, now, force=False, msg_suffix=''):
        if not self.enabled:
            return
        if not force and (now - self.last_push) < self.every:
            return
        try:
            os.makedirs(self.push_dir, exist_ok=True)
            meta = {
                "id": f"{self.kaggle_user}/{self.slug}",
                "title": "Mystic RWKV checkpoints",
                "subtitle": "Auto-pushed training progress (pretrain/SFT)",
                "isPrivate": True,
                "licenses": [{"name": "MIT"}],
            }
            with open(os.path.join(self.push_dir, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            msg = f"auto save {self.n_pushes + 1}" + (f" {msg_suffix}" if msg_suffix else "")
            # NOTE: do NOT capture_output - kaggle CLI tqdm progress on stderr fills the
            # 64KB child pipe and hangs the subprocess. Inherit stdout/stderr instead.
            r = subprocess.run(['kaggle', 'datasets', 'version', '-p', self.push_dir, '-m', msg],
                               timeout=3600)
            ok = r.returncode == 0
            print(f"[push] {'ok' if ok else 'FAIL'} ({msg})", flush=True)
            self.n_pushes += 1
            self.last_push = now
        except Exception as e:  # never let the push break training
            print(f"[push] exception (ignored): {e}", flush=True)


def _which(cmd):
    from shutil import which
    return which(cmd)


# ----------------------------------------------------------------------------- eval + samples
def evaluate(model, val_mix, eval_iters, device, ctx):
    model.eval()
    total = 0.0
    narr = qa = None
    for _ in range(eval_iters):
        X, Y = val_mix.get_batch(4, 128, device)
        with ctx:
            _, loss = model(X, Y)
        total += loss.item()
    model.train()
    return total / max(1, eval_iters)


def generate_samples(model, tokenizer, prompts, device, max_new=160, temperature=0.8, top_k=40):
    out_lines = []
    model.eval()
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer.encode(p).ids[:200]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            if idx.size(1) % 128 != 0:
                pad = 128 - (idx.size(1) % 128)
                idx = F.pad(idx, (0, pad), value=0)
            try:
                gen = model.generate(idx, max_new, temperature=temperature, top_k=top_k)
                text = tokenizer.decode(gen[0].tolist())
                out_lines.append(f"PROMPT: {p}\nGEN:\n{text}\n{'-' * 60}")
            except Exception as e:
                out_lines.append(f"PROMPT: {p}\nGEN-ERROR: {e}\n{'-' * 60}")
    model.train()
    return '\n'.join(out_lines)


# ----------------------------------------------------------------------------- main trainer
class Trainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.device = args.device
        if args.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_type = 'cuda' if self.device.startswith('cuda') else 'cpu'

        # dtype / autocast
        if self.device_type == 'cuda':
            if args.dtype == 'float32':
                self.ptdtype = torch.float32
            else:
                self.ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            self.ptdtype = torch.float32
        self.ctx = (torch.autocast(device_type=self.device_type, dtype=self.ptdtype)
                    if self.device_type == 'cuda' else nullcontext())

        self.tokens_per_iter = args.batch_size * args.grad_accum * args.block_size

        # ----------------------------------------------------------------- datasets
        self.train_mix = DataMix()
        if args.train_narr:
            self.train_mix.add(args.train_narr, 'narr')
        if args.train_qa:
            self.train_mix.add(args.train_qa, 'qa')
        self.n_tokens = {it['name']: it['n_tokens'] for it in self.train_mix.items}

        self.val_mix = DataMix()
        if args.val_narr:
            self.val_mix.add(args.val_narr, 'narr')
        if args.val_qa:
            self.val_mix.add(args.val_qa, 'qa')
        self.eval_iters = args.eval_iters

        # ----------------------------------------------------------------- phases
        self.phases = make_phases(
            qa_epochs=args.qa_epochs_a, epochs_b=args.epochs_b, phase_c=args.phase_c,
            narr_frac_c=args.narr_frac_c, lr_a_init=args.lr_a_init, lr_a_final=args.lr_a_final,
            lr_b_init=args.lr_b_init, lr_b_final=args.lr_b_final,
            warmup_a=args.warmup_a, warmup_b=args.warmup_b,
            dropout_a=args.dropout_a, dropout_b=args.dropout_b)
        if not args.phase_c:
            self.phases.pop('C', None)
        self.phase_order = [p for p in self.phases]
        for ph in self.phases.values():
            ntok = 0
            for (name, wt, rep) in ph['sources']:
                base = self.n_tokens.get(name)
                if base is None:
                    base = sum(v for k, v in self.n_tokens.items())  # fallback
                ntok += base * rep
            ph['iters'] = max(1, int(math.ceil(ntok / self.tokens_per_iter)))
            ph['warmup_steps'] = max(1, int(ph['iters'] * ph['warmup']))

        # tokenizer (optional; for generation only)
        self.tokenizer = None
        if args.tokenizer_path and os.path.exists(args.tokenizer_path):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(args.tokenizer_path)
            except Exception as e:
                print(f"[warn] tokenizer not loaded: {e}", flush=True)

        # ----------------------------------------------------------------- model
        self.model = build_model(
            vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
            n_embd=args.n_embd, n_head=args.n_head, dropout=self.phases[self.phase_order[0]]['dropout'])
        nparams = self.model.get_num_params()
        print(f"[model] params: {nparams/1e6:.2f}M (target ~{MODEL_TARGET_M}M)", flush=True)
        self.model.to(self.device)

        self.compile = args.compile and self.device_type == 'cuda'
        if self.compile:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"[warn] compile failed: {e}", flush=True)
                self.compile = False

        # ----------------------------------------------------------------- optimizer
        self.optimizer = self._make_optimizer(model=self.model, lr=self.phases[self.phase_order[0]]['lr_init'])
        self.grad_clip = args.grad_clip

        # ----------------------------------------------------------------- state
        self.iter_num = 0            # global across phases
        self.phase = self.phase_order[0]
        self.phase_iter = 0
        self.best_val = 1e9
        self.history = []
        self.start_wall = time.time()
        self.mfu = -1.0
        self.saved_any = False

        os.makedirs(args.save_dir, exist_ok=True)
        if args.resume:
            self._try_resume()
        print(f"[trainer] phase={self.phase} iter={self.iter_num} (global) "
              f"phase_iter={self.phase_iter}/{self.phases[self.phase]['iters']}", flush=True)

        self.pusher = CheckpointPusher(
            kaggle_user=args.kaggle_user, dataset_slug=args.ckpt_dataset,
            push_every_sec=args.push_every, push_dir=os.path.join(args.save_dir, 'push'),
            enabled=args.auto_push)

    def _make_optimizer(self, model, lr):
        decay = []
        nodecay = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and 'time_' not in n:
                decay.append(p)
            else:
                nodecay.append(p)
        groups = [
            {'params': decay, 'weight_decay': self.args.weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        try:
            opt = torch.optim.AdamW(
                groups, lr=lr, betas=(0.9, self.args.beta2), eps=self.args.adam_eps,
                fused=(self.device_type == 'cuda'))
        except TypeError:
            opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, self.args.beta2), eps=self.args.adam_eps)
        return opt

    # ---------------- checkpoint disk
    def _ckpt_path(self, kind):
        return os.path.join(self.args.save_dir, f'{kind}.pt')

    def _state_dict(self, force_torch_state=True):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': dict(vocab_size=self.args.vocab_size, block_size=self.args.block_size,
                               n_layer=self.args.n_layer, n_embd=self.args.n_embd,
                               n_head=self.args.n_head),
            'config': vars(self.args),
            'iter_num': self.iter_num,
            'phase': self.phase,
            'phase_iter': self.phase_iter,
            'best_val': self.best_val,
            'rng': {
                'torch': torch.random.get_rng_state().tolist(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            },
            'prompts': getattr(self, 'prompts', None),
        }

    def save_ckpt(self, kind='ckpt_last', extra_msg=''):
        path = self._ckpt_path(kind)
        tmp = path + '.tmp'
        torch.save(self._state_dict(), tmp)
        os.replace(tmp, path)
        self.saved_any = True
        self._write_progress()
        print(f"[ckpt] {kind} saved (global_iter={self.iter_num}, phase={self.phase}, "
              f"lr={self._lr():.2e}) {extra_msg}", flush=True)

    def _write_progress(self):
        prog = {
            'phase': self.phase,
            'global_iter': self.iter_num,
            'phase_iter': self.phase_iter,
            'phase_iters': self.phases[self.phase]['iters'],
            'best_val': self.best_val,
            'tokens_processed': self.iter_num * self.tokens_per_iter,
            'elapsed_min': round((time.time() - self.start_wall) / 60, 1),
            'device': self.device,
        }
        with open(os.path.join(self.args.save_dir, 'progress.json'), 'w', encoding='utf-8') as f:
            json.dump(prog, f, ensure_ascii=False, indent=2)

    def _try_resume(self):
        path = self._ckpt_path('ckpt_last')
        if not os.path.exists(path):
            print("[resume] no ckpt_last.pt - starting from scratch", flush=True)
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        # force architecture from checkpoint
        ma = ck['model_args']
        self.model = build_model(vocab_size=ma['vocab_size'], block_size=ma['block_size'],
                                 n_layer=ma['n_layer'], n_embd=ma['n_embd'], n_head=ma['n_head'],
                                 dropout=self.args.dropout_a)
        st = ck['model']
        unwanted = '_orig_mod.'
        for k, v in list(st.items()):
            if k.startswith(unwanted):
                st[k[len(unwanted):]] = st.pop(k)
        self.model.load_state_dict(st, strict=False)
        self.model.to(self.device)
        ph = ck.get('phase', self.phase)
        if ph in self.phases:
            self.phase = ph
        else:
            idx = self.phase_order.index(ck.get('phase', self.phase_order[0])) if ck.get('phase') in self.phases else 0
            self.phase = self.phase_order[min(idx, len(self.phase_order) - 1)]
        self.iter_num = ck.get('iter_num', 0)
        self.phase_iter = ck.get('phase_iter', 0)
        self.best_val = ck.get('best_val', 1e9)
        rng = ck.get('rng')
        if rng:
            try:
                torch.random.set_rng_state(torch.tensor(rng.get('torch'), dtype=torch.uint8))
                np.random.set_state(rng.get('numpy'))
                random.setstate(rng.get('python'))
            except Exception as e:
                print(f"[warn] rng restore failed: {e}", flush=True)
        self.optimizer = self._make_optimizer(self.model, self.phases[self.phase]['lr_init'])
        try:
            self.optimizer.load_state_dict(ck['optimizer'])
        except Exception as e:
            print(f"[warn] optimizer state load failed: {e}", flush=True)
        print(f"[resume] ok: phase={self.phase}, iter={self.iter_num}, phase_iter={self.phase_iter}, "
              f"best_val={self.best_val:.4f}", flush=True)
        return True

    # ---------------- lr
    def _lr(self):
        ph = self.phases[self.phase]
        it = self.phase_iter
        tot = ph['iters']
        if it < ph['warmup_steps']:
            return ph['lr_init'] * (1 + it) / ph['warmup_steps']
        ratio = (it - ph['warmup_steps']) / max(1, tot - ph['warmup_steps'])
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0)))
        return ph['lr_final'] + coeff * (ph['lr_init'] - ph['lr_final'])

    def _advance_phase(self):
        idx = self.phase_order.index(self.phase)
        if idx + 1 < len(self.phase_order):
            self.phase = self.phase_order[idx + 1]
            self.phase_iter = 0
            # restart LR / optimizer continuity: keep optimizer, set per-phase dropout on model? dropout fixed; ok
            print(f"[phase] -> {self.phase} ({self.phases[self.phase]['name']})", flush=True)
            self.save_ckpt(kind='ckpt_last', extra_msg='phase-transition')
            self.pusher.maybe_push(time.time(), force=True, msg_suffix=f'phase-{self.phase}')
            return True
        return False

    # ---------------- main loop
    def run(self):
        budget = self.args.time_budget
        eval_every = self.args.eval_every
        log_every = self.args.log_every
        ckpt_every = self.args.ckpt_every
        if self.compile:
            print("[train] torch.compile warming up...", flush=True)

        # warmup a batch
        X, Y = self.train_mix.get_batch(self.args.batch_size, self.args.block_size, self.device)
        t0 = time.time()
        total_loss = 0.0
        local_iter = 0
        stop_reason = 'max-iter'

        while True:
            ph = self.phases[self.phase]

            # set lr + dropout for this phase
            lr = self._lr()
            for g in self.optimizer.param_groups:
                g['lr'] = lr

            # eval + samples + ckpt
            if self.iter_num % eval_every == 0 and local_iter >= 0:
                losses = {}
                vq = self.eval_iters
                for split in ['train', 'val']:
                    mix = self.train_mix if split == 'train' else self.val_mix
                    if not mix.items:
                        losses[split] = None
                        continue
                    model = self.model
                    model.eval()
                    acc = 0.0
                    for _k in range(vq):
                        Xv, Yv = mix.get_batch(4, 128, self.device)
                        with torch.no_grad(), self.ctx:
                            _, l = model(Xv, Yv)
                        acc += l.item()
                    losses[split] = acc / vq
                    model.train()
                vl = losses['val']
                print(f"[eval] step {self.iter_num} phase {self.phase}: train {losses['train'] if losses['train'] is not None else float('nan'):.4f} "
                      f"val {vl if vl is not None else float('nan'):.4f}", flush=True)
                if vl is not None:
                    if vl < self.best_val and self.iter_num > 0:
                        self.best_val = vl
                        self.save_ckpt(kind='ckpt_best', extra_msg=f'val={vl:.4f}')
                    self.history.append({'iter': self.iter_num, 'val': vl, 'phase': self.phase})
                if self.tokenizer is not None and self.args.prompts_file:
                    try:
                        self._write_samples()
                    except Exception as e:
                        print(f"[warn] sample eval failed: {e}", flush=True)
                if self.iter_num % ckpt_every == 0:
                    self.save_ckpt(kind='ckpt_last')
                    self.pusher.maybe_push(time.time(), msg_suffix=f'iter-{self.iter_num}')

            # termination / phase advance
            if self.phase_iter >= ph['iters']:
                if not self._advance_phase():
                    stop_reason = 'all-phases-done'
                    break
                ph = self.phases[self.phase]
                continue

            # time budget
            elapsed = time.time() - self.start_wall
            if budget and elapsed > budget:
                stop_reason = 'time-budget'
                self.save_ckpt(kind='ckpt_last')
                self.pusher.maybe_push(time.time(), force=True, msg_suffix='time-budget-stop')
                break

            # training step with grad accumulation
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro in range(self.args.grad_accum):
                X, Y = self.train_mix.get_batch(self.args.batch_size, self.args.block_size, self.device)
                with self.ctx:
                    _, loss = self.model(X, Y)
                    loss = loss / self.args.grad_accum
                loss.backward()
                step_loss += loss.item()
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[warn] loss spike {loss.item()}, skipping step", flush=True)
                    self.optimizer.zero_grad(set_to_none=True)
                    break
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % log_every == 0:
                tps = self.tokens_per_iter / max(dt, 1e-9)
                print(f"[iter {self.iter_num}] phase {self.phase} ({self.phase_iter}/{ph['iters']}) "
                      f"loss {step_loss:.4f} lr {lr:.2e} {tps:.0f} tok/s dt {dt*1000:.0f}ms", flush=True)
                line = f"{self.iter_num}\t{self.phase}\t{step_loss:.4f}\t{lr:.2e}\t{tps:.0f}\t{dt*1000:.0f}\n"
                with open(os.path.join(self.args.save_dir, 'train_log.txt'), 'a', encoding='utf-8') as f:
                    f.write(line)

            self.iter_num += 1
            self.phase_iter += 1
            local_iter += 1

        # final save + push
        self.save_ckpt(kind='ckpt_last', extra_msg=f'done-{stop_reason}')
        self.pusher.maybe_push(time.time(), force=True, msg_suffix=f'done-{stop_reason}')
        print(f"[done] {stop_reason} at global_iter={self.iter_num}, elapsed={(time.time()-self.start_wall)/60:.1f}min", flush=True)

    def _write_samples(self):
        prompts = []
        if os.path.exists(self.args.prompts_file):
            with open(self.args.prompts_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        prompts.append(line)
        if not prompts:
            prompts = [
                "Вопрос: Какой камень лучше всего подходит для защиты дома\nОтвет:",
                "Вопрос: Что означает сон про чёрную кошку\nОтвет:",
                "Вопрос: Расскажи про пирамиду Кукулькана\nОтвет:",
                "Вопрос: Какой ритуал проводят в полнолуние\nОтвет:",
                "Вопрос: Что символизирует руна Одал\nОтвет:",
            ]
        self.prompts = prompts
        try:
            text = generate_samples(self.model, self.tokenizer, prompts, self.device,
                                    max_new=140, temperature=0.8, top_k=40)
        except Exception as e:
            text = f"sample-eval-error: {e}"
        with open(os.path.join(self.args.save_dir, 'eval_samples.txt'), 'a', encoding='utf-8') as f:
            f.write(f"\n=== step {self.iter_num} phase {self.phase} ==="
                    f" ({time.strftime('%Y-%m-%d %H:%M:%S')})\n{text}\n")
        print(f"[samples] step {self.iter_num} -> eval_samples.txt", flush=True)


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Mystic RWKV-5.2 11M trainer')
    # model
    p.add_argument('--vocab-size', type=int, default=3000)
    p.add_argument('--block-size', type=int, default=512)
    p.add_argument('--n-layer', type=int, default=13)
    p.add_argument('--n-embd', type=int, default=256)
    p.add_argument('--n-head', type=int, default=8)
    p.add_argument('--dropout-a', type=float, default=0.02)
    p.add_argument('--dropout-b', type=float, default=0.05)
    # data
    p.add_argument('--train-narr', type=str, default=None)
    p.add_argument('--train-qa', type=str, default=None)
    p.add_argument('--val-narr', type=str, default=None)
    p.add_argument('--val-qa', type=str, default=None)
    p.add_argument('--tokenizer-path', type=str, default=None)
    p.add_argument('--prompts-file', type=str, default=None)
    # phases
    p.add_argument('--qa-epochs-a', type=float, default=1.0)
    p.add_argument('--epochs-b', type=float, default=3.0)
    p.add_argument('--phase-c', action='store_true')
    p.add_argument('--narr-frac-c', type=float, default=0.9)
    # lr
    p.add_argument('--lr-a-init', type=float, default=3e-4)
    p.add_argument('--lr-a-final', type=float, default=3e-5)
    p.add_argument('--lr-b-init', type=float, default=5e-5)
    p.add_argument('--lr-b-final', type=float, default=1e-5)
    p.add_argument('--warmup-a', type=float, default=0.015)
    p.add_argument('--warmup-b', type=float, default=0.02)
    # optimizer
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--adam-eps', type=float, default=1e-8)
    p.add_argument('--grad-clip', type=float, default=1.0)
    # run
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--ckpt-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--eval-iters', type=int, default=40)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float32', 'bfloat16', 'float16'])
    p.add_argument('--compile', action='store_true')
    p.add_argument('--save-dir', type=str, default='ckpt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--time-budget', type=float, default=0.0)
    # push / kaggle
    p.add_argument('--auto-push', action='store_true')
    p.add_argument('--kaggle-user', type=str, default='splendor1')
    p.add_argument('--ckpt-dataset', type=str, default='mystic-rwkv-checkpoints')
    p.add_argument('--push-every', type=int, default=1800)
    return p.parse_args(argv)


def train_mystic_main(argv=None):
    args = parse_args(argv)
    tr = Trainer(args)
    tr.run()


if __name__ == '__main__':
    sys.exit(train_mystic_main())