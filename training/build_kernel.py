"""
Assemble the self-contained Kaggle script kernel `mystic-rwkv-train.py`.

The kernel is a concatenation of:
    rwkv_model.py   (model, import-tolerant)
    train_mystic.py (trainer, tail main-guard stripped, import-tolerant)
    kernel_main.py  (Kaggle runtime: data resolution, resume pull, entrypoint)

Kaggle 'script' kernels are a single .py upload, so everything must live in one file.
Run:  python build_kernel.py [--out mystic-rwkv-train.py]
"""

import argparse
import os

BLOCKS = ["rwkv_model.py", "train_mystic.py", "kernel_main.py"]
TRAIN_TAIL = 'if __name__ == \'__main__\':\n    sys.exit(train_mystic_main())'


def split_at_tail(text, marker):
    i = text.rfind(marker)
    if i == -1:
        raise SystemExit(f"[build] tail marker not found in train_mystic.py:\n{marker!r}")
    return text[:i]


def assemble():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='mystic-rwkv-train.py')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    parts = []
    parts.append("# ==== FILE: rwkv_model.py ====\n")
    with open(os.path.join(here, "rwkv_model.py"), encoding="utf-8") as f:
        parts.append(f.read())


    parts.append("\n# ==== FILE: train_mystic.py (tail-guard stripped) ====\n")
    with open(os.path.join(here, "train_mystic.py"), encoding="utf-8") as f:
        parts.append(split_at_tail(f.read(), TRAIN_TAIL))

    parts.append("\n# ==== FILE: kernel_main.py ====\n")
    with open(os.path.join(here, "kernel_main.py"), encoding="utf-8") as f:
        parts.append(f.read())

    out = "".join(parts)
    out_path = os.path.join(here, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[build] wrote {out_path} ({len(out.splitlines())} lines, {len(out)} bytes)")


if __name__ == "__main__":
    assemble()