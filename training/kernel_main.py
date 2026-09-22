"""
Kaggle runtime for Mystic RWKV-5.2 11M (concatenation tail of the script kernel).

Concatenated after rwkv_model.py + train_mystic.py, so these names are live:
    RWKV, build_model, Trainer, parse_args, train_mystic_main
Flow:
    1. locate inputs (dataset mount or CLI download)
    2. verify data sizes
    3. fresh resume: pull latest checkpoint from ckpt dataset via CLI
    4. run phases A(+B+C) under wall-clock budget; checkpoints auto-pushed
"""

import os
import shutil
import subprocess
import sys

KAGGLE_USER = "splendor1"
DATA_DATASET = "mystic-rwkv-data"
CKPT_DATASET = "mystic-rwkv-checkpoints"

# expected sizes (bytes) of the uint16 .bin files committed in the data bundle
EXPECTED = {
    "narr_train.bin": 439_327_828,
    "narr_val.bin": 4_437_654,
    "qa_train.bin": 15_050_336,
    "qa_val.bin": 41_238,
}

REQUIRED_FILES = list(EXPECTED) + ["tokenizer.json", "prompts.txt"]


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, **kw)


def find_data_dir():
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            have = [f for f in REQUIRED_FILES if os.path.exists(os.path.join(path, f))]
            if len(have) >= len(REQUIRED_FILES):
                print(f"[main] dataset mount: {path} ({len(have)} files)", flush=True)
                return path
    return None


def download_cli(dataset, dest):
    os.makedirs(dest, exist_ok=True)
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", f"{KAGGLE_USER}/{dataset}",
         "-p", dest, "--unzip", "--force"],
        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"[main] CLI download of {dataset} FAILED: {r.stderr[:400]}", flush=True)
        return {}
    files = {}
    for root, _, fs in os.walk(dest):
        for f in fs:
            files[f] = os.path.join(root, f)
    print(f"[main] pulled {dataset}: {sorted(files)}", flush=True)
    return files


def setup_environment():
    try:
        import torch
        gpu = ''
        if torch.cuda.is_available():
            gpu = f" {torch.cuda.get_device_name(0)} vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
        print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()}{gpu}", flush=True)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "torch"])
    try:
        import tokenizers  # noqa
        print("[env] tokenizers ok", flush=True)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"])
    # NOTE: bundled kaggle CLI 2.0.2 is buggy for 'list --page-size' but its
    # 'datasets version -p' WORKS. Do NOT upgrade: newer CLI (2.2.x) fails pushes
    # with HTTP 400 on CreateDatasetVersion (observed 19.09.2026).
    try:
        import numpy
        print(f"[env] numpy {numpy.__version__}", flush=True)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "numpy"])


def kaggle_main():
    setup_environment()

    work = "/kaggle/working"
    os.makedirs(work, exist_ok=True)

    data_dir = find_data_dir()
    if data_dir is None:
        files = download_cli(DATA_DATASET, os.path.join(work, "data"))
        data_dir = os.path.join(work, "data") if files else None
    if data_dir is None:
        print("[main] FATAL: no data found", flush=True)
        sys.exit(2)

    paths = {f: os.path.join(data_dir, f) for f in REQUIRED_FILES}
    missing = [f for f in REQUIRED_FILES if not os.path.exists(paths[f])]
    if missing:
        print(f"[main] FATAL: missing inputs {missing}", flush=True)
        sys.exit(2)

    for f, want in EXPECTED.items():
        got = os.path.getsize(paths[f])
        if got != want:
            print(f"[main] FATAL: {f} size {got} != expected {want}", flush=True)
            sys.exit(3)
    print("[main] data sizes OK", flush=True)

    save_dir = os.path.join(work, "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_files = download_cli(CKPT_DATASET, os.path.join(work, "ckpt_pull"))
    pulled = ckpt_files.get("ckpt_last.pt") or ckpt_files.get("ckpt_best.pt")
    if pulled:
        shutil.copyfile(pulled, os.path.join(save_dir, "ckpt_last.pt"))
        print(f"[main] resuming from checkpoint: {pulled}", flush=True)
    else:
        print("[main] no remote checkpoint yet - training from scratch", flush=True)

    argv = [
        "--train-narr", paths["narr_train.bin"],
        "--train-qa", paths["qa_train.bin"],
        "--val-narr", paths["narr_val.bin"],
        "--val-qa", paths["qa_val.bin"],
        "--tokenizer-path", paths["tokenizer.json"],
        "--prompts-file", paths["prompts.txt"],
        "--save-dir", save_dir,
        "--resume",
        "--device", "auto",
        "--dtype", "auto",
        "--time-budget", os.environ.get("MYSTIC_TIME_BUDGET", "37800"),
        "--qa-epochs-a", "1.0",
        "--epochs-b", "3.0",
        "--phase-c",
        "--narr-frac-c", "0.9",
        "--lr-a-init", "3e-4",
        "--lr-a-final", "3e-5",
        "--lr-b-init", "5e-5",
        "--lr-b-final", "1e-5",
        "--batch-size", "8",
        "--grad-accum", "4",
        "--eval-every", "100",
        "--ckpt-every", "100",
        "--log-every", "25",
        "--auto-push",
        "--kaggle-user", KAGGLE_USER,
        "--ckpt-dataset", CKPT_DATASET,
        "--push-every", "1200",
    ]
    sys.exit(train_mystic_main(argv))


if __name__ == "__main__":
    kaggle_main()