"""
Build the Kaggle data bundle folder `mystic-rwkv-data/` (a private dataset to push).

Copies the 4 uint16 .bin + tokenizer + prompts, and writes data_report.json
(hash+size+token counts) so notebook integrity checks and humans can compare.

Run:  python build_data_bundle.py [--out mystic-rwkv-data]
"""

import argparse
import hashlib
import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FILES = {
    "narr_train.bin": os.path.join(ROOT, "Синтетика от Соннет 5", "train.bin"),
    "narr_val.bin": os.path.join(ROOT, "Синтетика от Соннет 5", "val.bin"),
    "qa_train.bin": os.path.join(ROOT, "data_unified_vocab3000", "qa_train.bin"),
    "qa_val.bin": os.path.join(ROOT, "data_unified_vocab3000", "qa_val.bin"),
    "tokenizer.json": os.path.join(ROOT, "Синтетика от Соннет 5", "tokenizer.json"),
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for name, src in FILES.items():
        assert os.path.exists(src), f"missing source: {src}"
        dst = os.path.join(out_dir, name)
        shutil.copyfile(src, dst)
        size = os.path.getsize(dst)
        entry = {"source": src, "bytes": size}
        if name.endswith(".bin"):
            entry["tokens"] = size // 2
            entry["md5"] = md5(dst)
        report[name] = entry
        print(f"[bundle] {name}: {size/1e6:.1f} MB, {entry['tokens'] if 'tokens' in entry else '-'} tokens")

    shutil.copyfile(os.path.join(os.path.dirname(__file__), "prompts.txt"),
                    os.path.join(out_dir, "prompts.txt"))

    meta = {
        "id": "splendor1/mystic-rwkv-data",
        "title": "Mystic RWKV data (unified vocab 3000)",
        "subtitle": "narr(219.7M)+qa(7.5M) uint16 bins + tokenizer for RWKV-5.2 11M pretrain/SFT",
        "isPrivate": True,
        "licenses": [{"name": "MIT"}],
    }
    with open(os.path.join(out_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[bundle] wrote {os.path.join(out_dir, 'data_report.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mystic-rwkv-data")
    args = ap.parse_args()
    build(os.path.abspath(args.out))