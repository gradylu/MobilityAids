#!/usr/bin/env python
"""从已转好的全量数据集里抽取随机子集（用于本地 CPU 试训，无需重新解压 zip）。

用法:
  python make_subset.py --train 1000 --val 250 --out dataset_1000
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=250)
    ap.add_argument("--out", default="dataset_subset")
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    dst = Path(args.out)

    for split, n in (("train", args.train), ("val", args.val)):
        imgs = list((src / "images" / split).glob("*.png"))
        random.shuffle(imgs)
        picked = imgs[:n]
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im in picked:
            stem = im.stem
            shutil.copy(im, dst / "images" / split / im.name)
            shutil.copy(src / "labels" / split / f"{stem}.txt", dst / "labels" / split / f"{stem}.txt")
        print(f"{split}: {len(picked)} 张")

    d = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))
    d["path"] = str(dst.resolve())
    yaml.safe_dump(d, open(dst / "data.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(f"子集已生成 → {dst.resolve()}（{args.train}+{args.val} 张）")


if __name__ == "__main__":
    main()
