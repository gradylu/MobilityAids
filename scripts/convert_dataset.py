#!/usr/bin/env python
"""MobilityAids (Pascal VOC YAML) → YOLO 格式转换。

用法:
  # 全量转换（15278 张，约 10.6GB）
  python convert_dataset.py \
      --raw-dir datasets/OpenDataLab___MobilityAids/raw \
      --out-dir dataset

  # smoke test 子集（本地快速验证，只取前 N 张）
  python convert_dataset.py --max-train 200 --max-val 50

需要 raw-dir 下存在 3 个源文件（从 OpenDataLab 下载，见 README「数据准备」）:
  - ImageSets.zip        数据划分（train/val 列表）
  - Images_RGB.zip       RGB 图像
  - Annotations_RGB.zip  Pascal VOC YAML 标注

产物（out-dir 下）:
  images/train/*.png  images/val/*.png
  labels/train/*.txt  labels/val/*.txt
  data.yaml
"""
import argparse
import zipfile
from pathlib import Path

import yaml

CLASS_NAMES = ["person", "wheelchair", "push_wheelchair", "crutches", "walking_frame"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

TRAIN_SETS = ["ImageSets/TrainSet_RGB.txt", "ImageSets/TrainAdditional_InOutDoor.txt"]
VAL_SETS = [
    "ImageSets/TestSet1.txt",
    "ImageSets/TestSet2_seq1.txt",
    "ImageSets/TestSet2_seq2.txt",
    "ImageSets/TestSet2_seq3.txt",
    "ImageSets/TestSet2_seq4.txt",
]


def load_split(raw_dir: Path) -> dict:
    split = {}
    with zipfile.ZipFile(raw_dir / "ImageSets.zip") as z:
        for f in TRAIN_SETS:
            for line in z.read(f).decode().split():
                split[line.strip()] = "train"
        for f in VAL_SETS:
            for line in z.read(f).decode().split():
                split[line.strip()] = "val"
    return split


def parse_annotation(text: str) -> tuple[str, list]:
    data = yaml.safe_load(text)
    ann = data["annotation"]
    w = float(ann["size"]["width"])
    h = float(ann["size"]["height"])
    objs = ann.get("object", [])
    if objs is None:
        objs = []
    elif isinstance(objs, dict):
        objs = [objs]
    boxes = []
    for o in objs:
        name = o["name"].strip()
        if name not in CLASS_ID:
            continue
        bb = o["bndbox"]
        xmin = float(bb["xmin"])
        ymin = float(bb["ymin"])
        xmax = float(bb["xmax"])
        ymax = float(bb["ymax"])
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        boxes.append((CLASS_ID[name], cx, cy, bw, bh))
    return ann["filename"], boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir",
        default="datasets/OpenDataLab___MobilityAids/raw",
        help="OpenDataLab 下载的原始数据目录（含 ImageSets.zip / Images_RGB.zip / Annotations_RGB.zip）",
    )
    ap.add_argument("--out-dir", default="dataset", help="转换产物输出目录")
    ap.add_argument("--max-train", type=int, default=None, help="只取前 N 张训练图（smoke test 用）")
    ap.add_argument("--max-val", type=int, default=None, help="只取前 N 张验证图（smoke test 用）")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    split = load_split(raw_dir)
    limits = {"train": args.max_train, "val": args.max_val}
    counts = {"train": 0, "val": 0}

    for s in ("train", "val"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    img_zip = zipfile.ZipFile(raw_dir / "Images_RGB.zip")
    ann_zip = zipfile.ZipFile(raw_dir / "Annotations_RGB.zip")

    ann_entries = [n for n in ann_zip.namelist() if n.endswith(".yml")]
    skipped = 0
    done = 0
    for ann_path in ann_entries:
        done += 1
        if done % 2000 == 0:
            print(f"  进度: {done}/{len(ann_entries)}  已写入 train={counts['train']} val={counts['val']}", flush=True)
        txt = ann_zip.read(ann_path).decode("utf-8", errors="replace")
        filename, boxes = parse_annotation(txt)
        stem = filename[:-4]  # 去掉 .png
        s = split.get(stem, "train")
        if limits[s] is not None and counts[s] >= limits[s]:
            continue
        try:
            img_bytes = img_zip.read(f"Images_RGB/{filename}")
        except KeyError:
            skipped += 1
            continue
        label = "".join(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cid, cx, cy, bw, bh in boxes)
        (out_dir / "labels" / s / f"{stem}.txt").write_text(label, encoding="utf-8")
        (out_dir / "images" / s / filename).write_bytes(img_bytes)
        counts[s] += 1

    img_zip.close()
    ann_zip.close()

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    print(f"完成: train={counts['train']}, val={counts['val']}, 跳过(缺图)={skipped}")
    print(f"data.yaml → {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
