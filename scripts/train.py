#!/usr/bin/env python
"""YOLO26n 微调脚本（本地 smoke test / 云端全量通用）。

用法:
  # 本地 smoke test（CPU，子集，几轮）
  python train.py --data dataset/data.yaml --epochs 2 --imgsz 320 --batch 4 --device cpu

  # 云端全量（GPU）
  python train.py --data dataset/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu / 0 / 0,1")
    ap.add_argument("--name", default="mobilityaids_yolo26n")
    ap.add_argument("--project", default="runs")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        workers=0 if args.device == "cpu" else 8,  # Windows CPU 训练 workers 必须为 0，避免多进程报错
    )


if __name__ == "__main__":
    main()
