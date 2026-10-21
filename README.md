# MobilityAids + YOLO26n 微调

YOLO26n 在 MobilityAids 数据集（5 类：行人/轮椅/推轮椅/拄拐/助行架）上微调。

## 已完成（本地 smoke test ✅）

- 数据转换脚本 + 训练脚本已跑通
- 250 张子集（200 train + 50 val）CPU 短训练验证通过，best.pt 已生成
- 依赖装在独立 venv（`venv/`，torch 2.14.1 CPU + ultralytics 8.4.180）

## 目录

```
yolo-finetune/
├── venv/                          # 独立环境（CPU 版）
├── scripts/
│   ├── convert_dataset.py         # Pascal VOC YAML → YOLO txt
│   └── train.py                   # YOLO26n 微调
├── dataset/                       # YOLO 格式数据集（当前为 250 张子集）
│   ├── images/train,val  labels/train,val  data.yaml
├── yolo26n.pt                     # 预训练权重（已下载）
└── runs/.../weights/best.pt       # 训练产物
```

## 一、全量数据转换（本地）

```bash
cd /e/hermes_data/yolo-finetune
venv/Scripts/python.exe scripts/convert_dataset.py
# 产出约 15278 张图 + 标签到 dataset/（约 3GB，几分钟）
```

## 二、云端全量微调（推荐）

本地 CPU 跑不动 1.5 万张图，走云 GPU（AutoDL 等）。

**1. 租 GPU**：AutoDL 选 RTX 4090（约 1.5 元/时）或 3090，PyTorch 镜像。

**2. 上传数据**：把 `dataset/`（转好的 ~3GB）传到云（AutoDL 用 JupyterLab 拖拽或网盘/`scp`）。

**3. 安装 + 训练**（云端，GPU）：
```bash
pip install ultralytics
python scripts/train.py --data dataset/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --device 0 --name mobilityaids_full
```

> 云上 GPU 下载 yolo26n.pt 同样走 GitHub，可能慢；可本地先下好 `yolo26n.pt`（已下）一起传上去，训练时 `--model /path/yolo26n.pt`。

**4. 结果**：`runs/.../weights/best.pt`（可 `yolo export format=onnx` 部署）。

## 三、本地推理测试

```bash
venv/Scripts/python.exe -c "
from ultralytics import YOLO
m = YOLO('runs/detect/runs/mobilityaids_yolo26n/weights/best.pt')
r = m.predict('dataset/images/val', imgsz=320, save=True)
"
```

## 参考指标（250 张子集、仅 2 epoch，仅供参考）

mAP50: all 0.423 / person 0.678 / wheelchair 0.622 / push_wheelchair 0.434 / crutches 0.231 / walking_frame 0.149

> 全量 100 epoch 训练后 mAP 会大幅提升（预计 all mAP50 可达 0.8+）。
