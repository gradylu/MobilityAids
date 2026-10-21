# AutoDL 云端全量训练指南

本地 CPU 跑不动 1.5 万张图，上云 GPU（AutoDL）全量训练。完整步骤：

## 1. 租机器

1. 注册 [AutoDL](https://www.autodl.com/) 并充值（几块钱就够）
2. 租实例：**RTX 4090（24G）** 或 3090，选离你近的区域，镜像选 **PyTorch**（带 2.x 的）
3. 启动后点 **JupyterLab** 进入

## 2. 上传数据（3 个东西）

传到 `~/autodl-tmp/`（数据盘，别放系统盘）：

| 内容 | 大小 | 说明 |
|---|---|---|
| `dataset/` | 10.63 GB | 转好的 YOLO 数据集（images + labels + data.yaml）|
| `yolo26n.pt` | 5.3 MB | 预训练权重（本地已下好，云上不用再下 GitHub）|
| `train.py` | — | 训练脚本 |

上传方式（任选）：
- **JupyterLab 网页拖拽**：支持断点续传，10GB 约 10-30 分钟
- 或先打包 `tar czf dataset.tar.gz dataset/` 再传一个文件
- 或 `scp -P <端口> -r dataset/ root@<地址>:/root/autodl-tmp/`

## 3. 一键训练

在 JupyterLab 终端里：
```bash
cd ~/autodl-tmp
bash train_autodl.sh
```

脚本会自动：开学术加速 → 装 ultralytics → 修正 data.yaml 路径 → 100 epoch 训练。

## 4. 取结果

```bash
# 权重（可直接部署）
~/autodl-tmp/runs/mobilityaids_full/weights/best.pt

# 导出 ONNX 部署
python -c "from ultralytics import YOLO; YOLO('runs/mobilityaids_full/weights/best.pt').export(format='onnx')"
```

## 参数调优提示

- **batch**：4090 24G 可以开到 32（更快），显存不够再降到 16
- **epochs**：100 是保守值，可看 val mAP 收敛情况决定是否加
- **中断续训**：加 `--resume`（需在 train.py 加 resume 参数，或用 ultralytics 原生命令）
- **多卡**：`--device 0,1`（需改 batch 为 2 的倍数）

## 预期结果

全量 100 epoch 训练后，val mAP50 预计 **0.8+**（当前 250 张 × 2 epoch 的 smoke test 是 0.42，只是占位）。
