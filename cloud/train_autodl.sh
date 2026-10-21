#!/usr/bin/env bash
# =============================================================
# AutoDL 一键训练脚本：MobilityAids + YOLO26n 全量微调
#
# 前提：已把以下内容传到 ~/autodl-tmp/
#   - dataset/       (转好的 YOLO 数据集，含 data.yaml)
#   - yolo26n.pt     (预训练权重，本地已下好，避免云上 GitHub 慢)
#   - train.py       (训练脚本)
#
# 运行：bash train_autodl.sh
# =============================================================
set -e

# 0. 学术资源加速（加速 GitHub/PyPI，可选，失败不阻断）
source /etc/network_turbo 2>/dev/null || true

# 1. 安装 ultralytics（清华镜像，快）
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 路径约定
BASE="$HOME/autodl-tmp"
DATA_DIR="$BASE/dataset"
DATA_YAML="$DATA_DIR/data.yaml"

# 3. 修正 data.yaml 的 path 为云端绝对路径（本地转出来的是 Windows 路径，Linux 上无效）
python - "$DATA_YAML" "$DATA_DIR" <<'PY'
import sys, yaml
p, data_dir = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(p, encoding="utf-8"))
d["path"] = data_dir
yaml.safe_dump(d, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print("data.yaml path 已修正为:", d["path"])
PY

# 4. 训练（100 epoch，GPU 0）
cd "$BASE"
python train.py \
    --data "$DATA_YAML" \
    --model "$BASE/yolo26n.pt" \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --project "$BASE/runs" \
    --name mobilityaids_full

echo "=========================================="
echo "训练完成！权重在："
echo "  $BASE/runs/mobilityaids_full/weights/best.pt"
echo "=========================================="
