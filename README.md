# MobilityAids + YOLO26n 微调

在 [MobilityAids](https://opendatalab.com/OpenDataLab/MobilityAids) 数据集上微调 YOLO26n，做 5 类目标检测（行人 + 移动辅助设备）：

| id | 类别 |
|---|---|
| 0 | person（行人）|
| 1 | wheelchair（坐轮椅的人）|
| 2 | push_wheelchair（推轮椅的人）|
| 3 | crutches（拄拐的人）|
| 4 | walking_frame（用助行架的人）|

数据来源：弗莱堡大学 [MobilityAids](http://mobility-aids.informatik.uni-freiburg.de/)（论文 [arXiv:1708.00674](https://arxiv.org/pdf/1708.00674v1.pdf)，数据集许可 MobilityAids Custom）。

> ⚠️ 本仓库**不含数据集**（约 10.6GB）。请按下面「数据准备」步骤自行生成 `dataset/`。

---

## 目录

```
yolo-finetune/
├── scripts/
│   ├── convert_dataset.py   # 原始数据 → YOLO 格式
│   ├── make_subset.py       # 抽取子集（本地试训用）
│   └── train.py             # YOLO26n 微调
├── cloud/
│   ├── train_autodl.sh      # AutoDL 云端一键训练
│   └── README.md            # 上云指南
├── dataset/                 # （生成物，不入库）YOLO 格式数据集
└── README.md
```

---

## 一、环境准备

Python 3.10+，建议独立 venv：

```bash
python -m venv venv
# Windows
venv/Scripts/pip install ultralytics pyyaml
# Linux / macOS
venv/bin/pip install ultralytics pyyaml
```

> `convert_dataset.py` 只需 `pyyaml`；训练需 `ultralytics`（会自动装 torch）。
> 本地 CPU 训练在 Windows 上需 torch 的 CPU 版（ultralytics 默认装的就是 CPU 版）。

---

## 二、数据准备（制作 dataset/）

数据集**不入库**，需要自己下载原始数据并转换成 YOLO 格式。

### 2.1 下载原始数据

数据托管在 OpenDataLab：**`OpenDataLab/MobilityAids`**
- 网页：https://opendatalab.com/OpenDataLab/MobilityAids
- 原始官网（免登录，国内可能慢）：http://mobility-aids.informatik.uni-freiburg.de/

RGB 检测**只需要 3 个文件**（深度图用不到，可跳过约 5GB 的 depth 包）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `raw/ImageSets.zip` | 304 KB | train/val 划分列表 |
| `raw/Images_RGB.zip` | 12.4 GB | RGB 图像 |
| `raw/Annotations_RGB.zip` | 7.6 MB | Pascal VOC YAML 标注 |

**下载方式 ⓪ — 一键脚本（推荐）：**

```bash
venv/Scripts/python scripts/download_dataset.py --target datasets
```

脚本自动拉这 3 个文件（跳过 depth 包）、下完校验，并提示下一步。前提是已装好并登录 openxlab（见下）。

**下载方式 ① — openxlab CLI 手动（需 AK/SK）：**

```bash
pip install openxlab
openxlab login    # 粘贴 Access Key / Secret Key
# AK/SK 获取：https://sso.openxlab.org.cn/usercenter?tab=secret

# 只下需要的 3 个文件（跳过 depth，省约 5GB）：
openxlab dataset download --dataset-repo "OpenDataLab/MobilityAids" \
    --source-path "/raw/ImageSets.zip" --target-path datasets
openxlab dataset download --dataset-repo "OpenDataLab/MobilityAids" \
    --source-path "/raw/Images_RGB.zip" --target-path datasets
openxlab dataset download --dataset-repo "OpenDataLab/MobilityAids" \
    --source-path "/raw/Annotations_RGB.zip" --target-path datasets
```

**下载方式 ② — 原始官网（免登录）：** 从 http://mobility-aids.informatik.uni-freiburg.de/ 下载对应的 `ImageSets.zip`、`Images_RGB.zip`、`Annotations_RGB.zip`。

下载后，把这 3 个 zip 放到一起，目录结构应如下（默认 `--raw-dir` 路径）：

```
datasets/OpenDataLab___MobilityAids/raw/
├── ImageSets.zip
├── Images_RGB.zip
└── Annotations_RGB.zip
```

> 如果数据放在别处，转换时用 `--raw-dir <实际路径>` 指定即可。

### 2.2 转换格式（Pascal VOC → YOLO）

```bash
venv/Scripts/python scripts/convert_dataset.py \
    --raw-dir datasets/OpenDataLab___MobilityAids/raw \
    --out-dir dataset
```

脚本遍历 `Annotations_RGB.zip` 里的 YAML 标注，按 `ImageSets.zip` 的划分，把坐标转成 YOLO 归一化格式。产出 **15278 张**：

```
dataset/
├── images/train/   # 10961 张
├── images/val/     # 4317 张
├── labels/train/   # 10961 个 txt
├── labels/val/     # 4317 个 txt
└── data.yaml       # path / train / val / names
```

总大小约 **10.6 GB**，几分钟～十几分钟跑完（取决于磁盘速度）。

### 2.3 抽子集（可选，本地 CPU 快速试训）

全量在 CPU 上很慢，试训可先抽小集合：

```bash
venv/Scripts/python scripts/make_subset.py --train 1000 --val 250 --out dataset_1000
```

---

## 三、训练

### 本地 CPU（小规模 / 试训）

```bash
venv/Scripts/python scripts/train.py \
    --data dataset/data.yaml --model yolo26n.pt \
    --epochs 30 --imgsz 320 --batch 8 --device cpu \
    --name mobilityaids_full_30ep
```

> `--model yolo26n.pt`：若本地没有，ultralytics 会自动下载（国内网络慢，可先手动下好）。全量 1.5 万张在 CPU 上约 14 分钟/epoch，正式训练建议上云 GPU。

### 云端 GPU（AutoDL，推荐）

见 `cloud/README.md`，或直接：

```bash
bash cloud/train_autodl.sh
```

---

## 四、推理测试

```bash
venv/Scripts/python -c "
from ultralytics import YOLO
m = YOLO('runs/mobilityaids_full_30ep/weights/best.pt')
m.predict('dataset/images/val', imgsz=320, save=True)
"
```

导出 ONNX 部署：

```bash
venv/Scripts/python -c "from ultralytics import YOLO; YOLO('runs/.../weights/best.pt').export(format='onnx')"
```

---

## 参考指标

| 配置 | mAP50 (all) |
|---|---|
| 250 张 × 2 epoch（smoke test）| 0.423 |
| 1000 张 × 10 epoch（imgsz 320, CPU）| 0.917 |
| 全量 15278 张 × 30 epoch（imgsz 320, CPU）| 训练中… |
