#!/usr/bin/env python
"""一键下载 MobilityAids 数据集（OpenDataLab）。

只下 RGB 检测需要的 3 个文件（跳过约 5GB 的 depth 包）：
  - ImageSets.zip        (304 KB)    train/val 划分列表
  - Images_RGB.zip       (12.4 GB)   RGB 图像
  - Annotations_RGB.zip  (7.6 MB)    Pascal VOC YAML 标注

用法:
  python download_dataset.py [--target datasets]

前提:
  - 已安装 openxlab:  pip install openxlab
  - 已登录:           openxlab login
      （AK/SK 获取：https://sso.openxlab.org.cn/usercenter?tab=secret）

下载完成后，接着运行 scripts/convert_dataset.py 转成 YOLO 格式。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "OpenDataLab/MobilityAids"
FILES = [
    ("/raw/ImageSets.zip", "304 KB", "train/val 划分列表"),
    ("/raw/Images_RGB.zip", "12.4 GB", "RGB 图像"),
    ("/raw/Annotations_RGB.zip", "7.6 MB", "Pascal VOC YAML 标注"),
]


def run(cmd):
    print("+ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("\n⚠️ 下载失败。若提示 401 / login required，说明未登录，请先执行：", file=sys.stderr)
        print("    openxlab login   # AK/SK: https://sso.openxlab.org.cn/usercenter?tab=secret", file=sys.stderr)
        sys.exit(r.returncode)


def find(target: Path, name: str) -> Path | None:
    """在 target 下递归查找指定文件（openxlab 落盘目录层级可能不同）。"""
    if not target.exists():
        return None
    for p in target.rglob(name):
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="datasets", help="下载目标父目录（默认 datasets/）")
    args = ap.parse_args()

    if shutil.which("openxlab") is None:
        sys.exit(
            "未找到 openxlab，请先安装并登录：\n"
            "  pip install openxlab\n"
            "  openxlab login   # AK/SK: https://sso.openxlab.org.cn/usercenter?tab=secret"
        )

    print(f"开始下载 {REPO} 的 {len(FILES)} 个文件（跳过 depth 包）\n")
    for path, size, desc in FILES:
        print(f"▶ {path}  ({size}, {desc})")
        run([
            "openxlab", "dataset", "download",
            "--dataset-repo", REPO,
            "--source-path", path,
            "--target-path", args.target,
        ])
        print()

    print("=" * 60)
    print("下载完成，校验文件：")
    target = Path(args.target)
    found = True
    for path, _, _ in FILES:
        name = Path(path).name
        p = find(target, name)
        if p:
            print(f"  ✓ {p}  ({p.stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
        else:
            found = False
            print(f"  ✗ {name} 未找到（请检查 openxlab 实际输出目录）")
    print("=" * 60)

    if not found:
        sys.exit("部分文件未下载成功，请检查上面的错误信息。")

    print("\n下一步：转成 YOLO 格式")
    print("  venv/Scripts/python scripts/convert_dataset.py \\")
    print("      --raw-dir datasets/OpenDataLab___MobilityAids/raw --out-dir dataset")
    print("\n（若 raw 目录不在上述位置，用 --raw-dir 指向实际包含 3 个 zip 的目录）")


if __name__ == "__main__":
    main()
