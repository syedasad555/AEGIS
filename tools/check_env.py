#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
环境检查工具 — 检查 Python、PyTorch、Ultralytics 等依赖版本。

Usage:
    python tools/check_env.py
"""
import sys
import torch
import psutil
import platform
from importlib.metadata import version as get_version


def print_environment():
    print("=" * 50)
    print("🔧 硬件环境信息")
    print("=" * 50)

    # CPU信息 (Windows兼容版)
    try:
        print(f"CPU型号: {platform.processor()}")
    except Exception as e:
        print(f"CPU型号: 获取失败: {e}")
    print(f"CPU核心数: {psutil.cpu_count(logical=False)} (物理) / {psutil.cpu_count(logical=True)} (逻辑)")

    # 内存信息
    mem = psutil.virtual_memory()
    print(f"总内存: {mem.total / (1024**3):.2f} GB")
    print(f"可用内存: {mem.available / (1024**3):.2f} GB")

    # GPU信息（如果有）
    if torch.cuda.is_available():
        print(f"\nGPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print("CUDA可用: ✅ 是")
    else:
        print("\nGPU: ❌ 未检测到CUDA可用GPU")

    print("\n" + "=" * 50)
    print("📦 软件环境信息")
    print("=" * 50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")

    # Ultralytics版本获取（通用兼容版）
    try:
        ultralytics_ver = get_version("ultralytics")
        print(f"Ultralytics YOLO版本: {ultralytics_ver}")
    except Exception as e:
        print(f"Ultralytics YOLO版本: 获取失败: {e}")

    print(f"CUDA版本 (PyTorch编译用): {torch.version.cuda}")  # type: ignore[attr-defined]
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")


if __name__ == "__main__":
    print_environment()
