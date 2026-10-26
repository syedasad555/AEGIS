<div align="center">
  <img src="assets/icon.png" width="200" alt="密集人群背景下人体异常行为检测">
</div>

<h1 align="center">密集人群背景下人体异常行为检测</h1>

<h3 align="center">Crowd Abnormal Behavior Detection</h3>
<p align="center"><strong>基于 YOLOv8 与 ByteTrack 的智能行为分析系统</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-red.svg" alt="License: AGPL-3.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.8+"></a>
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/Framework-Ultralytics%20YOLOv8-orange?logo=ultralytics" alt="Ultralytics YOLOv8"></a>
  <a href="https://github.com/juzishazhou/crowd-abnormal-behavior-detection"><img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-00bcd4" alt="Platform Windows | Linux"></a>
  <a href="https://github.com/juzishazhou/crowd-abnormal-behavior-detection/issues"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs welcome"></a>
</p>

<p align="center">
  <a href="README.md">🌐 English</a> · <a href="README_zh.md">🇨🇳 简体中文</a> · <a href="docs/USAGE.md">📖 使用文档</a> · <a href="https://github.com/juzishazhou/crowd-abnormal-behavior-detection/issues">🐞 报告问题</a>
</p>

---

<div align="left" style="margin: 20px 0;">

<details>
<summary><strong>📑 目录 / Table of Contents</strong></summary>

  **[功能特性](#功能特性)**  
  **[效果演示](#效果演示)**  
  **[快速开始](#快速开始)**  
  **[项目结构](#项目结构)**  
  **[环境安装](#环境安装)**  
  **[权重准备](#权重准备)**  
  **[使用方法](#使用方法)**  
  **[性能参考](#性能参考)**  
  **[工具脚本说明](#工具脚本说明)**  
  **[注意事项](#注意事项)**  
  **[参与贡献](#-参与贡献)**  
  **[License](#license)**  
  **[致谢](#致谢)**  

</details>

</div>

---

> 基于 Ultralytics YOLO 和 ByteTrack 的密集场景人体异常行为实时检测。

毕业设计项目，支持三类人体异常行为检测：

- **跌倒检测** — 姿态估计 + ByteTrack 跟踪 + 轨迹分析 + 时序平滑
- **奔跑检测** — 目标检测 + ByteTrack 跟踪 + 像素速度阈值
- **禁入区域入侵检测** — 多边形区域配置 + 脚底中心点/轨迹判定

---

## 功能特性

### 🧠 跌倒检测
基于 YOLOv8-pose 关键点提取、ByteTrack 多目标跟踪，结合轨迹分析和时序平滑过滤，有效抑制误检。

### 🏃 奔跑检测
使用 YOLO 目标检测 + ByteTrack 跟踪，计算像素位移速度并与可配置阈值比对，识别异常高速移动个体。

### 🚫 禁入区域检测
支持用户自定义多边形禁入区域，当人员脚底中心或轨迹进入禁区时触发告警。提供交互式区域标定工具 `tools/zone_selector.py`。

---

## 效果演示

### 跌倒检测
![fall detection demo](assets/demo/fall_demo.gif)

### 奔跑检测
![running detection demo](assets/demo/running_demo.gif)

### 禁入区域检测
![intrusion detection demo](assets/demo/intrusion_demo.gif)

---

## 快速开始

```bash
git clone https://github.com/juzishazhou/crowd-abnormal-behavior-detection.git
cd crowd-abnormal-behavior-detection
pip install -r requirements.txt
python scripts/fall_detection.py --source assets/fall.mp4 --output outputs/fall_demo.mp4
```

`assets/` 目录下已提供测试视频，权重文件首次运行时自动下载。

三个检测器都可以直接用样例视频体验：

```bash
# 跌倒检测
python scripts/fall_detection.py --source assets/fall.mp4 --output outputs/fall_demo.mp4

# 奔跑检测
python scripts/running_detection.py --source assets/running.mp4 --output outputs/running_demo.mp4

# 禁入区域检测（使用内置示例区域）
python scripts/intrusion_detection.py --source assets/intrusion.mp4 --output outputs/intrusion_demo.mp4
```

---

## 项目结构

```
crowd-abnormal-behavior-detection/
├── README.md
├── README_zh.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── _common.py                       # 共享工具模块 (视频 I/O、关键帧抽取等)
├── scripts/                         # 核心检测脚本
│   ├── fall_detection.py            # 跌倒检测
│   ├── running_detection.py         # 奔跑检测
│   └── intrusion_detection.py       # 禁入区域入侵检测
├── configs/                         # 配置文件
│   ├── bytetrack_fall.yaml          # ByteTrack 跌倒检测配置
│   ├── bytetrack.yaml               # ByteTrack 奔跑检测配置
│   └── forbidden_zone.example.json  # 禁入区域示例配置
├── tools/                           # 辅助工具
│   ├── check_env.py                 # 环境检查
│   ├── zone_selector.py             # 禁入区域交互式标定
│   ├── export_alerts.py             # 告警导出
│   ├── frame_viewer.py              # 视频帧查看器
│   ├── measure_fps.py               # FPS 测量
│   ├── measure_pipeline.py          # 流水线延迟测量
│   ├── measure_tracking.py          # 跟踪性能测量
│   └── run_eval.py                  # 模型评估
├── docs/                            # 文档
│   └── USAGE.md                     # 详细使用说明
├── assets/                          # 资源文件
│   ├── fall.mp4                     # 跌倒检测测试视频
│   ├── running.mp4                  # 奔跑检测测试视频
│   ├── intrusion.mp4                # 禁入区域测试视频
│   └── demo/                        # README 效果演示 GIF
│       ├── fall_demo.gif
│       ├── running_demo.gif
│       └── intrusion_demo.gif
└── outputs/                         # 本地运行结果（不入库）
    └── .gitkeep                     # 占位文件保证目录可见
```

---

## 环境安装

**Python 版本**：建议 Python 3.8+

```bash
git clone https://github.com/juzishazhou/crowd-abnormal-behavior-detection.git
cd crowd-abnormal-behavior-detection
pip install -r requirements.txt
```

### 依赖清单

| 包 | 用途 |
|---|---|
| `ultralytics>=8.4.65` | YOLO 推理 + 跟踪 |
| `opencv-python>=4.13.0` | 视频读写 + 可视化 |
| `numpy>=2.4.4` | 数组运算 |
| `torch>=2.12.0` | PyTorch 后端 |
| `shapely>=2.1.2` | 多边形精确判断（可选，禁入区域检测有 fallback） |
| `psutil>=7.2.2` | 系统信息检查 |
| `matplotlib>=3.10.9` | 论文配图绘制 |

---

## 权重准备

**本仓库不包含任何 `.pt` 权重文件。**

### 官方预训练权重

`pip install ultralytics` 后，首次运行脚本时会自动从 Ultralytics 服务器下载所需权重（如 `yolov8n.pt`、`yolov8n-pose.pt`、`yolov8s.pt` 等）。

你也可以手动下载放入 `weights/` 目录（此目录已加入 `.gitignore`）：

```bash
mkdir weights
# 下载官方权重
# https://github.com/ultralytics/assets/releases
```

### 微调权重

WiderPerson 数据集微调后的权重文件较大（~225 MB），暂不在仓库中提供。

> **TODO**：后续通过 GitHub Release 提供下载链接。

---

## 使用方法

所有脚本从项目根目录运行。

### 跌倒检测

```bash
python scripts/fall_detection.py --source <video_path> --output outputs/fall_demo.mp4
```

> *预期效果：输出视频中跌倒人员会被高亮框标注并显示告警标签，同时可选择抽取关键帧用于后续分析。*

- 模型：自动使用 `weights/yolov8n-pose.pt`（姿态估计）
- 追踪器：`configs/bytetrack_fall.yaml`
- 可选参数：`--extract-keyframes` 抽取跌倒关键帧

### 奔跑检测

```bash
python scripts/running_detection.py --source <video_path> --output outputs/running_demo.mp4
```

> *预期效果：输出视频中高速移动的个体会被高亮并显示速度标注，所有奔跑事件将记录到 `outputs/running_events.json`。*

- 模型：自动使用 `weights/yolov8s.pt`（目标检测）
- 追踪器：`configs/bytetrack.yaml`
- 日志：`outputs/running_events.json`（JSON 事件日志，含置信度分数）
- 灵敏度调节：`--conf-trigger 0.40`（降低阈值更多检测，提高阈值减少误检）
- 快速体验：`python scripts/running_detection.py --source assets/running.mp4`

### 禁入区域入侵检测

两步流程：

```bash
# 步骤 1：交互式标定禁入区域
python tools/zone_selector.py --source <video_path> --output configs/forbidden_zone.json

# 步骤 2：运行检测
python scripts/intrusion_detection.py --source <video_path> --zone configs/forbidden_zone.json --output outputs/intrusion_demo.mp4
```

> *预期效果：当人员脚底中心或轨迹进入禁入区域时，输出视频会触发可视化告警。*

如果不指定 `--zone`，脚本使用内置示例多边形（适用于 1280×720 视频中央区域）。

详细参数说明见 [docs/USAGE.md](docs/USAGE.md)。

---

## 性能参考

以下数据在 NVIDIA GeForce RTX 3050（4 GB）、CUDA 12.6、PyTorch 2.12.0 环境下测得：

| 测试项 | 分辨率 | FPS | 模型 |
|------|--------|-----|------|
| 纯检测（跌倒视频） | 720×1280 | **26.6** | YOLOv8n-pose |
| 完整流水线（奔跑视频） | 632×480 | **71.5** | YOLOv8s |

基准测试工具：`tools/measure_fps.py` 和 `tools/measure_pipeline.py`。

> ✅ **实时可用** — 完整流水线在 RTX 3050（4 GB）上可达 **26–71 FPS**，满足实时视频监控分析需求，无需跳帧处理。

---

## 工具脚本说明

| 工具 | 命令 | 用途 |
|------|------|------|
| 环境检查 | `python tools/check_env.py` | 检查 Python / PyTorch / Ultralytics 版本 |
| 区域标定 | `python tools/zone_selector.py --source <video> --output configs/zone.json` | 交互式标定禁入区域 |
| 告警导出 | `python tools/export_alerts.py --video <video>` | 批量导出异常告警 JSON |
| 帧查看器 | `python tools/frame_viewer.py <video_path>` | 逐帧查看视频 |
| FPS 测量 | `python tools/measure_fps.py --video <video> --weights <weights.pt>` | 测量纯检测 FPS |
| 流水线延迟 | `python tools/measure_pipeline.py --video <video> --weights <weights.pt>` | 测量各阶段延迟 |
| 跟踪性能 | `python tools/measure_tracking.py --video <video> --weights <weights.pt>` | 测量跟踪指标 |
| 模型评估 | `python tools/run_eval.py --weights <model.pt> --data <dataset.yaml>` | 评估 mAP / PR 曲线 |

> **注意**：`run_eval.py`、`export_alerts.py`、`measure_*.py` 是可选的高级工具，需要训练权重或数据集。

---

## 注意事项

1. **本仓库不包含**：数据集、权重文件（.pt）、训练结果
2. `assets/` 目录下已提供三个测试视频，可直接用于快速体验
3. 如果不使用 `assets/` 下的示例视频，用户需自行准备输入视频，通过 `--source` 参数传入
4. 禁入区域的坐标需根据实际视频分辨率通过 `tools/zone_selector.py` 标定
5. 权重文件放入 `weights/` 目录后即可使用（该目录已被 `.gitignore` 忽略）

---

## 🤝 参与贡献

欢迎任何形式的贡献！无论是报告 Bug、建议新功能，还是提交 Pull Request，你的帮助都能让这个项目变得更好。

- **发现 Bug？** [提交 Issue](https://github.com/juzishazhou/crowd-abnormal-behavior-detection/issues)
- **有新想法？** 在 Issues 中发起讨论
- **提交代码？** Fork 本仓库，创建特性分支，然后提交 PR

请确保代码风格与现有风格保持一致，并附带相应的文档说明。

---

## License

本项目基于 [AGPL-3.0](LICENSE) 开源。

本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0) 开发，沿用其许可证。

```
Copyright (C) 2026 juzishazhou
```

---

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 优秀的开源目标检测框架
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — 多目标跟踪算法
- [WiderPerson](https://github.com/ShiqiYu/WiderPerson) — 密集人群检测数据集
