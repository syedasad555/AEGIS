# 异常行为检测脚本使用说明

三个独立脚本，共用 YOLO + ByteTrack 基础设施。所有脚本从项目根目录运行。

---

## 环境准备

```bash
pip install -r requirements.txt
```

权重文件需自行放入 `weights/` 目录，详见 README.md。

---

## 1. 跌倒检测 — `scripts/fall_detection.py`

基于姿态估计 + ByteTrack 跟踪 + 轨迹分析，检测人体跌倒。

```bash
python scripts/fall_detection.py --source <video_path> --output outputs/fall_demo.mp4
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | 无（必填） | 输入视频路径 |
| `--output` | `outputs/fall_demo.mp4` | 输出视频路径 |
| `--extract-keyframes` / `--no-extract-keyframes` | 启用 | 是否抽取跌倒关键帧 |
| `--keyframe-dir` | `outputs/highlights` | 关键帧输出目录 |
| `--keyframe-range` | `10` | 关键帧抽取范围 ±N 帧 |
| `--keyframe-step` | `2` | 关键帧抽取步长 |

输出：`outputs/fall_demo.mp4`、`outputs/highlights/fall_*.png`

---

## 2. 奔跑检测 — `scripts/running_detection.py`

基于 YOLO 检测 + ByteTrack 跟踪 + 像素速度计算，标签格式 `ID:X running 0.XX`。

```bash
python scripts/running_detection.py --source <video_path> --output outputs/running_demo.mp4
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | 无（必填） | 输入视频路径 |
| `--output` | `outputs/running_demo.mp4` | 输出视频路径 |
| `--log` | `outputs/running_events.json` | 事件日志 JSON 路径 |
| `--conf-trigger` | `0.50` | 显示阈值（越低越多检测） |
| `--model` | `weights/yolov8s.pt` | YOLO 模型路径 |
| `--tracker` | `configs/bytetrack.yaml` | ByteTrack 配置路径 |

输出：`outputs/running_demo.mp4`、`outputs/running_events.json`

---

## 3. 禁入区检测 — `scripts/intrusion_detection.py`

检测行人进入预设禁入区域。**需先用选区工具标定区域**（两步流程）。

### 步骤 1：标定禁入区

```bash
python tools/zone_selector.py --source <video_path> --output configs/forbidden_zone.json
```

| 操作 | 功能 |
|---|---|
| 左键点击 | 在当前区域添加顶点 |
| 右键点击 | 撤销上一个顶点 |
| `n` | 新建一个区域 |
| `Tab` / `]` / `[` | 切换编辑区域 |
| `d` | 删除当前区域 |
| `c` | 清空当前区域顶点 |
| `s` | 保存所有区域到 JSON |
| `q` | 退出（不保存） |
| `1`-`9` | 跳转到视频 10%-90% 位置 |

### 步骤 2：运行检测

```bash
python scripts/intrusion_detection.py --source <video_path> --zone configs/forbidden_zone.json --output outputs/intrusion_demo.mp4
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | 无（必填） | 输入视频路径 或 0 使用摄像头 |
| `--output` | `outputs/intrusion_demo.mp4` | 输出视频路径 |
| `--zone` | 无（使用示例区域） | 禁入区 JSON 配置路径 |

输出：视频 + 终端 JSON 事件日志

---

## 4. 快速体验（使用仓库自带测试视频）

Clone 后可直接使用 `assets/` 目录下的测试视频跑通全流程：

### 4.1 跌倒检测

```bash
# 使用自带测试视频
python scripts/fall_detection.py --source assets/fall.mp4 --output outputs/fall_result.mp4 --extract-keyframes --keyframe-dir outputs/fall_keyframes

# 禁用关键帧（只跑检测，更快）
python scripts/fall_detection.py --source assets/fall.mp4 --no-extract-keyframes
```

### 4.2 奔跑检测

```bash
# 使用自带测试视频
python scripts/running_detection.py --source assets/running.mp4 --output outputs/running_result.mp4 --conf-trigger 0.50 --log outputs/running_events.json

# 换轻量模型提速
python scripts/running_detection.py --source assets/running.mp4 --model yolov8n.pt --conf-trigger 0.35
```

### 4.3 禁入区检测

```bash
# 步骤 1：标定（只做一次）
python tools/zone_selector.py --source assets/intrusion.mp4 --output configs/forbidden_zone.json

# 步骤 2：检测
python scripts/intrusion_detection.py --source assets/intrusion.mp4 --zone configs/forbidden_zone.json --output outputs/intrusion_result.mp4

# 跳过 JSON：直接用示例区域跑检测
python scripts/intrusion_detection.py --source assets/intrusion.mp4
```

### 4.4 统一上下文异常检测管线与 2D 数字孪生

```bash
# 启动 2D 数字孪生 Web 监控大屏
python server.py
# 浏览器访问: http://localhost:8080/

# 运行统一行人检测管线 (自动提取轨迹与疏散异常)
python scripts/pedestrian_pipeline.py --source assets/intrusion.mp4

# 载入自定义禁入区多边形 JSON 配置
python scripts/pedestrian_pipeline.py --source assets/intrusion.mp4 --zone configs/forbidden_zone.example.json
```

### 使用自己的视频

将上述命令中的 `assets/xxx.mp4` 替换为你的视频路径即可：

```bash
python scripts/fall_detection.py --source your_video.mp4 --output outputs/fall_result.mp4
```

---

## 5. 工具脚本说明

| 工具 | 用途 |
|---|---|
| `tools/zone_selector.py` | 交互式禁入区域标定 |
| `tools/export_alerts.py` | 导出所有告警到 JSON 用于评估 |
| `tools/frame_viewer.py` | 逐帧查看视频 |
| `tools/measure_fps.py` | 测量检测 FPS |
| `tools/measure_pipeline.py` | 测量各阶段延迟 |
| `tools/measure_tracking.py` | 测量跟踪指标 |
| `tools/run_eval.py` | 模型评估 (需数据集) |
| `tools/check_env.py` | 环境检查 |

---

## 6. 目录约定

- 输入视频由用户自行准备，通过 `--source` 传入
- 输出默认写入 `outputs/` 目录
- 模型权重放入 `weights/` 目录（不纳入 Git）
- 区域配置写入 `configs/` 目录
