# 视频音频处理工具集

## 功能概述

1. **视频到音频转换器** (`video_to_audio_mapper.py`): 将视频帧转换为X-Y示波器音频信号
2. **音频交替播放器** (`alternate_audio.py`): 将两段音频按指定时间间隔交替播放
3. **X-Y示波器播放器** (`player.py`): 实时播放音频并显示X-Y示波器可视化效果

## 系统要求

- Python 3.7+
- FFmpeg
- 可选: CUDA (用于GPU加速)

## 安装

### 1. 安装Python依赖

```bash
pip install numpy opencv-python scipy pydub pyaudio matplotlib tqdm
```

### 2. 安装FFmpeg

- Windows: 从 [FFmpeg官网](https://ffmpeg.org/download.html) 下载并添加到系统PATH
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### 3. (可选) 安装CUDA加速支持

如果您有NVIDIA GPU并希望使用GPU加速处理视频:

```bash
pip install cupy
```

## 使用说明

### 视频到音频转换器

将视频转换为X-Y示波器音频信号:

```bash
python video_to_audio_mapper.py <输入视频路径> <输出音频路径> [--gpu]
```

**参数说明:**
- `输入视频路径`: 要转换的视频文件路径
- `输出音频路径`: 输出的音频文件路径
- `--gpu`: (可选) 使用GPU加速处理

**示例:**
```bash
# CPU处理
python video_to_audio_mapper.py input.mp3 output.wav

# GPU加速处理
python video_to_audio_mapper.py input.mp3 output.wav --gpu
```

**工作原理:**
1. 读取视频每一帧
2. 将帧转换为灰度图像并二值化
3. 检测图像边缘
4. 对边缘点坐标进行就近排序
5. 将坐标映射为立体声音频信号
6. 生成WAV格式的音频文件

### 音频交替播放器

将两段音频按指定时间间隔交替播放:

```bash
python alternate_audio.py <音频1路径> <音频2路径> <输出路径> [间隔秒数] [--disable-threading]
```

**参数说明:**
- `音频1路径`: 第一段音频文件路径
- `音频2路径`: 第二段音频文件路径
- `输出路径`: 输出音频文件路径
- `间隔秒数`: (可选) 交替时间间隔(秒)，默认为1/30秒
- `--disable-threading`: (可选) 禁用多线程处理

**示例:**
```bash
# 使用默认间隔(1/30秒)
python alternate_audio.py audio1.wav audio2.wav output.wav

# 自定义间隔为0.1秒
python alternate_audio.py audio1.wav audio2.wav output.wav 0.1

# 禁用多线程处理
python alternate_audio.py audio1.wav audio2.wav output.wav --disable-threading
```

### X-Y示波器播放器

实时播放音频并显示X-Y示波器可视化效果:

```bash
python player.py <音频文件路径>
```

**示例:**
```bash
python player.py audio.wav
```

**功能特点:**
- 实时播放音频
- 同步显示X-Y示波器可视化
- 支持立体声音频
- 自动调整声道数和采样率
- 美观的暗色主题界面

## 项目结构

```
vid_aud/
├── alternate_audio.py         # 音频交替播放器
├── player.py                  # X-Y示波器播放器
├── video_to_audio_mapper.py   # 视频到音频转换器
└── README.md                  # 项目文档
```

## 技术细节

### 视频到音频转换器

- 使用OpenCV进行视频处理
- 使用scipy进行音频文件操作
- 支持CUDA加速(需安装cupy)
- 使用KDTree优化最近邻搜索
- 自动处理视频帧与音频采样的对应关系

### 音频交替播放器

- 使用pydub进行音频处理
- 支持多线程处理以提高性能
- 自动调整音频采样率和声道数
- 使用tqdm显示进度条

### X-Y示波器播放器

- 使用pyaudio进行音频播放
- 使用matplotlib进行实时可视化
- 使用线程同步确保音频和可视化同步
- 支持多种音频格式

## 注意事项

1. 确保已正确安装FFmpeg并添加到系统PATH
2. 使用GPU加速需要安装CUDA和cupy
3. 音频交替播放器处理大文件时可能需要较长时间
4. X-Y示波器播放器需要系统支持音频输出
