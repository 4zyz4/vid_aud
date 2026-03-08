import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
from pydub import AudioSegment
import pyaudio

# 全局参数
CHUNK = 1024          # 每次读取/播放的帧数
MAX_BUFFER_POINTS = 5000  # 波形缓存最大点数

# 共享数据结构
xy_queue = queue.Queue()  # 用于传递 (x, y) 数据块
audio_finished = threading.Event()

def load_audio_to_numpy(file_path):
    """加载音频为归一化的 float32 左右声道"""
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.channels == 1:
            audio = audio.set_channels(2)
        elif audio.channels > 2:
            audio = AudioSegment.from_mono_audiosegments(*audio.split_to_mono()[:2])
        
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            samples = samples.astype(np.float32) / 32768.0

        left = samples[0::2]
        right = samples[1::2]
        return left, right, audio.frame_rate
    except Exception as e:
        print(f"加载音频失败: {e}")
        return None, None, None

def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio 回调函数：播放音频并推送数据到绘图队列"""
    global current_pos, left_channel, right_channel, sample_rate
    
    end_pos = current_pos + frame_count
    if current_pos >= len(left_channel):
        audio_finished.set()
        return (None, pyaudio.paComplete)

    # 获取当前块
    x_chunk = left_channel[current_pos:end_pos]
    y_chunk = right_channel[current_pos:end_pos]

    # 补零（如果不足 frame_count）
    if len(x_chunk) < frame_count:
        pad_len = frame_count - len(x_chunk)
        x_chunk = np.pad(x_chunk, (0, pad_len), constant_values=0)
        y_chunk = np.pad(y_chunk, (0, pad_len), constant_values=0)

    # 推送到绘图队列
    xy_queue.put((x_chunk.copy(), y_chunk.copy()))

    # 准备播放数据（立体声 interleaved）
    stereo_out = np.empty(frame_count * 2, dtype=np.float32)
    stereo_out[0::2] = x_chunk
    stereo_out[1::2] = y_chunk

    # 转为 int16（PyAudio 要求）
    out_int16 = (stereo_out * 32767).astype(np.int16).tobytes()

    current_pos = end_pos
    return (out_int16, pyaudio.paContinue)

# === 主程序 ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python xy_oscilloscope_sync.py <音频文件路径>")
        sys.exit(1)

    file_path = sys.argv[1]
    left_channel, right_channel, sample_rate = load_audio_to_numpy(file_path)
    if left_channel is None:
        sys.exit(1)

    print(f"音频加载成功: {len(left_channel)/sample_rate:.2f} 秒")

    # 初始化全局状态
    current_pos = 0
    audio_finished.clear()

    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=sample_rate,
        output=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    # 启动音频流
    stream.start_stream()

    # 绘图缓冲区
    x_buffer = []
    y_buffer = []

    # 设置绘图窗口
    plt.style.use('dark_background')
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("X-Y 示波器 (L vs R)", color='lime', fontsize=14)
    scatter = ax.scatter([], [], c='lime', s=1, alpha=0.7)

    def update_plot(frame):
        global x_buffer, y_buffer
        # 从队列中取出所有新数据
        while not xy_queue.empty():
            try:
                x_chunk, y_chunk = xy_queue.get_nowait()
                x_buffer.extend(x_chunk)
                y_buffer.extend(y_chunk)
            except queue.Empty:
                break

        # 限制缓冲区大小
        if len(x_buffer) > MAX_BUFFER_POINTS:
            x_buffer = x_buffer[-MAX_BUFFER_POINTS:]
            y_buffer = y_buffer[-MAX_BUFFER_POINTS:]

        # 更新绘图
        if x_buffer and y_buffer:
            scatter.set_offsets(np.column_stack((x_buffer, y_buffer)))
        return scatter,

    ani = FuncAnimation(fig, update_plot, interval=30, blit=True, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        # 清理
        stream.stop_stream()
        stream.close()
        p.terminate()
        audio_finished.set()