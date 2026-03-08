import sys
import numpy as np
from pydub import AudioSegment
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math


def load_audio(file_path):
    """
    加载音频文件并返回音频对象和采样率
    """
    try:
        audio = AudioSegment.from_file(file_path)
        return audio
    except Exception as e:
        print(f"加载音频文件失败 {file_path}: {e}")
        return None


def get_audio_segment(audio, start_time_ms, end_time_ms, total_duration_ms):
    """
    获取音频片段，如果超出范围则补充静音
    """
    if start_time_ms >= total_duration_ms:
        # 如果起始时间超出音频长度，返回静音
        duration = end_time_ms - start_time_ms
        return AudioSegment.silent(duration=int(duration), frame_rate=audio.frame_rate)
    
    actual_end_time = min(end_time_ms, total_duration_ms)
    segment = audio[start_time_ms:actual_end_time]
    
    # 如果片段长度不够，补静音
    if len(segment) < (end_time_ms - start_time_ms):
        segment = segment + AudioSegment.silent(
            duration=int(end_time_ms - start_time_ms - len(segment)),
            frame_rate=audio.frame_rate
        )
    
    return segment


def process_alternate_chunk(args):
    """
    处理音频交替的一个片段
    """
    chunk_idx, start_time_ms, end_time_ms, audio1, audio2, duration1_ms, duration2_ms, use_first_audio = args
    
    # 获取当前要添加的音频片段
    if use_first_audio:
        segment = get_audio_segment(audio1, start_time_ms, end_time_ms, duration1_ms)
    else:
        segment = get_audio_segment(audio2, start_time_ms, end_time_ms, duration2_ms)
    
    return chunk_idx, segment


def alternate_audio_tracks(audio1_path, audio2_path, output_path, interval_seconds=1/30.0, use_threads=True):
    """
    将两段音频按指定时间间隔交替播放
    
    :param audio1_path: 第一段音频路径
    :param audio2_path: 第二段音频路径
    :param output_path: 输出音频路径
    :param interval_seconds: 交替时间间隔（秒），默认为1/30秒
    :param use_threads: 是否使用多线程处理
    """
    # 加载两段音频
    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)
    
    if audio1 is None or audio2 is None:
        print("音频加载失败，退出...")
        return
    
    # 确保两段音频具有相同的声道数和采样率
    if audio1.channels != audio2.channels:
        max_channels = max(audio1.channels, audio2.channels)
        if audio1.channels < max_channels:
            audio1 = audio1.set_channels(max_channels)
        if audio2.channels < max_channels:
            audio2 = audio2.set_channels(max_channels)
    
    if audio1.frame_rate != audio2.frame_rate:
        # 统一使用较高采样率
        target_rate = max(audio1.frame_rate, audio2.frame_rate)
        audio1 = audio1.set_frame_rate(target_rate)
        audio2 = audio2.set_frame_rate(target_rate)
    
    # 验证处理前后音频时长的一致性
    duration1_ms = len(audio1)
    duration2_ms = len(audio2)
    max_duration_ms = max(duration1_ms, duration2_ms)
    
    print(f"音频1时长: {duration1_ms/1000:.2f}秒")
    print(f"音频2时长: {duration2_ms/1000:.2f}秒")
    print(f"最大时长: {max_duration_ms/1000:.2f}秒")
    
    # 计算每次间隔的毫秒数
    interval_ms = interval_seconds * 1000
    
    # 计算总段数
    total_chunks = math.ceil(max_duration_ms / interval_ms)
    
    if use_threads:
        # 使用多线程处理
        print("使用多线程处理音频...")
        chunks_args = []
        
        for i in range(total_chunks):
            start_time_ms = i * interval_ms
            end_time_ms = min((i + 1) * interval_ms, max_duration_ms)
            use_first_audio = (i % 2) == 0  # 从第一段音频开始
            
            chunks_args.append((i, start_time_ms, end_time_ms, audio1, audio2, 
                                duration1_ms, duration2_ms, use_first_audio))
        
        # 使用ThreadPoolExecutor处理音频片段
        segments = [None] * total_chunks
        with ThreadPoolExecutor() as executor:
            # 使用tqdm显示进度
            results = list(tqdm(executor.map(process_alternate_chunk, chunks_args), 
                               total=len(chunks_args), desc="处理音频片段"))
            
            # 按索引排序结果
            for chunk_idx, segment in results:
                segments[chunk_idx] = segment
        
        # 合并所有片段
        result_audio = AudioSegment.empty()
        for segment in tqdm(segments, desc="合并音频片段"):
            result_audio += segment
    
    else:
        # 单线程处理
        print("使用单线程处理音频...")
        result_audio = AudioSegment.silent(duration=0, frame_rate=audio1.frame_rate)
        
        # 使用tqdm显示进度
        for i in tqdm(range(total_chunks), desc="处理音频"):
            start_time_ms = i * interval_ms
            end_time_ms = min((i + 1) * interval_ms, max_duration_ms)
            use_first_audio = (i % 2) == 0  # 从第一段音频开始
            
            # 获取当前要添加的音频片段
            if use_first_audio:
                segment = get_audio_segment(audio1, start_time_ms, end_time_ms, duration1_ms)
            else:
                segment = get_audio_segment(audio2, start_time_ms, end_time_ms, duration2_ms)
            
            # 添加到结果音频
            result_audio += segment
    
    # 验证输出音频时长
    output_duration = len(result_audio)
    print(f"输出音频时长: {output_duration/1000:.2f}秒")
    
    # 导出结果音频
    result_audio.export(output_path, format=os.path.splitext(output_path)[1][1:])
    print(f"交替音频已保存至: {output_path}")


def main():
    if len(sys.argv) < 4:
        print("使用方法: python alternate_audio.py <音频1路径> <音频2路径> <输出路径> [间隔秒数] [--disable-threading]")
        print("例如: python alternate_audio.py audio1.wav audio2.wav output.wav 0.0333")
        print("注意: 1/30秒 ≈ 0.0333秒")
        return
    
    audio1_path = sys.argv[1]
    audio2_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # 解析时间间隔参数，默认为1/30秒
    interval_seconds = 1/30.0
    if len(sys.argv) > 4 and not sys.argv[4].startswith('--'):
        interval_seconds = float(sys.argv[4])
    
    # 检查是否禁用多线程
    use_threads = True
    if '--disable-threading' in sys.argv or '--disable-threads' in sys.argv:
        use_threads = False
    
    print(f"开始交替音频...")
    print(f"音频1: {audio1_path}")
    print(f"音频2: {audio2_path}")
    print(f"输出: {output_path}")
    print(f"时间间隔: {interval_seconds}秒 ({1/interval_seconds}次/秒)")
    print(f"多线程处理: {'启用' if use_threads else '禁用'}")
    
    alternate_audio_tracks(audio1_path, audio2_path, output_path, interval_seconds, use_threads)


if __name__ == "__main__":
    main()