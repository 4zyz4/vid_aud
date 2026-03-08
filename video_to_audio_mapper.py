import cv2
import numpy as np
import subprocess
import json
import math
from scipy.io import wavfile
import os

# 尝试导入cupy用于CUDA加速
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("CUDA可用，将使用GPU加速")
except ImportError:
    CUDA_AVAILABLE = False
    print("未安装cupy，无法使用CUDA加速")


def get_video_info(video_path):
    """
    1. 使用ffmpeg打开视频文件并读取属性数据
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
    info = json.loads(result.stdout)
    video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    
    if not video_stream:
        raise ValueError("视频文件中未找到视频流")
    
    fps_str = video_stream.get('avg_frame_rate', '30/1')
    if fps_str:
        num, den = map(int, fps_str.split('/') if '/' in fps_str else [fps_str, '1'])
        fps = num / den if den != 0 else 30.0
    else:
        fps = 30.0
        
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration = float(video_stream.get('duration', 0))
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration,
        'total_frames': int(fps * duration) if duration > 0 else 0
    }


def simple_edge_detection(binary_img, use_gpu=False):
    """
    4. 边缘检测：简化版本，检测周围像素点与目标像素点不同的点
    """
    if use_gpu and CUDA_AVAILABLE:
        # 使用GPU进行边缘检测
        binary_cp = cp.asarray(binary_img, dtype=cp.uint8)
        
        # 创建四个方向的移位图像（上下左右）
        top = cp.pad(binary_cp[:-1, :], ((1, 0), (0, 0)), mode='constant')
        bottom = cp.pad(binary_cp[1:, :], ((0, 1), (0, 0)), mode='constant')
        left = cp.pad(binary_cp[:, :-1], ((0, 0), (1, 0)), mode='constant')
        right = cp.pad(binary_cp[:, 1:], ((0, 0), (0, 1)), mode='constant')
        
        # 检查中心像素是否与任意邻居不同
        edges = (
            (binary_cp != top) | 
            (binary_cp != bottom) | 
            (binary_cp != left) | 
            (binary_cp != right)
        )
        
        # 只有原始值为1的点才是边缘
        edge_map = cp.where(edges & (binary_cp == 255), 255, 0)
        return cp.asnumpy(edge_map.astype(np.uint8))
    else:
        # 使用CPU进行边缘检测
        h, w = binary_img.shape
        edge_map = np.zeros_like(binary_img)
        
        for y in range(h):
            for x in range(w):
                center_val = binary_img[y, x]
                
                # 检查四个相邻像素
                neighbors_different = False
                
                # 上
                if y > 0 and binary_img[y-1, x] != center_val:
                    neighbors_different = True
                # 下
                elif y < h-1 and binary_img[y+1, x] != center_val:
                    neighbors_different = True
                # 左
                elif x > 0 and binary_img[y, x-1] != center_val:
                    neighbors_different = True
                # 右
                elif x < w-1 and binary_img[y, x+1] != center_val:
                    neighbors_different = True
                # 对角线
                elif y > 0 and x > 0 and binary_img[y-1, x-1] != center_val:
                    neighbors_different = True
                elif y > 0 and x < w-1 and binary_img[y-1, x+1] != center_val:
                    neighbors_different = True
                elif y < h-1 and x > 0 and binary_img[y+1, x-1] != center_val:
                    neighbors_different = True
                elif y < h-1 and x < w-1 and binary_img[y+1, x+1] != center_val:
                    neighbors_different = True
                
                if neighbors_different and center_val == 255:
                    edge_map[y, x] = 255
    
    return edge_map


def bfs_nearest_point_sorting(edge_coords, start_idx=0):
    """
    5. 使用广度优先搜索对边缘点进行就近排序（优化版）
    """
    if len(edge_coords) == 0:
        return edge_coords

    # 将坐标转换为numpy数组
    coords = np.array(edge_coords)
    n_points = len(coords)
    
    # 如果点的数量很少，直接使用原算法
    if n_points < 100:
        # 已访问点集合
        visited = set()
        sorted_coords = []

        # 从指定起始点开始
        current_idx = start_idx
        visited.add(current_idx)
        sorted_coords.append(coords[current_idx])

        while len(visited) < n_points:
            current_point = coords[current_idx]

            # 计算当前点到所有未访问点的距离
            unvisited_indices = [i for i in range(n_points) if i not in visited]
            if not unvisited_indices:
                break

            unvisited_points = coords[unvisited_indices]

            # 计算欧氏距离
            distances = np.linalg.norm(unvisited_points - current_point, axis=1)

            # 找到最近的点
            nearest_local_idx = np.argmin(distances)
            nearest_global_idx = unvisited_indices[nearest_local_idx]

            # 添加到已访问集合和排序结果
            visited.add(nearest_global_idx)
            sorted_coords.append(coords[nearest_global_idx])

            # 更新当前点为最近点
            current_idx = nearest_global_idx

        return np.array(sorted_coords)
    
    # 对于大量点的情况，使用KDTree来加速最近邻搜索
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    
    visited = set()
    sorted_coords = []
    
    # 从指定起始点开始
    current_point = coords[start_idx]
    current_idx = start_idx
    visited.add(current_idx)
    sorted_coords.append(coords[current_idx])
    
    while len(visited) < n_points:
        # 查询下一个最近的未访问点
        # 使用足够大的k值，确保能找到未访问的点
        k = min(n_points, max(20, n_points // 10))  # 动态调整查询数量
        distances, indices = tree.query(current_point, k=k)
        
        # 如果只有一个查询结果，直接检查
        if k == 1:
            if indices not in visited:
                next_idx = indices
            else:
                break
        else:
            # 找到第一个未访问的点
            next_idx = None
            for idx in indices:
                if idx not in visited:
                    next_idx = idx
                    break
            
            # 如果没找到未访问的点，增加查询范围
            attempts = 0
            while next_idx is None and attempts < 3:
                attempts += 1
                k = min(n_points, k * 2)  # 增加查询数量
                distances, indices = tree.query(current_point, k=k)
                for idx in indices:
                    if idx not in visited:
                        next_idx = idx
                        break
        
        # 如果仍然没找到未访问的点，跳出循环
        if next_idx is None:
            break
            
        # 更新当前点和访问状态
        visited.add(next_idx)
        sorted_coords.append(coords[next_idx])
        current_point = coords[next_idx]
        
    return np.array(sorted_coords)


def process_video_to_audio(video_path, output_audio_path, use_gpu=CUDA_AVAILABLE):
    """
    主函数：将视频转换为音频
    """
    print(f"开始处理视频: {video_path}")
    
    # 1. 获取视频信息
    video_info = get_video_info(video_path)
    print(f"视频信息: {video_info['width']}x{video_info['height']}, "
          f"{video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    # 音频参数
    audio_sample_rate = 44100
    samples_per_frame = int(audio_sample_rate / video_info['fps'])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_samples = total_frames * samples_per_frame
    
    print(f"音频参数: {audio_sample_rate}Hz, 每帧采样数: {samples_per_frame}, 总采样数: {total_samples}")
    
    # 初始化音频数组
    stereo_wave = np.zeros((total_samples, 2), dtype=np.float32)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"正在处理第 {frame_idx+1}/{total_frames} 帧", end='\r')
        
        # 2. 读取当前帧每个像素点灰度值，并存入数组
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 3. 二值化，以平均值为界限
        if use_gpu and CUDA_AVAILABLE:
            gray_gpu = cp.asarray(gray)
            mean_val = cp.mean(gray_gpu)
            binary = cp.where(gray_gpu > mean_val, 255, 0).astype(cp.uint8)
            binary = cp.asnumpy(binary)
        else:
            mean_val = np.mean(gray)
            binary = np.where(gray > mean_val, 255, 0).astype(np.uint8)
        
        # 4. 边缘检测
        edge_map = simple_edge_detection(binary, use_gpu=use_gpu)
        
        # 获取边缘点坐标
        edge_coords = np.column_stack(np.where(edge_map > 0))
        
        if len(edge_coords) == 0:
            # 如果没有边缘点，生成一些随机点
            print(f"\n警告: 第 {frame_idx+1} 帧未检测到边缘，使用随机点")
            n_samples = min(samples_per_frame, video_info['width'] * video_info['height'])
            y_coords = np.random.randint(0, video_info['height'], n_samples)
            x_coords = np.random.randint(0, video_info['width'], n_samples)
            coords = np.column_stack([y_coords, x_coords])
        else:
            # 5. 对边缘像素点坐标就近排序
            if len(edge_coords) > samples_per_frame:
                # 如果边缘点过多，先排序再采样
                sorted_coords = bfs_nearest_point_sorting(edge_coords)
                # 采样所需的点数
                indices = np.linspace(0, len(sorted_coords)-1, samples_per_frame, dtype=int)
                coords = sorted_coords[indices]
            else:
                # 如果边缘点不足，使用BFS排序后重复
                sorted_coords = bfs_nearest_point_sorting(edge_coords)
                if len(sorted_coords) < samples_per_frame:
                    # 重复点直到达到所需数量
                    repeats_needed = math.ceil(samples_per_frame / len(sorted_coords))
                    extended_coords = np.tile(sorted_coords, (repeats_needed, 1))
                    coords = extended_coords[:samples_per_frame]
                else:
                    coords = sorted_coords[:samples_per_frame]
        
        # 6. 翻转Y轴（示波器左下角为原点）
        # 原来的Y坐标是从顶部开始的，现在翻转为从底部开始
        coords[:, 0] = video_info['height'] - coords[:, 0]
        
        # 7. 将坐标映射到音频文件，x坐标对应右声道，y坐标对应左声道
        y_coords_normalized = (coords[:, 0] / video_info['height']) * 2 - 1  # [-1, 1]范围
        x_coords_normalized = (coords[:, 1] / video_info['width']) * 2 - 1   # [-1, 1]范围
        
        start_idx = frame_idx * samples_per_frame
        end_idx = start_idx + samples_per_frame
        
        if end_idx <= total_samples:
            # 左声道对应x坐标，右声道对应y坐标 (修正旋转问题)
            stereo_wave[start_idx:end_idx, 0] = x_coords_normalized  # 左声道 - 修正为x坐标
            stereo_wave[start_idx:end_idx, 1] = y_coords_normalized  # 右声道 - 修正为y坐标
        else:
            # 处理最后一帧可能超出数组边界的情况
            actual_samples = total_samples - start_idx
            stereo_wave[start_idx:, 0] = x_coords_normalized[:actual_samples]  # 左声道 - 修正为x坐标
            stereo_wave[start_idx:, 1] = y_coords_normalized[:actual_samples]  # 右声道 - 修正为y坐标
        
        frame_idx += 1
    
    cap.release()
    print(f"\n视频处理完成，共处理 {frame_idx} 帧")
    
    # 限制音频值在[-1, 1]范围内
    stereo_wave = np.clip(stereo_wave, -1.0, 1.0)
    
    # 转换为int16格式
    stereo_int16 = np.int16(stereo_wave * 32767)
    
    # 保存音频文件
    wavfile.write(output_audio_path, audio_sample_rate, stereo_int16)
    print(f"音频文件已保存至: {output_audio_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='将视频转换为X-Y示波器音频信号')
    parser.add_argument('input_video', help='输入视频文件路径')
    parser.add_argument('output_audio', help='输出音频文件路径')
    parser.add_argument('--gpu', action='store_true', help='使用GPU加速处理')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_video):
        print(f"错误: 输入视频文件不存在 - {args.input_video}")
        return
    
    # 处理视频到音频
    process_video_to_audio(
        args.input_video, 
        args.output_audio, 
        use_gpu=args.gpu
    )


if __name__ == "__main__":
    main()