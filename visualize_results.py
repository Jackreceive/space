# src/visualize_results.py

import cv2
import json
import os
import argparse
from tqdm import tqdm

def visualize_tracking_results(video_path: str, json_path: str, output_path: str):
    """
    将JSON文件中的追踪结果可视化到视频上。

    Args:
        video_path (str): 原始输入视频的路径。
        json_path (str): 包含预测边界框的JSON结果文件路径。
        output_path (str): 输出带标注视频的路径。
    """
    print(f"开始可视化处理...")
    print(f"输入视频: {video_path}")
    print(f"结果文件: {json_path}")
    print(f"输出视频: {output_path}")

    # 1. 加载JSON结果文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 结果文件未找到 -> {json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 结果文件格式不正确，无法解析 -> {json_path}")
        return

    # 提取追踪数据 (兼容您之前的输出格式)
    # 假设JSON文件中只有一个视频键
    if not results_data:
        print("错误: JSON文件为空。")
        return
    video_key = list(results_data.keys())[0]
    tracking_info = results_data[video_key]
    query = tracking_info.get("query", "Unknown Query")
    refined_query = tracking_info.get("refined_query", "Unknown Target")
    bboxes = tracking_info.get("pred_bboxs", {})

    print(f"成功加载到视频 '{video_key}' 的结果。")
    print(f"精炼后的追踪目标: '{refined_query}'")

    # 2. 初始化视频读写对象
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 -> {video_path}")
        return

    # 获取视频属性以创建写入对象
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 定义视频编码器并创建VideoWriter对象
    # 使用 'mp4v' 编码器来创建 .mp4 文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 3. 逐帧处理
    frame_idx = 0
    with tqdm(total=total_frames, desc="生成可视化视频") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 检查当前帧是否有对应的边界框数据
            frame_key = str(frame_idx)
            if frame_key in bboxes and bboxes[frame_key]:
                bbox = bboxes[frame_key]
                xmin = int(bbox['xmin'])
                ymin = int(bbox['ymin'])
                xmax = int(bbox['xmax'])
                ymax = int(bbox['ymax'])

                # 在帧上绘制矩形框
                # BGR color for green: (0, 255, 0)
                # Thickness: 2
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # 在框的上方绘制标签
                label = refined_query
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_w, text_h = text_size
                
                # 绘制标签背景
                cv2.rectangle(frame, (xmin, ymin - text_h - 10), (xmin + text_w, ymin - 5), (0, 255, 0), -1)
                # 绘制标签文字 (黑色)
                cv2.putText(frame, label, (xmin, ymin - 10), font, font_scale, (0, 0, 0), font_thickness)

            # 将处理过的帧写入输出视频
            out.write(frame)
            
            frame_idx += 1
            pbar.update(1)

    # 4. 释放资源
    cap.release()
    out.release()
    print("\n可视化视频处理完成！")
    print(f"输出文件已保存至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将追踪结果JSON文件可视化到视频上")
    parser.add_argument('--video_path', type=str, required=True, help='原始输入视频的路径 (例如: sample_videos/video_32.mp4)')
    parser.add_argument('--json_path', type=str, required=True, help='追踪结果JSON文件的路径 (例如: output/results_zhipu_api.json)')
    parser.add_argument('--output_path', type=str, required=True, help='带标注的输出视频路径 (例如: output/video_32_annotated.mp4)')

    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualize_tracking_results(args.video_path, args.json_path, args.output_path)