# src/visualize_ground_truth.py

import cv2
import json
import os
import argparse
from tqdm import tqdm
import re

def visualize_ground_truth(video_path: str, main_json_path: str, output_path: str):
    """
    将主JSON文件中的真实边界框 (Ground Truth) 可视化到视频上。

    Args:
        video_path (str): 原始输入视频的路径。
        main_json_path (str): 包含所有任务和真值的主JSON文件路径。
        output_path (str): 输出带标注视频的路径。
    """
    print(f"开始生成真实框可视化...")
    print(f"输入视频: {video_path}")
    print(f"主JSON文件: {main_json_path}")
    print(f"输出视频: {output_path}")

    # 1. 加载包含所有任务和真值的主JSON文件
    try:
        with open(main_json_path, 'r', encoding='utf-8') as f:
            all_tasks_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 主JSON文件未找到 -> {main_json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 主JSON文件格式不正确 -> {main_json_path}")
        return

    # 从视频文件名中提取视频编号作为键
    video_filename = os.path.basename(video_path)
    video_number_match = re.search(r'(\d+)', video_filename)
    if not video_number_match:
        print(f"错误: 无法从文件名 {video_filename} 中提取视频编号。")
        return
    video_key = video_number_match.group(1)

    if video_key not in all_tasks_data:
        print(f"错误: 在JSON文件中未找到视频 '{video_key}' 的数据。")
        return

    # 2. 提取当前视频的真值数据
    task_info = all_tasks_data[video_key]
    gt_bboxes_list = task_info.get("target_bboxs")
    start_frame = task_info.get("temp_gt", {}).get("begin_fid")
    target_category = task_info.get("target_category", "ground_truth")

    if not gt_bboxes_list or start_frame is None:
        print(f"错误: 视频 '{video_key}' 的真值数据不完整 (缺少 'target_bboxs' 或 'begin_fid')。")
        return

    print(f"成功加载视频 '{video_key}' 的真值数据。目标类别: '{target_category}'")

    # 3. 初始化视频读写对象
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 -> {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 4. 逐帧处理
    frame_idx = 0
    with tqdm(total=total_frames, desc="生成真值可视化视频") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 计算当前帧在真值列表中的索引
            gt_index = frame_idx - start_frame
            
            # 检查当前帧是否在标注范围内，并且索引有效
            if 0 <= gt_index < len(gt_bboxes_list):
                bbox = gt_bboxes_list[gt_index]
                
                # 兼容两种可能的真值框格式
                if isinstance(bbox, dict) and 'xmin' in bbox:
                    # 格式: {'xmin':, 'ymin':, 'xmax':, 'ymax':}
                    xmin = int(bbox['xmin'])
                    ymin = int(bbox['ymin'])
                    xmax = int(bbox['xmax'])
                    ymax = int(bbox['ymax'])
                elif isinstance(bbox, list) and len(bbox) == 4:
                    # 格式: [x, y, w, h]
                    x, y, w, h = bbox
                    xmin, ymin, xmax, ymax = int(x), int(y), int(x+w), int(y+h)
                else:
                    # 格式无法识别，跳过绘制
                    out.write(frame)
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # 在帧上绘制矩形框 (使用红色以区分预测框)
                # BGR color for red: (0, 0, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                # 在框的上方绘制标签
                label = f"GT: {target_category}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_w, text_h = text_size
                
                cv2.rectangle(frame, (xmin, ymin - text_h - 10), (xmin + text_w, ymin - 5), (0, 0, 255), -1)
                cv2.putText(frame, label, (xmin, ymin - 10), font, font_scale, (255, 255, 255), font_thickness)

            # 将处理过的帧写入输出视频
            out.write(frame)
            
            frame_idx += 1
            pbar.update(1)

    # 5. 释放资源
    cap.release()
    out.release()
    print("\n真实框可视化视频处理完成！")
    print(f"输出文件已保存至: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将JSON文件中的真实边界框可视化到视频上")
    parser.add_argument('--video_path', type=str, required=True, help='原始输入视频的路径 (例如: sample_videos/video_1.mp4)')
    parser.add_argument('--main_json_path', type=str, default='sample_video.json', help='包含所有任务和真值的主JSON文件路径。')
    parser.add_argument('--output_path', type=str, required=True, help='带真实框标注的输出视频路径 (例如: output/video_1_ground_truth.mp4)')

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualize_ground_truth(args.video_path, args.main_json_path, args.output_path)