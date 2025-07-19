# run_all_videos.py (已修正真值框解析逻辑)

import os
import sys
import json
import subprocess
import argparse
from tqdm import tqdm
import re

from src.iou_calculator import calculate_iou

def process_all_videos(args):
    """
    自动化处理所有视频的主函数。
    """
    python_executable = sys.executable
    print(f"将使用此Python解释器来运行子进程: {python_executable}")
    
    if not os.path.exists(args.videos_dir):
        print(f"错误: 视频目录不存在 -> {args.videos_dir}")
        return
    if not os.path.exists(args.main_json_path):
        print(f"错误: 主JSON任务文件不存在 -> {args.main_json_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.main_json_path, 'r', encoding='utf-8') as f:
        all_tasks_data = json.load(f)

    video_files = sorted([f for f in os.listdir(args.videos_dir) if f.endswith('.mp4')])
    
    if not video_files:
        print(f"错误: 在目录 {args.videos_dir} 中未找到.mp4视频文件。")
        return

    print(f"找到 {len(video_files)} 个视频待处理。")

    for video_filename in tqdm(video_files, desc="总处理进度"):
        
        video_number_match = re.search(r'(\d+)', video_filename)
        if not video_number_match:
            print(f"跳过: 无法从 {video_filename} 中提取视频编号。")
            continue
        
        video_number_str = video_number_match.group(1)
        video_path = os.path.join(args.videos_dir, video_filename)
        result_json_path = os.path.join(args.output_dir, f"{video_number_str}_result.json")
        
        print(f"\n--- 正在处理视频 {video_number_str}: {video_filename} ---")
        
        main_script_command = [
            python_executable,
            'src/main_llm.py',
            '--video_path', video_path,
            '--json_path', args.main_json_path,
            '--api_key', args.api_key,
            '--output_path', result_json_path
        ]
        
        try:
            subprocess.run(main_script_command, check=True, capture_output=True, text=True, timeout=300)
            print(f"视频 {video_number_str} 处理完成，结果已保存至 {result_json_path}")
        except subprocess.CalledProcessError as e:
            print(f"错误: 运行主脚本处理视频 {video_number_str} 时失败。")
            print(f"标准错误: {e.stderr}")
            continue # 跳过当前视频，继续处理下一个

        # 计算并追加 IoU
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            if video_number_str not in all_tasks_data:
                print(f"警告: 在主JSON文件中找不到视频 {video_number_str} 的真值数据。")
                continue
            
            # --- 这是被修正的关键逻辑 ---
            # 1. 从 'target_bboxs' 获取真值框列表
            gt_bbox_list = all_tasks_data[video_number_str].get('target_bboxs')
            if not gt_bbox_list:
                print(f"警告: 视频 {video_number_str} 的真值数据中没有 'target_bboxs' 字段。")
                continue
            
            # 2. 获取开始帧，用于计算偏移量
            start_frame = all_tasks_data[video_number_str].get('temp_gt', {}).get('begin_fid')
            if start_frame is None:
                print(f"警告: 视频 {video_number_str} 中找不到 'begin_fid'。")
                continue
            # --------------------------

            pred_bboxs = result_data.get(video_number_str, {}).get('pred_bboxs', {})
            total_iou, iou_count, frame_ious = 0, 0, {}

            # 遍历所有有预测结果的帧
            for frame_idx_str, pred_box in pred_bboxs.items():
                if pred_box: # 只在有预测框的帧上计算IoU
                    frame_idx = int(frame_idx_str)
                    # 计算当前帧在真值列表中的索引
                    gt_index = frame_idx - start_frame
                    
                    # 确保索引有效
                    if 0 <= gt_index < len(gt_bbox_list):
                        gt_box = gt_bbox_list[gt_index]
                        
                        iou = calculate_iou(pred_box, gt_box)
                        frame_ious[frame_idx_str] = iou
                        total_iou += iou
                        iou_count += 1
            
            average_iou = total_iou / iou_count if iou_count > 0 else 0
            
            result_data[video_number_str]['average_iou'] = average_iou
            result_data[video_number_str]['frame_by_frame_iou'] = frame_ious
            
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=4)
            
            print(f"视频 {video_number_str} 的平均 IoU 为: {average_iou:.4f}")

        except Exception as e:
            print(f"错误: 在为视频 {video_number_str} 计算或追加 IoU 时失败: {e}")

        if args.visualize:
            annotated_video_path = os.path.join(args.output_dir, f"{video_number_str}_annotated.mp4")
            visualize_command = [python_executable, 'src/visualize_results.py', '--video_path', video_path, '--json_path', result_json_path, '--output_path', annotated_video_path]
            try:
                print(f"正在为视频 {video_number_str} 生成可视化结果...")
                subprocess.run(visualize_command, check=True, capture_output=True, text=True, timeout=300)
                print(f"可视化视频已生成: {annotated_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"错误: 运行可视化脚本处理视频 {video_number_str} 时失败: {e.stderr}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量处理所有视频，计算IoU并进行可视化。")
    parser.add_argument('--videos_dir', type=str, default='sample_videos', help='存放所有输入视频的目录。')
    parser.add_argument('--main_json_path', type=str, default='sample_video.json', help='包含所有任务描述和真值的主JSON文件。')
    parser.add_argument('--api_key', type=str, required=True, help='你的智谱AI API密钥。')
    parser.add_argument('--output_dir', type=str, default='output_batch', help='存放所有输出结果的目录。')
    parser.add_argument('--visualize', action='store_true', help='是否为每个视频生成带标注的可视化结果。')
    
    args = parser.parse_args()
    
    process_all_videos(args)

    print("\n--- 所有任务执行完毕 ---")