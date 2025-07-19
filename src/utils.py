# src/utils.py

import cv2
import json

def read_video_frames(video_path, start_frame, end_frame):
    """
    (旧功能，保持不变)
    顺序读取从 start_frame 到 end_frame 的视频帧。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame += 1
        
    cap.release()

def read_single_frame(video_path, frame_number):
    """
    (新增功能)
    直接读取指定编号的单帧画面。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return None
    
    # 直接跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print(f"警告: 无法读取视频 {video_path} 的第 {frame_number} 帧。")
        return None

def save_results_to_json(data, output_path):
    """
    (旧功能，保持不变)
    将结果保存为JSON文件。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)