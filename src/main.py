# src/main_llm.py (视频输入最终版)
import argparse
import os
from zhipuai import ZhipuAI # 导入智谱AI官方库
from PIL import Image
from tqdm import tqdm
import re

# 从其他模块导入
from data_loader import load_video_data
from detector import Detector
from tracker import Tracker
from utils import read_video_frames, save_results_to_json

class APIQueryRefiner:
    """
    使用智谱AI (GLM-4V) 的视频理解能力来生成精确的指代短语。
    """
    def __init__(self, api_key: str):
        print("开始初始化智谱AI API (视频模式)...")
        if not api_key:
            raise ValueError("错误: 未提供智谱AI API 密钥。")
        
        self.client = ZhipuAI(api_key=api_key)
        print("智谱AI API 初始化完成。")

    def refine_query(self, video_path: str, complex_query: str) -> str:
        """
        上传视频并发送复杂问题给智谱AI API，返回一个精确的英文指代短语。
        """
        print(f"正在上传视频文件: {video_path} ...")
        try:
            # 步骤 1: 上传文件
            with open(video_path, "rb") as video_file:
                file_obj = self.client.files.create(file=video_file, purpose="vision")
            
            print(f"视频上传成功，文件ID: {file_obj.id}")

            # 步骤 2: 构建指令 (Prompt)
            phrase_prompt = (
                "You are a computer vision assistant. Your task is to analyze the provided video and the user's question to create a concise English phrase that uniquely identifies the target object. "
                "This phrase will be fed into a visual grounding model. It must be descriptive and include spatial relationships if necessary. "
                "Do NOT just respond with a single word. Be specific.\n\n"
                "Here are some examples of perfect responses:\n"
                "Question: 'what is next to the other squirrel?' -> 'the squirrel on the left'\n"
                "Question: 'what does the adult woman in black hold in the living room?' -> 'the book in her hands'\n"
            )
            
            print("正在调用智谱AI API (glm-4v) 分析视频内容...")
            
            # 步骤 3: 发送带有文件ID的聊天请求
            response = self.client.chat.completions.create(
                model="glm-4v",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{phrase_prompt}Now, analyze this video.\nQuestion: '{complex_query}' ->"},
                        {"type": "file_url", "file_url": {"url": file_obj.id}} # <--- 关键修改点
                    ]
                }],
                max_tokens=20,
                temperature=0.0,
            )
            
            refined_phrase = response.choices[0].message.content.strip().replace("'", "").replace('"', '')
            print(f"复杂查询 '{complex_query}' 被精炼为 -> 指代短语: '{refined_phrase}'")
            return refined_phrase
            
        except Exception as e:
            print(f"调用智谱AI API 或上传文件时发生错误: {e}")
            return ""

def main(args):
    """主执行函数"""
    print(f"开始处理视频: {args.video_path}")
    print(f"使用JSON任务文件: {args.json_path}")

    try:
        query_refiner = APIQueryRefiner(api_key=args.api_key)
    except ValueError as e:
        print(e)
        return

    video_filename = os.path.basename(args.video_path)
    
    try:
        video_path, start_frame, end_frame, complex_query, _ = load_video_data(args.json_path, video_filename)
        print(f"任务加载成功: 在 {start_frame}-{end_frame} 帧之间寻找与 '{complex_query}' 相关的内容。")
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        return
    
    # --- 核心修改：直接将视频路径传递给 refiner ---
    # 我们不再需要手动提取第一帧给API
    refined_phrase = query_refiner.refine_query(video_path, complex_query)
    
    if not refined_phrase:
        print("错误: API未能从查询中提炼出有效的指代短语。程序终止。")
        return

    # --- 后续流程不变，仍然是对帧进行检测和追踪 ---
    print("\n--- 开始目标检测与追踪流程 ---")
    detector = Detector(model_path='yolov8l-world.pt')
    tracker = Tracker(tracker_type='CSRT')

    frame_generator = read_video_frames(video_path, start_frame, end_frame)
    try:
        first_frame_np = next(frame_generator)
    except StopIteration:
        print("错误：视频帧区间为空或无法读取第一帧。")
        return

    print(f"正在第一帧使用短语 '{refined_phrase}' 进行初始目标检测...")
    initial_bbox = detector.detect_object(first_frame_np, refined_phrase)
    all_bboxes = {}
    
    if initial_bbox:
        tracker.initialize(first_frame_np, initial_bbox)
        cx, cy, w, h = initial_bbox
        x_min, y_min, x_max, y_max = cx - w // 2, cy - h // 2, cx + w, cy + h
        all_bboxes[str(start_frame)] = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        print(f"目标已找到，边界框: {all_bboxes[str(start_frame)]}，开始追踪...")
    else:
        all_bboxes[str(start_frame)] = {}
        print("警告：在第一帧未找到目标。")

    frame_count = end_frame - start_frame
    for i, frame in enumerate(tqdm(frame_generator, total=max(0, frame_count), desc="追踪进度")):
        if i == 0 and frame_count > 0: continue # 跳过第一帧
        frame_idx = start_frame + i
        success, new_bbox = tracker.update(frame)
        
        current_bbox_for_json = {} 
        if success:
            cx, cy, w, h = new_bbox
            x_min, y_min, x_max, y_max = cx - w // 2, cy - h // 2, cx + w, y_min + h
            current_bbox_for_json = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        else:
            redetected_bbox = detector.detect_object(frame, refined_phrase)
            if redetected_bbox:
                tracker.initialize(frame, redetected_bbox)
                cx, cy, w, h = redetected_bbox
                x_min, y_min, x_max, y_max = cx - w // 2, cy - h // 2, cx + w, y_min + h
                current_bbox_for_json = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        
        all_bboxes[str(frame_idx)] = current_bbox_for_json

    video_key, _ = os.path.splitext(video_filename)
    match = re.search(r'(\d+)', video_key)
    if match: video_key = match.group(1)

    final_output = {
        video_key: {
            "query": complex_query,
            "refined_query": refined_phrase,
            "pred_bboxs": all_bboxes
        }
    }
    save_results_to_json(final_output, args.output_path)
    print(f"\n处理完成，结果已保存至 {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="集成智谱AI视频理解API的视频时空定位脚本")
    parser.add_argument('--video_path', type=str, required=True, help='输入视频文件的路径')
    parser.add_argument('--json_path', type=str, required=True, help='包含任务描述的JSON文件路径')
    parser.add_argument('--api_key', type=str, required=True, help='你的智谱AI API 密钥')
    parser.add_argument('--output_path', type=str, default='output/result.json', help='输出结果JSON文件的路径')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    main(args)