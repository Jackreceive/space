# src/main_llm.py (单帧图片 + 智谱API 最终稳定版)
import argparse
import os
from openai import OpenAI
import base64
import io
from PIL import Image
from tqdm import tqdm
import re

# 从其他模块导入
from data_loader import load_video_data
from detector import Detector
from tracker import Tracker
from utils import read_video_frames, save_results_to_json, read_single_frame

class APIQueryRefiner:
    """
    使用智谱AI，通过分析单帧图片，生成一个精确的、带有指代信息的英文短语。
    """
    def __init__(self, api_key: str):
        print("开始初始化智谱AI API (单帧图片模式)...")
        if not api_key:
            raise ValueError("错误: 未提供智谱AI API 密钥。")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        print("智谱AI API 初始化完成。")

    def _encode_image_to_base64(self, frame: Image.Image) -> str:
        """将 PIL Image 对象编码为 Base64 字符串"""
        buffered = io.BytesIO()
        frame.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def refine_query(self, frame: Image.Image, complex_query: str) -> str:
        """
        发送一帧图像和复杂问题给智谱AI API，返回一个精确的英文指代短语。
        """
        base64_image = self._encode_image_to_base64(frame)
        
        phrase_prompt = (
        "You are an expert visual assistant. Your task is to analyze an image and a complex question. "
        "Based on the question, identify the object or entity that is performing the action described in the question. "
        "If the question describes an action (e.g., 'who caresses the white cat at home?'), focus on the object or entity performing that action (e.g., 'hand'). "
        "You must respond with ONLY the name of that single object or entity and nothing else. "
        "For example, if the question is 'Who is holding the book?', and the image shows a hand holding a book, your response should be exactly 'hand'. "
        "If the question is 'Who caresses the white cat at home?', and the image shows a hand caressing a white cat, your response should be exactly 'hand'."
        )
        
        print("正在调用智谱AI API (glm-4v) 生成指代短语...")
        try:
            response = self.client.chat.completions.create(
                model="glm-4v",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{phrase_prompt}Now, create the phrase for this question and image.\nQuestion: '{complex_query}' ->"},
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" }
                        }
                    ]
                }],
                max_tokens=20,
                temperature=0.0,
            )
            
            refined_phrase = response.choices[0].message.content.strip().replace("'", "").replace('"', '')
            print(f"复杂查询 '{complex_query}' 被精炼为 -> 指代短语: '{refined_phrase}'")
            return refined_phrase
            
        except Exception as e:
            print(f"调用智谱AI API 时发生错误: {e}")
            return ""

# main_llm.py 中从 main 函数开始的部分

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
    
    # --- 核心修改：计算并读取中间帧 ---
    middle_frame_idx = start_frame + (end_frame - start_frame) // 2
    print(f"选择第 {middle_frame_idx} 帧（中间帧）发送给大模型进行分析...")
    
    # 使用新函数直接读取中间帧
    frame_for_api_np = read_single_frame(video_path, middle_frame_idx)
    
    if frame_for_api_np is None:
        print("错误：无法读取用于分析的中间帧，程序终止。")
        return
        
    frame_for_api_pil = Image.fromarray(frame_for_api_np)
    # ------------------------------------
    
    refined_phrase = query_refiner.refine_query(frame_for_api_pil, complex_query)
    
    if not refined_phrase:
        print("错误: API未能从查询中提炼出有效的指代短语。程序终止。")
        return

    print("\n--- 开始目标检测与追踪流程 ---")
    detector = Detector(model_path='yolov8l-world.pt')
    tracker = Tracker(tracker_type='CSRT')
    print("YOLO模型初始化完成。")

    # 注意：初始检测和追踪仍然从 start_frame 开始
    frame_generator = read_video_frames(video_path, start_frame, end_frame)
    try:
        first_frame_np = next(frame_generator)
    except StopIteration:
        print("错误：视频帧区间为空或无法读取第一帧。")
        return

    print(f"正在第一帧（第 {start_frame} 帧）使用短语 '{refined_phrase}' 进行初始目标检测...")
    initial_bbox = detector.detect_object(first_frame_np, refined_phrase)
    all_bboxes = {}
    
    if initial_bbox:
        tracker.initialize(first_frame_np, initial_bbox)
        cx, cy, w, h = initial_bbox
        x_min, y_min = cx - w // 2, cy - h // 2
        x_max, y_max = x_min + w, y_min + h
        all_bboxes[str(start_frame)] = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        print(f"目标已找到，边界框: {all_bboxes[str(start_frame)]}，开始追踪...")
    else:
        all_bboxes[str(start_frame)] = {}
        print("警告：在第一帧未找到目标。")

    frame_count = end_frame - start_frame
    for i, frame in enumerate(tqdm(frame_generator, total=max(0, frame_count), desc="追踪进度")):
        frame_idx = start_frame + i + 1
        success, new_bbox = tracker.update(frame)
        
        current_bbox_for_json = {} 
        if success:
            cx, cy, w, h = new_bbox
            x_min, y_min = cx - w // 2, cy - h // 2
            x_max, y_max = x_min + w, y_min + h
            current_bbox_for_json = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        else:
            redetected_bbox = detector.detect_object(frame, refined_phrase)
            if redetected_bbox:
                tracker.initialize(frame, redetected_bbox)
                cx, cy, w, h = redetected_bbox
                x_min, y_min = cx - w // 2, cy - h // 2
                x_max, y_max = x_min + w, y_min + h
                current_bbox_for_json = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        
        all_bboxes[str(frame_idx)] = current_bbox_for_json

    video_key, _ = os.path.splitext(video_filename)
    match = re.search(r'(\d+)', video_key)
    if match:
        video_key = match.group(1)

    final_output = {
        video_key: {
            "query": complex_query,
            "refined_query": refined_phrase,
            "pred_bboxs": all_bboxes
        }
    }
    save_results_to_json(final_output, args.output_path)
    print(f"\n处理完成，结果已保存至 {args.output_path}")

# __main__ 部分保持不变
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="集成智谱AI API的视频时空定位脚本")
    # ... (命令行参数部分无需改动)
    parser.add_argument('--video_path', type=str, required=True, help='输入视频文件的路径')
    parser.add_argument('--json_path', type=str, required=True, help='包含任务描述的JSON文件路径')
    parser.add_argument('--api_key', type=str, required=True, help='你的智谱AI API 密钥')
    parser.add_argument('--output_path', type=str, default='output/results_zhipu_api.json', help='输出结果JSON文件的路径')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    main(args)