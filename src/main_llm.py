# src/main_llm.py (使用官方 zhipuai 库的最终版)
import argparse
import os
from zhipuai import ZhipuAI # <--- 关键修改点 1：导入官方库
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
    使用智谱AI官方SDK，生成一个精确的、带有指代信息的英文短语。
    """
    def __init__(self, api_key: str):
        print("开始初始化智谱AI API (官方SDK模式)...")
        if not api_key:
            raise ValueError("错误: 未提供智谱AI API 密钥。")
        
        # --- 关键修改点 2：使用官方库进行初始化 ---
        self.client = ZhipuAI(api_key=api_key)
        # ----------------------------------------
        print("智谱AI API 初始化完成。")

    def _encode_image_to_base64(self, frame: Image.Image) -> str:
        buffered = io.BytesIO()
        frame.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def refine_query(self, frame: Image.Image, complex_query: str) -> str:
        base64_image = self._encode_image_to_base64(frame)
        
        phrase_prompt = (
            "You are a visual grounding assistant. Your task is to respond to a question about an image by creating a concise descriptive phrase. "
            "The subject of this phrase MUST be the direct answer to the question. "
            "The phrase should include descriptive modifiers (like color or position) to uniquely identify the subject. "
            "Do NOT write a full sentence, only the descriptive phrase itself.\n\n"
            "Here are some examples of perfect responses:\n"
            "Question: 'who holds hand of the other adult on stage?' -> 'the man in the blue suit'\n"
            "Question: 'what is next to the other squirrel?' -> 'the squirrel on the grass'\n"
        )
        
        print("正在调用智谱AI API (glm-4v) 生成指代短语...")
        try:
            # --- 关键修改点 3：API调用结构保持不变，因为官方库兼容OpenAI格式 ---
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
                max_tokens=40,
                temperature=0.0,
            )
            
            refined_phrase = response.choices[0].message.content.strip().replace("'", "").replace('"', '')
            print(f"复杂查询 '{complex_query}' 被精炼为 -> 指代短语: '{refined_phrase}'")
            return refined_phrase
            
        except Exception as e:
            print(f"调用智谱AI API 时发生错误: {e}")
            return ""

# (main 函数和 __main__ 部分无需修改，但为保证完整性，全部贴出)
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
    
    middle_frame_idx = start_frame 
    print(f"选择第 {middle_frame_idx} 帧（中间帧）发送给大模型进行分析...")
    
    frame_for_api_np = read_single_frame(video_path, middle_frame_idx)
    
    if frame_for_api_np is None:
        print("错误：无法读取用于分析的中间帧，程序终止。")
        return
        
    frame_for_api_pil = Image.fromarray(frame_for_api_np)
    
    refined_phrase = query_refiner.refine_query(frame_for_api_pil, complex_query)
    
    if not refined_phrase:
        print("错误: API未能从查询中提炼出有效的指代短语。程序终止。")
        return

    print("\n--- 开始目标检测与追踪流程 ---")
    detector = Detector(model_path='IDEA-Research/grounding-dino-base')
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
        x_min, y_min, x_max, y_max = cx - w // 2, cy - h // 2, cx + w//2,cy + h//2
        all_bboxes[str(start_frame)] = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        print(f"目标已找到，边界框: {all_bboxes[str(start_frame)]}，开始追踪...")
    else:
        all_bboxes[str(start_frame)] = {}
        print("警告：在第一帧未找到目标。")

    frame_count = end_frame - start_frame
    for i, frame in enumerate(tqdm(frame_generator, total=max(0, frame_count), desc="追踪进度")):
        if i == 0 and frame_count >= 0: continue
        frame_idx = start_frame + i
        success, new_bbox = tracker.update(frame)
        
        current_bbox_for_json = {} 
        if success:
            cx, cy, w, h = new_bbox
            x_min, y_min, x_max, y_max = cx - w // 2, cy - h // 2, cx + w //2, cy + h //2 
            current_bbox_for_json = {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}
        else:
            redetected_bbox = detector.detect_object(frame, refined_phrase)
            if redetected_bbox:
                tracker.initialize(frame, redetected_bbox)
                cx, cy, w, h = redetected_bbox
                x_min, y_min, x_max, y_max = cx - w // 2, cy - h // 2, cx + w //2, cy + h //2
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="集成智谱AI API和Grounding DINO的视频时空定位脚本")
    parser.add_argument('--video_path', type=str, required=True, help='输入视频文件的路径')
    parser.add_argument('--json_path', type=str, required=True, help='包含任务描述的JSON文件路径')
    parser.add_argument('--api_key', type=str, required=True, help='你的智谱AI API 密钥')
    parser.add_argument('--output_path', type=str, default='output/results_zhipu_api.json', help='输出结果JSON文件的路径')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    main(args)