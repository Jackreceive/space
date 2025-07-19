# src/detector.py (最终推荐版)

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import supervision as sv
import numpy as np
import cv2

class Detector:
    def __init__(self, model_path='IDEA-Research/grounding-dino-base'):
        """
        使用Hugging Face Transformers库初始化Grounding DINO模型。
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Grounding DINO 检测器将在 {self.device} 上运行。")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
            print("Grounding DINO 模型初始化完成。")
        except Exception as e:
            print(f"错误：加载Grounding DINO模型失败。请检查网络连接和模型路径 '{model_path}'。")
            print(f"详细错误: {e}")
            raise

    def detect_object(self, frame: np.ndarray, text_prompt: str) -> tuple[int, int, int, int] | None:
        """
        使用 Grounding DINO 检测与文本短语匹配的物体。
        """
        # print(text_prompt)
        text_prompt=" a red thing"
        if not isinstance(text_prompt, str) or not text_prompt:
            print(f"警告: 传入了无效的文本提示 '{text_prompt}'，跳过检测。")
            return None
            
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.1,
            text_threshold=0.1,
            target_sizes=[image_pil.size[::-1]]
        )[0] # 取出第一张（也是唯一一张）图片的结果
        
        # 确保有检测结果，results["scores"]的元素数量不为0
        if results["scores"].nelement() == 0:
            return None

        # --- 核心修改：手动从 results 字典中提取数据来创建 Detections 对象 ---
        # 将 PyTorch Tensors 转换为 NumPy 数组
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        # 使用这些NumPy数组直接创建 Detections 对象，绕过不兼容的 from_transformers 函数
        detections = sv.Detections(xyxy=boxes, confidence=scores)
        # -----------------------------------------------------------------

        if len(detections) > 0:
            # supervision 会根据置信度自动排序，我们取第一个
            best_box_xyxy = detections.xyxy[0]
            
            # 将 xyxy 格式 [xmin, ymin, xmax, ymax] 转换为中心点 xywh 格式
            xmin, ymin, xmax, ymax = best_box_xyxy
            width = xmax - xmin
            height = ymax - ymin
            x_center = xmin + width / 2
            y_center = ymin + height / 2
            
            return (int(x_center), int(y_center), int(width), int(height))
            
        return None