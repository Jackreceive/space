# src/detector.py (使用 ultralytics 库的正确版本)

import torch
from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_path='yolov8l-world.pt'):
        """
        使用 ultralytics 库初始化 YOLOv8-world 模型。
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLO-World 检测器将在 {self.device} 上运行。")
        
        # 这是加载 YOLO .pt 文件的唯一正确方法
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_object(self, frame: np.ndarray, object_name: str) -> tuple[int, int, int, int] | None:
        """
        使用加载好的 YOLO 模型检测指定名称的物体。
        """
        if not isinstance(object_name, str) or not object_name:
            print(f"警告: 传入了无效的目标名称 '{object_name}'，跳过检测。")
            return None

        # 为 world 模型设置要寻找的类别
        self.model.set_classes([object_name])

        # 执行推理
        results = self.model(frame, verbose=False)

        # YOLO 的返回结果是一个列表，我们取第一个
        if results and results[0].boxes:
            result_for_frame = results[0]
            
            # 找到置信度最高的那个边界框
            best_box = None
            max_conf = 0
            for box in result_for_frame.boxes:
                if box.conf[0] > max_conf:
                    max_conf = box.conf[0]
                    best_box = box

            if best_box is not None:
                # 提取坐标，并从 NumPy 类型转换为标准的 Python int
                x_center, y_center, width, height = best_box.xywh[0].cpu().numpy()
                return (int(x_center), int(y_center), int(width), int(height))

        # 如果没有找到任何物体
        return None