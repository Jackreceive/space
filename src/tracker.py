# tracker.py
import cv2

class Tracker:
    """
    OpenCV CSRT 跟踪器的封装。
    """
    def __init__(self, tracker_type='CSRT'):
        """
        初始化跟踪器。
        """
        self.tracker_type = tracker_type
        self.tracker = None
        print(f"跟踪器已初始化，类型为：{self.tracker_type}")

    def initialize(self, frame, bbox):
        """
        使用第一帧和边界框初始化跟踪器。

        Args:
            frame: 视频的第一帧。
            bbox: 物体的初始边界框 (cx, cy, w, h)。
        """
        # CSRT 跟踪器需要 (x_min, y_min, w, h) 格式
        cx, cy, w, h = bbox
        x_min = cx - w // 2
        y_min = cy - h // 2
        init_bbox_format = (x_min, y_min, w, h)

        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, init_bbox_format)
        print(f"跟踪器已使用边界框 {init_bbox_format} 初始化")

    def update(self, frame):
        """
        使用新帧更新跟踪器。

        Args:
            frame: 用于跟踪物体的新帧。

        Returns:
            一个元组 (success, new_bbox)，其中 success 是布尔值，new_bbox 是更新后的边界框 [cx, cy, w, h]。
        """
        if self.tracker is None:
            return False, None

        success, bbox = self.tracker.update(frame)
        
        if success:
            # 转换回 (cx, cy, w, h) 格式
            x_min, y_min, w, h = map(int, bbox)
            cx = x_min + w // 2
            cy = y_min + h // 2
            return True, (cx, cy, w, h)
        else:
            return False, None