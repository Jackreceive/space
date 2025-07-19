# src/iou_calculator.py

def calculate_iou(boxA, boxB):
    """
    计算两个边界框的交并比 (IoU)。

    Args:
        boxA (dict): 第一个边界框, 格式为 {'xmin':, 'ymin':, 'xmax':, 'ymax':}
        boxB (dict): 第二个边界框, 格式与 boxA 相同。

    Returns:
        float: 交并比的值，范围在 0.0 到 1.0 之间。
    """
    # 确定相交矩形的坐标
    xA = max(boxA['xmin'], boxB['xmin'])
    yA = max(boxA['ymin'], boxB['ymin'])
    xB = min(boxA['xmax'], boxB['xmax'])
    yB = min(boxA['ymax'], boxB['ymax'])

    # 计算相交区域的面积
    # 如果没有相交，宽度或高度会是负数，面积为0
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个边界框各自的面积
    boxAArea = (boxA['xmax'] - boxA['xmin']) * (boxA['ymax'] - boxA['ymin'])
    boxBArea = (boxB['xmax'] - boxB['xmin']) * (boxB['ymax'] - boxB['ymin'])

    # 计算并集面积
    unionArea = float(boxAArea + boxBArea - interArea)

    # 计算交并比
    iou = interArea / unionArea
    
    # 返回交并比
    return iou