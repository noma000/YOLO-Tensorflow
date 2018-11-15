import numpy as np

def IOU(offset_xy, offset_xy2):
    area1 = np.product(np.subtract(offset_xy[1], offset_xy[0]))
    area2 = np.product(np.subtract(offset_xy2[1], offset_xy2[0]))
    left_up = np.maximum(offset_xy[0], offset_xy2[0])
    right_down = np.minimum(offset_xy[1], offset_xy2[1])
    # inter_area  = np.product(np.subtract(left_up,right_down))
    inter_area = np.product(np.maximum([0, 0], np.subtract(right_down, left_up)))
    union = np.maximum(area1 + area2 - inter_area, 1e-6)
    return inter_area / union

def Inarea(coord):
    cx,cy,w,h = coord
    if (cx + w * .5 < 0.0) | (cx - w * .5 > 1.0) | (cy + h * .5 < 0.0) | (cy - h * .5 > 1.0):
        return False
    else:
        return True