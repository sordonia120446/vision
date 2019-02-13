import re

import numpy as np


def load_gt_bbox(filepath):
    with open(filepath) as f:
        data = f.read()
    objs = re.findall('\d+ \d+ \d+ \d+ \d+', data)

    # Creates ground truth bounding box matrix of nums_obj x 4
    nums_obj = len(objs)
    gtBBs = np.zeros((nums_obj, 4))
    for idx, obj in enumerate(objs):
        info = re.findall('\d+', obj)
        x1 = float(info[0])
        y1 = float(info[1])
        x2 = float(info[2])
        y2 = float(info[3])
        gtBBs[idx, :] = [x1, y1, x2, y2]
    return gtBBs


def load_yolo_label(label):
    """
    Parses label and returns (x_center, y_center, width, height).
    Note the 0th element is the class ID.

    label <str>:  e.g., '6 0.551 0.600600600601 0.898 0.798798798799'
    """
    yolo_bb = label.split(' ')[1:]
    return [el.strip() for el in yolo_bb]


def map_yolo_bounding_box(gt_bounding_box, size):
    """
    Converts the input bounding box into YOLO format. Scales based on img size.
    Args:
        gt_bounding_box: straight from load_gt_bbox
        size: (width, height) tuple or list. These are the width and height of the ENTIRE image.
    Returns:
        The YOLO output of (x_center, y_center, width, height)
    """
    (x1, y1, x2, y2) = gt_bounding_box

    # This stuff taken from the YOLO VOC labels script
    box = (x1, x2, y1, y2)
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def map_gt_bounding_box(yolo_bb, size):
    x_center = float(yolo_bb[0])
    y_center = float(yolo_bb[1])
    width_bb = float(yolo_bb[2])
    height_bb = float(yolo_bb[3])
    (width, height) = size  # should already be int

    x1 = round((x_center - width_bb / 2)*width)
    y1 = round((y_center - height_bb / 2)*height)
    x2 = round((x_center + width_bb / 2)*width)
    y2 = round((y_center + height_bb / 2)*height)

    return [x1, y1, x2, y2]
