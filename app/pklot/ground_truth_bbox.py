import xml.etree.ElementTree


def map_yolo_bounding_box(label, img_size):
    """
    Converts the input bounding box into YOLO format. Scales based on img size.
    Args:
        labels: list of dict elements from parse_labels_file()
        img_size: (width, height) tuple or list. These are the width and height of the ENTIRE image.
    Returns:
        The YOLO output of (x_center, y_center, width, height)
    """
    # This stuff taken from the YOLO VOC labels script
    dw = 1./img_size[0]
    dh = 1./img_size[1]
    x = label['x_center']
    y = label['y_center']
    w = label['width']
    h = label['height']
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def parse_labels_file(labels_path, only_occupied=False):
    try:
        el = xml.etree.ElementTree.parse(labels_path).getroot()
    except FileNotFoundError:
        print(f'Could not find XML file {labels_path}')
        return []
    labels = []
    for space in el.findall('space'):
        center = {}
        size = {}
        is_occupied = bool(int(space.attrib.get('occupied', 0)))
        for child in space:
            if child.tag != 'rotatedRect':
                continue
            center = child.find('center').attrib
            size = child.find('size').attrib
        label = {
            'is_occupied': is_occupied,
            'x_center': int(center.get('x')),
            'y_center': int(center.get('y')),
            'width': int(size.get('w')),
            'height': int(size.get('h')),
        }
        if only_occupied and not label['is_occupied']:
            continue
        labels.append(label)
    return labels


def derive_labels_filepath(img_path):
    return img_path.replace('jpg', 'xml')


def derive_yolo_labels_path(img_path):
    return img_path.replace('jpg', 'txt')


def load_yolo_label(label):
    """
    Parses label and returns (x_center, y_center, width, height).
    Note the 0th element is the class ID.

    label <str>:  e.g., '6 0.551 0.600600600601 0.898 0.798798798799'
    """
    yolo_bb = label.split(' ')[1:]
    return [el.strip() for el in yolo_bb]


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
