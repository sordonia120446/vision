import csv
import os


images_folder = 'FULL_IMAGE_1000x750'


def map_yolo_bounding_box(label, img_size):
    """
    Converts the input bounding box into YOLO format. Scales based on img size.
    Args:
        label: all you ever needed baby
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


def _camera_name_lookup(camera_id):
    return {
        'C01': 'camera1',
        'C02': 'camera2',
        'C03': 'camera3',
        'C04': 'camera4',
        'C05': 'camera5',
        'C06': 'camera6',
        'C07': 'camera7',
        'C08': 'camera8',
        'C09': 'camera9',
    }[camera_id]


def _scale_size(x, y):
    """convert the size here to scale from 2592x1944 -> 1000x750."""
    x_scaled = x/2592*1000
    y_scaled = y/1944*750
    return x_scaled, y_scaled


def index_labels(labels_path, images_dir):
    """Reads in the labels file and reconstructs the image path and formats the labels."""
    all_labels = {}
    with open(labels_path) as labels_file:
        for line in labels_file.readlines():
            label = line.strip()
            img_path, is_occupied = label.split(' ')
            parts = img_path.split('/')
            base, ext = os.path.splitext(parts[-1])
            weather_id, capture_date, capture_time, cam_id, slot_id = base.split('_')
            capture_time_joined = ''.join(capture_time.split('.'))
            img_file_basename = f'{capture_date}_{capture_time_joined}{ext}'
            img_file = os.path.join(
                images_dir,
                images_folder,
                parts[0],  # OVERCAST, RAINY, or SUNNY
                parts[1],  # capture_time
                parts[2],  # camera name
                img_file_basename,
            )

            img_labels = all_labels.get(img_file, [])
            img_labels.append({
                'is_occupied': bool(int(is_occupied)),
                'slot': int(slot_id),
                'camera': _camera_name_lookup(cam_id),
            })
            all_labels[img_file] = img_labels
    return all_labels


def index_cameras(source_dir):
    cameras = {}
    for file in os.listdir(source_dir):
        camera_name = os.path.splitext(file)[0]
        slots = {}
        filepath = os.path.join(source_dir, file)
        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                slot_id = int(row['SlotId'])

                x, y = _scale_size(int(row['X']), int(row['Y']))
                w, h = _scale_size(int(row['H']), int(row['W']))
                x_center = x + w/2.0
                y_center = y + h/2.0
                slots[slot_id] = {
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': w,
                    'height': h,
                }
        cameras[camera_name] = slots
    return cameras


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
    x2 = min(width - 1, round((x_center + width_bb / 2)*width))
    y2 = min(height - 1, round((y_center + height_bb / 2)*height))

    return [x1, y1, x2, y2]
