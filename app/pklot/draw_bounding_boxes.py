import argparse
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.misc import imsave

import ground_truth_bbox as GroundTruthBBoxBL

# Constants and module-level vars
YOLO_CAR_ID = 0
VALIDATION_RATIO = 0.1  # 10% of training data for validation/testing


def parse_args():
    parser = argparse.ArgumentParser(description='Draw ground truth bounding boxes.')
    parser.add_argument('--images_dir', help='Images directory. Includes the labels XML files.')
    parser.add_argument('--target_dir', help='Where YOLO is going to iterate across the training data.')
    parser.add_argument('--working_dir', help='The folder housing the images, list files, and YOLO-formatted labels.')
    parser.add_argument('--verbose', action='store_true', help='Stdout progress across PKLot dataset.')
    parser.add_argument('--yolo', action='store_true', help='Output ground truth in YOLO format')

    args = parser.parse_args()
    return args


def draw(bb, im, color):
    """
    Be sure the image input is read via Pillow.
    """
    L = int(bb[0]); T = int(bb[1]); R = int(bb[2]); B = int(bb[3])

    # draw horizontal line
    for x in range(L, R+1):
        for channel in range(len(color)):
            im[channel][x][T] = color[channel]  # top line
            im[channel][x][B] = color[channel]  # bottom line

    # draw vertical line
    for y in range(T, B+1):
        for channel in range(len(color)):
            im[channel][L][y] = color[channel]  # left line
            im[channel][R][y] = color[channel]  # right line

    return im


def draw_yolo(labels, im, size):
    color = [0, 255, 0]  # green
    im_updated = np.copy(np.array(im).T)
    for label in labels:
        yolo_bb = GroundTruthBBoxBL.load_yolo_label(label)
        bb = GroundTruthBBoxBL.map_gt_bounding_box(yolo_bb, size)
        im_updated = draw(bb, im_updated, color)
    return im_updated


def log_info(msg, verbose=False):
    if verbose:
        print(msg)


def convert_annotations(labels, img_size):
    """Convert PKLot XML things to YOLO format."""
    yolo_labels = []
    for label in labels:
        # add label to output list
        # each labels file must have the format: class_id x_center, y_center, width, height
        # e.g.,
        # 6 0.551 0.600600600601 0.898 0.798798798799
        # 6 0.127 0.493993993994 0.246 0.357357357357
        bounding_box = GroundTruthBBoxBL.map_yolo_bounding_box(label, img_size)
        bb_str = ' '.join([str(coordinate) for coordinate in bounding_box])
        yolo_label = f'{YOLO_CAR_ID} {bb_str}\n'
        yolo_labels.append(yolo_label)
    return yolo_labels


def get_img_size(img_path):
    img = cv2.imread(img_path, 0)
    height, width = img.shape[:2]
    return (width, height)


def main(args):
    # create train and test list files in the target_dir
    working_dir = os.path.join(args.working_dir, 'data')
    list_train_filename = os.path.join(working_dir, f'{args.working_dir}_train.txt')
    list_test_filename = os.path.join(working_dir, f'{args.working_dir}_test.txt')
    list_file_train = open(list_train_filename, 'w')
    list_file_test = open(list_test_filename, 'w')

    # iterate through images_dir
    ind = 0
    for subdir, dirs, files in os.walk(args.images_dir):
        log_info(f'Iterating through subdir {subdir}', verbose=args.verbose)
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                img_path = os.path.join(subdir, file)
                labels_path = GroundTruthBBoxBL.derive_labels_filepath(img_path)
                labels = GroundTruthBBoxBL.parse_labels_file(labels_path, only_occupied=True)
                if not labels:
                    continue  # seems to only happen for one missing labels file
                img_size = get_img_size(img_path)
                yolo_labels = convert_annotations(labels, img_size)

                # store the image and YOLO-formatted labels file in the working_dir
                target_img_path = os.path.join(working_dir, 'Images', file)
                target_labels_path = os.path.join(
                    working_dir, 'labels',
                    GroundTruthBBoxBL.derive_yolo_labels_path(file))
                shutil.copy2(img_path, target_img_path)  # be sure to COPY, not MOVE
                with open(target_labels_path, 'w') as f_out:
                    f_out.writelines(yolo_labels)

                # optionally draw the annotations onto the image
                # store to a separate folder in working_dir
                if args.yolo:
                    im = Image.open(img_path).convert('RGB')  # use Pillow to read file
                    img_drawing = draw_yolo(labels=yolo_labels, im=im, size=img_size)
                    output_filepath = os.path.join(working_dir, 'yolo_drawings', file)
                    with open(output_filepath, 'w') as f_out:
                        imsave(f_out, img_drawing.T)

                # Write image path to train.txt or test.txt
                # This is NOT the image path according to this directory.
                # It is the image path to the target directory that YOLO will be run.
                abs_img_path = os.path.join(args.target_dir, target_img_path)
                if ind % round(1/VALIDATION_RATIO) > 0:
                    list_file_train.write(abs_img_path + '\n')
                else:
                    list_file_test.write(abs_img_path + '\n')
                ind += 1
        print(f'Converted {round(len(files)/2)} PKLot images')

    # Close all files
    list_file_train.close()
    list_file_test.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
