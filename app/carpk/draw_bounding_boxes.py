import argparse
import os

import numpy as np
from PIL import Image
import scipy
from scipy.signal import *

import cv2

import ground_truth_bbox as GroundTruthBBoxBL


# Constants and module-level vars
# YOLO_CAR_ID = 80  # based on coco.names
YOLO_CAR_ID = 0  # single-class identifier
VALIDATION_RATIO = 0.1  # 10% of training data for validation/testing


def parse_args():
    parse = argparse.ArgumentParser(description='Draw ground truth bounding boxes.')
    # TODO remove these dest things, they not needed
    parse.add_argument('--annots_dir', dest='annots_dir', help='Annotations directory')
    parse.add_argument('--images_dir', dest='images_dir', help='Images directory')
    parse.add_argument('--target_dir', help='Where the data files are going to go')
    parse.add_argument('--draw', action='store_true', dest='draw_bb', help='Draw bounding boxes and output to output_dir')
    parse.add_argument('--yolo', action='store_true', help='Output ground truth in YOLO format')

    args = parse.parse_args()
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


def draw_bounding_boxes(args):
    """Wrapper function on the original CARPK and PUCPR+ dataset bounding box drawing."""
    images_base = args.images_dir.split('/')[0]
    for image_filename in os.listdir(args.images_dir):
        base_filename = os.path.splitext(image_filename)[0]

        # load image
        im = Image.open(os.path.join(args.images_dir, image_filename)).convert('RGB')
        im = np.array(im).T
        im_copy = np.copy(im)

        # load ground truth bounding box
        annot_filename = f'{base_filename}.txt'
        gtBBs = GroundTruthBBoxBL.load_gt_bbox(os.path.join(args.annots_dir, annot_filename))

        # draw ground truth annotations on the image
        for bb in gtBBs:
            draw(bb, im_copy, color=[0, 255, 0])  # color: green

        # output the result image
        # be sure output_dir is created beforehand with mkdir or something
        output_dir = os.path.join(images_base, 'data', 'output_images')
        output_filepath = os.path.join(output_dir, f'{base_filename}_gt_bbox.jpg')
        with open(output_filepath, 'w+') as output_fobj:
            scipy.misc.imsave(output_fobj, im_copy.T)
            print(f'Saving... {output_filepath}')


def convert_annotation(gtBBs, size):
    """
    Converts annotations to labels.
    Args:
        gtBBs: list of ground truth bounding boxes in CARPK format
        size: (width, height) tuple or list. These are the width and height of the ENTIRE image.
    """
    labels = []
    for gt_bounding_box in gtBBs:
        # convert the bb's into YOLO-friendly format (the correct order of coordinates)
        bounding_box = GroundTruthBBoxBL.map_yolo_bounding_box(gt_bounding_box, size)

        # add label to output list
        # each labels file must have the format: class_id x_center, y_center, width, height
        # e.g.,
        # 6 0.551 0.600600600601 0.898 0.798798798799
        # 6 0.127 0.493993993994 0.246 0.357357357357
        # label = str(cls_id) + " " + " ".join([str(a) for a in bounding_box]) + '\n'
        bb_str = ' '.join([str(coordinate) for coordinate in bounding_box])
        label = f'{YOLO_CAR_ID} {bb_str}\n'
        labels.append(label)

    return labels


def convert_to_yolo(args):
    input_folder = args.images_dir.split('/')[0]  # hacky but w/e

    # This assumes current working directory is a super-directory of the data folder
    working_dir = os.path.join(os.getcwd(), input_folder, 'data')

    # write all image paths to the train or test files
    list_train_filename = os.path.join(working_dir, f'{input_folder}_train.txt')
    list_test_filename = os.path.join(working_dir, f'{input_folder}_test.txt')
    list_file_train = open(list_train_filename, 'w')
    list_file_test = open(list_test_filename, 'w')

    # write to a labels directory a set of files each corresponding
    # to a different image file.
    for ind, image_filename in enumerate(os.listdir(args.images_dir)):
        base_filename = os.path.splitext(os.path.basename(image_filename))[0]

        # Write image path to train.txt or test.txt
        # This is NOT the image path according to this directory.
        # It is the image path to the target directory that YOLO will be run.
        img_path = os.path.join(args.target_dir, args.images_dir, image_filename)
        if ind % round(1/VALIDATION_RATIO) > 0:
            list_file_train.write(img_path + '\n')
        else:
            list_file_test.write(img_path + '\n')

        # load ground truth bounding box
        gtBBs = GroundTruthBBoxBL.load_gt_bbox(os.path.join(args.annots_dir, f'{base_filename}.txt'))

        # get width and height of image_filename
        img_path_true = os.path.join(args.images_dir, image_filename)
        img = cv2.imread(img_path_true, 0)
        height, width = img.shape[:2]
        size = (width, height)

        # write to labels file
        labels = convert_annotation(gtBBs, size)

        with open(os.path.join(working_dir, 'labels', f'{base_filename}.txt'), 'w') as f_out:
            f_out.writelines(labels)

        # TODO add some args thing here
        if True:
            # load image with Pillow, NOT open CV. cuz if it ain't broke :)
            im = Image.open(img_path_true).convert('RGB')
            img_drawing = draw_yolo(labels, im, size)
            output_filepath = os.path.join(working_dir, 'yolo_drawings', f'{base_filename}.jpg')
            with open(output_filepath, 'w') as f_out:
                scipy.misc.imsave(f_out, img_drawing.T)

    # Close all files
    list_file_train.close()
    list_file_test.close()


def main():
    args = parse_args()

    if args.draw_bb:
        print('Drawing bounding boxes onto images')
        draw_bounding_boxes(args)
    else:
        print('Skipping grouth truth annotation drawing')

    if args.yolo:
        print(f'Outputting annotations into YOLO format')
        convert_to_yolo(args)
    else:
        print('Skipping YOLO compilation')


if __name__ == '__main__':
    main()
