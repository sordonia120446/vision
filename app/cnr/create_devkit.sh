#!/usr/bin/env bash
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WORKING_DIR="CNR_devkit"
TARGET_DIR="/home/sam/Documents/sjsu/yolo_v3_pytorch_better/app/object_detection/data"
# TARGET_DIR="/home/sam/Documents/workspace/deep_learning/darknet/data"

echo "Target directory: $TARGET_DIR"

echo "Creating CNR Park YOLO directories"

mkdir -p $WORKING_DIR/data/yolo_drawings

mkdir -p $WORKING_DIR/data/Images
mkdir -p $WORKING_DIR/data/labels

echo -e "\nCompiling ${CYAN}CNR Park${NC} ground truth bounding boxes \n"
python app/cnr/draw_bounding_boxes.py \
    --yolo \
    --images_dir CNR_park \
    --target_dir $TARGET_DIR \
    --working_dir $WORKING_DIR

echo -e "\n${GREEN}Compilation complete. Results saved to app directory${NC} \n"
