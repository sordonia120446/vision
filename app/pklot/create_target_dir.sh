#!/usr/bin/env bash
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

PKLOT_DIR="PKLOT_devkit"
TARGET_DIR="/home/sam/Documents/sjsu/yolo_v3_pytorch_better/app/object_detection/data"

echo "Compiling PKLOT"

echo "Target directory: $TARGET_DIR"

echo "Creating PKLOT YOLO directories"

mkdir -p $PKLOT_DIR/data/yolo_drawings

mkdir -p $PKLOT_DIR/data/Images
mkdir -p $PKLOT_DIR/data/labels

echo -e "\nCompiling ${CYAN}PKLOT${NC} ground truth bounding boxes \n"
python app/pklot/draw_bounding_boxes.py \
    --yolo \
    --images_dir PKLot_data \
    --target_dir $TARGET_DIR \
    --working_dir $PKLOT_DIR

echo -e "\n${GREEN}Compilation complete. Results saved to app directory${NC} \n"
