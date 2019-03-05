#!/usr/bin/env bash
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

PUCPR_DIR="PUCPR_devkit"
CARPK_DIR="CARPK_devkit"
TARGET_DIR="/home/sam/Documents/sjsu/yolo_v3_pytorch_better/app/object_detection/data"

echo "Compiling PUCPR and CARPK"

echo "Target directory: $TARGET_DIR"

echo -e "\nCompiling ${CYAN}PUCPR${NC} ground truth bounding boxes \n"
mkdir -p $PUCPR_DIR/data/labels
mkdir -p $PUCPR_DIR/data/yolo_drawings
python app/carpk/draw_bounding_boxes.py \
    --yolo --annots_dir $PUCPR_DIR/data/Annotations/ \
    --images_dir $PUCPR_DIR/data/Images/ \
    --target_dir $TARGET_DIR

echo -e "\nCompiling ${CYAN}CARPK${NC} ground truth bounding boxes \n"
mkdir -p $CARPK_DIR/data/labels
mkdir -p $CARPK_DIR/data/yolo_drawings
python app/carpk/draw_bounding_boxes.py \
    --yolo --annots_dir $CARPK_DIR/data/Annotations/ \
    --images_dir $CARPK_DIR/data/Images/ \
    --target_dir $TARGET_DIR

echo -e "\n${GREEN}Compilation complete. Results saved to app directory${NC} \n"
