#!/usr/bin/env bash

echo "Modifying the target dataset for anchor box calculation."
echo "$1 better be a train or test list file."

echo "Be sure to modify the TARGET_DIR variable in the data compilation process. It'll need to match the darknet folder"

echo -e "e.g., TARGET_DIR=/home/sam/Documents/workspace/deep_learning/darknet/data\n"

sed -i -e 's/Images/labels/g' $1

echo -e "\nAfter this is done, copy paste this into the darknet data folder. And run the following, or whatever you're supposed to."
echo "e.g.,"
echo "./darknet detector calc_anchors data/carpk.data -num_of_clusters 9 -width 1280 -height 710 -show"
