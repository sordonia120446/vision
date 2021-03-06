echo -e "I thank you for your kind service, Hsieh\n"

# Draw PUCPR ground truth bounding boxes
mkdir -p PUCPR_devkit/data/output_images
python app/carpk/draw_bounding_boxes.py --draw --annots_dir PUCPR_devkit/data/Annotations/ --images_dir PUCPR_devkit/data/Images/

# Draw CARPK ground truth bounding boxes
mkdir -p CARPK_devkit/data/output_images
python app/carpk/draw_bounding_boxes.py --draw --annots_dir CARPK_devkit/data/Annotations/ --images_dir CARPK_devkit/data/Images/
