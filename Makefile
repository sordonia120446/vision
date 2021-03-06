all: build test opencv

build:
	@docker-compose build

build-fresh:
	@docker-compose build --pull --no-cache

carpk:
	@docker-compose run vision bash app/carpk/draw_bounding_boxes.sh

carpk-hsieh:
	@echo "Using the OG Hsieh annotation format to draw on the images."
	@docker-compose run vision bash app/carpk/hsieh_bounding_boxes.sh

cnr:
	@docker-compose run vision bash app/cnr/create_devkit.sh

clean: clean-carpk
	@rm -rf opencv_data/
	@rm -rf test_video/
	@rm -rf yolo_output/

clean-carpk:
	@rm -rf CARPK_devkit/data/labels
	@rm -rf PUCPR_devkit/data/labels
	@rm -f CARPK_devkit/data/CARPK_devkit_train.txt
	@rm -f CARPK_devkit/data/CARPK_devkit_test.txt
	@rm -f PUCPR_devkit/data/PUCPR_devkit_train.txt
	@rm -f PUCPR_devkit/data/PUCPR_devkit_test.txt

clean-cnr:
	@rm -rf CNR_devkit/

clean-drawings:
	@rm -rf CARPK_devkit/data/output_images
	@rm -rf CARPK_devkit/data/yolo_drawings
	@rm -rf PUCPR_devkit/data/output_images
	@rm -rf PUCPR_devkit/data/yolo_drawings

clean-pklot:
	@rm -rf PKLOT_devkit

opencv:
	@docker-compose run vision python opencv.py -h

pklot:
	@mkdir -p PKLOT_devkit
	@docker-compose run vision bash app/pklot/create_target_dir.sh

shell:
	@docker-compose run vision bash

status:
	@docker stats --no-stream

test:
	@docker-compose run vision bash ./app/opencv/get_test_video.sh
	@docker-compose run vision python opencv.py -v test_video/big_buck_bunny.mp4

.PHONY: build, build-fresh, carpk, cnr, clean, clean-carpk, clean-cnr, clean-drawings, clean-pklot, opencv, pklot, shell, status, test
