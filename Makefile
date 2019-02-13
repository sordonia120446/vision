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

clean: clean-carpk
	@rm -rf opencv_data/
	@rm -rf test_video/
	@rm -rf yolo_output/

clean-carpk:
	@rm -rf CARPK_devkit/data/labels
	@rm -rf PUCPR+_devkit/data/labels
	@rm -f CARPK_devkit/data/CARPK_devkit_train.txt
	@rm -f CARPK_devkit/data/CARPK_devkit_test.txt
	@rm -f PUCPR+_devkit/data/PUCPR+_devkit_train.txt
	@rm -f PUCPR+_devkit/data/PUCPR+_devkit_test.txt

clean-drawings:
	@rm -rf CARPK_devkit/data/output_images
	@rm -rf CARPK_devkit/data/yolo_drawings
	@rm -rf PUCPR+_devkit/data/output_images
	@rm -rf PUCPR+_devkit/data/yolo_drawings

opencv:
	@docker-compose run vision python opencv.py -h

shell:
	@docker-compose run vision bash

status:
	@docker stats --no-stream

test:
	@docker-compose run vision bash ./app/opencv/get_test_video.sh
	@docker-compose run vision python opencv.py -v test_video/big_buck_bunny.mp4

.PHONY: build, build-fresh, carpk, clean, clean-carpk, clean-drawings, opencv, shell, status, test
