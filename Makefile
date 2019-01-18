all: build test opencv

build:
	@docker-compose build

build-fresh:
	@docker-compose build --pull --no-cache

carpk:
	@docker-compose run vision bash app/carpk/draw_bounding_boxes.sh

clean:
	@rm -rf opencv_data/
	@rm -rf test_video/
	@rm -rf yolo_output/

clean-docker:
	@docker system prune -af

opencv:
	@docker-compose run vision python opencv.py -h

purge: clean clean-docker

shell:
	@docker-compose run vision bash

status:
	@docker stats --no-stream

test:
	@docker-compose run vision bash ./app/opencv/get_test_video.sh
	@docker-compose run vision python opencv.py -v test_video/big_buck_bunny.mp4

.PHONY: build, build-fresh, carpk, clean, clean-docker, opencv, purge, shell, status, test
