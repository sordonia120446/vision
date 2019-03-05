# Summary

Computer Vision Sandbox Container.

![alt text](docker_whale.jpg "Logo Title Text 1")

## Motivation

This is a toolbox of different methods to tackle Computer Vision.  The biggest one of note is **Open CV**.

Finally, I've set up Pytorch, a great deep learning framework.  You can use it do computer vision.  Or just use Open CV's great image processing functionality.

## Container Solution

Open CV can wreak havoc with your local environment.  Build the things in a Docker that provides a safe haven befitting of a sandbox.  And who doesn't like whales?

# Usage

## Setup and Open CV

Run `make` to build the container and test out the Open CV build.  You should see a folder containing still frames extracted from a video.

There's some useful image transformation functions.  Run `make opencv` to see details.

To ensure the Docker built successfully, run the following command that will download a test video and break it up into frame-by-frame image files and save them to a folder, opencv_data. You'll also want to change the group and user ownership of the folder to whatever you are logged in as currently.

```
make test
sudo chown -R <user_id>:<group_id> opencv_data/
```

To see what user and group you are logged in as currently, run `id` in terminal.

## Datasets

So far, there's only support to convert the CARPK and PUCPR datasets of annotated images into YOLO format. There may be more to come.

### CARPK and PKLOT

First, be sure to have the data folders placed into the top-level working directory. Then, you should just be able to run the make targets to convert them into YOLO. Alternatively, the bounding boxes can be drawn around the cars for visualiziaton sanity check.

To convert the annotations into YOLO format, run the below command. Refer to the "Setup and Open CV" section for more details on how to modify file and folder ownership and permissions.

```
make carpk
make pklot
```

For more details on the CARPK script, run the below command to find out more. The PKLOT one is pretty much the same.

```
docker-compose run vision python app/carpk/draw_bounding_boxes.py -h
```

You will want to change the ownership on the data folder created by the Docker. To do so, run

```
sudo chown <user_id>:<group_id> <image-path>
```

# Dependencies

    Docker
    docker-compose
