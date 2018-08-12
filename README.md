# Summary

Computer Vision Sandbox Container.

![alt text](docker_whale.jpg "Logo Title Text 1")

## Motivation

This is a toolbox of different methods to tackle Computer Vision.  The biggest one of note is **Open CV**.

Finally, I've set up Pytorch, a great deep learning framework.  You can use it do computer vision.  Or just use Open CV's great image processing functionality.

## Container Solution

Open CV can wreak havoc with your local environment.  Build the things in a Docker that provides a safe haven befitting of a sandbox.  And who doesn't like whales?

# Usage

## Instructions

Run `make` to build the container and test out the Open CV build.  You should see a folder containing still frames extracted from a video.

There's some useful image transformation functions.  Run `make opencv` to see details.

## Dependencies

    Docker
    docker-compose
