#!/bin/bash

# Check the Docker image Version
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml

NAME="r32.7.1-py3" # for Jetson Nano (Jetpack 4.6.1)
DIST="nvcr.io/nvidia/l4t-ml:$NAME"

xhost +local:docker
docker run -it --runtime nvidia \
    --ipc=host --net=host --privileged --cap-add SYS_PTRACE \
    -v $HOME/Docker/Containers/$NAME:/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /dev/cam:/dev/cam \
    -e DISPLAY=$DISPLAY --name $NAME $DIST \
    /bin/bash
