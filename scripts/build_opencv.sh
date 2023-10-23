#!/bin/bash

# OpenCV 4.8.0 Source Compile
# https://qengineering.eu/install-opencv-on-jetson-nano.html

wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-8-0.sh
sudo chmod 755 ./OpenCV-4-8-0.sh
./OpenCV-4-8-0.sh

rm OpenCV-4-8-0.sh
sudo rm -rf ~/opencv
sudo rm -rf ~/opencv_contrib
