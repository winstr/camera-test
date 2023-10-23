#!/bin/bash
# check dev: ex) udevadm info -a -n /dev/video0

#CSICAM0_NAME="CSI0"
#CSICAM1_NAME="CSI1"

USBCAM0_NAME="ThermoCam160B"
USBCAM0_VID="1209"  # vendor id
USBCAM0_PID="0160"  # product id

USBCAM1_NAME="oCamS-1CGN-U"
USBCAM1_VID="04b4"
USBCAM1_PID="00f9"

SYMLINK_DIR="cam"

#echo SUBSYSTEM==\"video4linux\", KERNEL==\"video0\", ATTR{name}==\"vi-output, imx219 7-0010\", SYMLINK+=\"$SYMLINK_DIR/$CSICAM0_NAME\" | sudo tee /etc/udev/rules.d/99-$CSICAM0_NAME.rules > /dev/null
#echo SUBSYSTEM==\"video4linux\", KERNEL==\"video1\", ATTR{name}==\"vi-output, imx219 8-0010\", SYMLINK+=\"$SYMLINK_DIR/$CSICAM1_NAME\" | sudo tee /etc/udev/rules.d/99-$CSICAM1_NAME.rules > /dev/null
echo SUBSYSTEM==\"video4linux\", ATTRS{idVendor}==\"$USBCAM0_VID\", ATTRS{idProduct}==\"$USBCAM0_PID\", SYMLINK+=\"$SYMLINK_DIR/$USBCAM0_NAME\" | sudo tee /etc/udev/rules.d/99-$USBCAM0_NAME.rules > /dev/null
echo SUBSYSTEM==\"video4linux\", ATTRS{idVendor}==\"$USBCAM1_VID\", ATTRS{idProduct}==\"$USBCAM1_PID\", SYMLINK+=\"$SYMLINK_DIR/$USBCAM1_NAME\" | sudo tee /etc/udev/rules.d/99-$USBCAM1_NAME.rules > /dev/null

sudo udevadm control --reload-rules && sudo udevadm trigger
