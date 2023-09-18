import os
from pathlib import Path

import cv2
import numpy as np


device_path = "/dev/video0"
img_dir = "out"
img_width = 640
img_height = 480
exposure = 500
num_corners_hor = 10
num_corners_ver = 7

# -----

output_dir = str(Path(__file__).parent.absolute() / img_dir)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(device_path)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

chessboard = (num_corners_hor, num_corners_ver)
objp = np.zeros((1, chessboard[0]*chessboard[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

img_idx = 0
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img_right, img_left = cv2.split(img)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BAYER_BG2GRAY)
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BAYER_BG2GRAY)

    img_left_ = img_left.copy()
    img_right_ = img_left.copy()

    ret_left, corners_left = cv2.findChessboardCorners(img_left_, chessboard)
    ret_right, corners_right = cv2.findChessboardCorners(img_right_, chessboard)

    if ret_left and ret_right:
        cv2.drawChessboardCorners(img_left_, chessboard, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right_, chessboard, corners_right, ret_right)

    preview = cv2.hconcat([img_left_, img_right_])
    cv2.imshow("preview", preview)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s"):
        cv2.imwrite(os.path.join(output_dir, f"{str(img_idx).zfill(5)}L.jpg"), img_left)
        cv2.imwrite(os.path.join(output_dir, f"{str(img_idx).zfill(5)}R.jpg"), img_right)
        img_idx += 1

cv2.destroyAllWindows()
cap.release()
