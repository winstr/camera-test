import os
import pickle
from pathlib import Path

import cv2
import numpy as np

"""
DEVICE = "/dev/video2"
SHUTTER_SPEED = 500  # (millisecond, ms)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

SQUARE_SIZE = 23.0  # (millimeter, mm)
NUM_CORNERS_HORIZONTAL = 10
NUM_CORNERS_VERTICAL = 7

cap = cv2.VideoCapture(DEVICE)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

obj_points = []
im_points_left = []
im_points_right = []
num_corners = (NUM_CORNERS_HORIZONTAL, NUM_CORNERS_VERTICAL)

objp = np.zeros((NUM_CORNERS_HORIZONTAL * NUM_CORNERS_VERTICAL, 3), np.float32)
objp[:, :2] = np.mgrid[0:NUM_CORNERS_HORIZONTAL, 0:NUM_CORNERS_VERTICAL].T.reshape(-1, 2) * SQUARE_SIZE

while cap.isOpened():
    ret_pair, buffer = cap.read()
    if not ret_pair:
        break

    buffer = buffer.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 2)
    left_image, right_image = cv2.split(buffer)

    left_image = cv2.cvtColor(left_image, cv2.COLOR_BAYER_GB2BGR)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BAYER_GB2BGR)

    ret_left, corners_left = cv2.findChessboardCorners(left_image, num_corners)
    ret_right, corners_right = cv2.findChessboardCorners(right_image, num_corners)

    if ret_left and ret_right:
        cv2.drawChessboardCorners(left_image, num_corners, corners_left, ret_left)
        cv2.drawChessboardCorners(right_image, num_corners, corners_right, ret_right)

        obj_points.append(objp)
        im_points_left.append(corners_left)
        im_points_right.append(corners_right)
    
    concatenated_image = cv2.hconcat([left_image, right_image])
    cv2.imshow('Detected Corners - Left & Right', concatenated_image)

    if cv2.waitKey(SHUTTER_SPEED) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, im_points_left, im_points_right,
    None, None, None, None, (640, 480), flags=cv2.CALIB_FIX_INTRINSIC)

calibration_data = {
    'ret': ret,
    'M1': M1,
    'd1': d1,
    'M2': M2,
    'd2': d2,
    'R': R,
    'T': T,
    'E': E,
    'F': F}

with open('calibration_data.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)
"""

def write_stereo_images(device, width, height, output_dir, shutter_interval=1000):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    im_idx = 0

    while cap.isOpened():
        rt, buffer = cap.read()
        if not rt:
            break
        buffer = buffer.reshape(height, width, 2)
        left_im, right_im = cv2.split(buffer)
        left_im = cv2.cvtColor(left_im, cv2.COLOR_BAYER_GB2BGR)
        right_im = cv2.cvtColor(right_im, cv2.COLOR_BAYER_GB2BGR)

        prefix = os.path.join(output_dir, str(im_idx).zfill(4))
        path_left_im = prefix + "L.jpg"
        path_right_im = prefix + "R.jpg"

        cv2.imshow("LEFT SIDE", left_im)
        cv2.imshow("RIGHT SIDE", right_im)

        cv2.imwrite(path_left_im, left_im)
        cv2.imwrite(path_right_im, right_im)

        im_idx += 1

        if cv2.waitKey(shutter_interval) == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


def calib_stereo_camera(num_corners_hor, num_corners_ver, square_size, image_dir):
    obj_points = []
    im_points_left = []
    im_points_right = []
    num_corners = (num_corners_hor, num_corners_ver)

    objp = np.zeros((num_corners_hor * num_corners_ver, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_hor, 0:num_corners_ver].T.reshape(-1, 2) * square_size

    num_pairs = int(len(os.listdir(image_dir)) / 2)
    for im_idx in range(num_pairs):
        prefix = os.path.join(image_dir, str(im_idx).zfill(4))
        left_im = cv2.imread(prefix + "L.jpg")
        right_im = cv2.imread(prefix + "R.jpg")

        ret_left, corners_left = cv2.findChessboardCorners(left_im, num_corners)
        ret_right, corners_right = cv2.findChessboardCorners(right_im, num_corners)

        if ret_left and ret_right:
            cv2.drawChessboardCorners(left_im, num_corners, corners_left, ret_left)
            cv2.drawChessboardCorners(right_im, num_corners, corners_right, ret_right)
            concatenated_image = cv2.hconcat([left_im, right_im])
            cv2.imshow('Detected Corners - Left & Right', concatenated_image)
            cv2.waitKey(0)

            obj_points.append(objp)
            im_points_left.append(corners_left)
            im_points_right.append(corners_right)

        print(f"{im_idx} / {num_pairs - 1}")
    
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, im_points_left, im_points_right, None, None, None, None,
        (640, 480), flags=cv2.CALIB_FIX_INTRINSIC)
    
    calibration_data = {
        'ret': ret,
        'M1': M1,
        'd1': d1,
        'M2': M2,
        'd2': d2,
        'R': R,
        'T': T,
        'E': E,
        'F': F}

    with open('calibration_data.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)


if __name__ == '__main__':
    image_dir = str(Path(__file__).parents[0].absolute() / "out")
    num_corners_hor = 10
    num_corners_ver = 7
    square_size = 23.0

    if not os.path.isdir(image_dir):
        write_stereo_images("/dev/video2", 640, 480, image_dir, 500)

    calib_stereo_camera(num_corners_hor, num_corners_ver, square_size, image_dir)