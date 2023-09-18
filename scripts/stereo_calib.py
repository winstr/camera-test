from pathlib import Path
import os
import pickle

import cv2
import numpy as np


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
            cv2.waitKey(500)

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

    calib_stereo_camera(num_corners_hor, num_corners_ver, square_size, image_dir)