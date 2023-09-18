import os
from pathlib import Path
import glob
import pickle

import cv2
import numpy as np


def get_internal_params(chessboard, img_paths, img_width, img_height):
    objp = np.zeros((1, chessboard[0]*chessboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ret, corners = cv2.findChessboardCorners(
            img,
            chessboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img_width, img_height), None, None)
    
    print(ret)
    
    return ret, mtx, dist, rvecs, tvecs


def save_calibration_data(ret, mtx, dist, rvecs, tvecs, filename):
    calibration_data = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,}

    with open(filename, 'wb') as f:
        pickle.dump(calibration_data, f)


if __name__ == "__main__":
    img_dir = "out"
    img_width = 640
    img_height = 480
    num_corners_hor = 10
    num_corners_ver = 7

    output_dir = str(Path(__file__).parent.absolute() / img_dir)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(output_dir)

    chessboard = (num_corners_hor, num_corners_ver)

    left_img_paths = glob.glob(f"{output_dir}/*L.jpg")
    ret, mtx, dist, rvecs, tvecs = get_internal_params(chessboard, left_img_paths, img_width, img_height)
    save_calibration_data(ret, mtx, dist, rvecs, tvecs, "left_cam_calib.pkl")

    right_img_paths = glob.glob(f"{output_dir}/*R.jpg")
    ret, mtx, dist, rvecs, tvecs = get_internal_params(chessboard, right_img_paths, img_width, img_height)
    save_calibration_data(ret, mtx, dist, rvecs, tvecs, "right_cam_calib.pkl")
