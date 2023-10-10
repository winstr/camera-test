import os
import glob
import pickle

import cv2
import numpy as np

from src.camera import oCamS_1CGN_U_ChessboardCapture


def save_data(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist)
    return dst


def calibrate_mono(chessboard, square_size, img_paths, img_width, img_height):
    objp = np.zeros((1, chessboard[0]*chessboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[
        0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError()
        ret, corners = cv2.findChessboardCorners(
            img,
            chessboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img_width, img_height), None, None)
    calibration_data = {
        "ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
    return calibration_data


def calibrate_stereo(chessboard, square_size, img_paths, img_width, img_height):
    objp = np.zeros((chessboard[0]*chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    left_img_paths, right_img_paths = img_paths

    for left_img_path, right_img_path in zip(left_img_paths, right_img_paths):
        left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        if left_img is None:
            raise ValueError()
        right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
        if right_img is None:
            raise ValueError()

        left_ret, left_corners = cv2.findChessboardCorners(left_img, chessboard)
        right_ret, right_corners = cv2.findChessboardCorners(right_img, chessboard)

        if left_ret and right_ret:
            objpoints.append(objp)
            imgpoints_left.append(left_corners)
            imgpoints_right.append(right_corners)

    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, None, None, None, None, (img_width, img_height))
    calibration_data = {
        'ret': ret, 'mtx_left': mtx_left, 'dist_left': dist_left,
        'mtx_right': mtx_right, 'dist_right': dist_right,
        'R': R, 'T': T, 'E': E, 'F': F}
    return calibration_data


if __name__ == '__main__':
    camera_source = '/dev/camera/oCamS-1CGN-U'
    img_width, img_height = 640, 480
    exposure = 400
    num_corners_col = 10
    num_corners_row = 7
    chessboard = (num_corners_col, num_corners_row)
    square_size = 23  # mm

    l_output_dir = 'out_left'
    r_output_dir = 'out_right'
    stereo_output_dir = 'out_stereo'

    l_calib_file = 'l_calib_mono.pkl'
    r_calib_file = 'r_calib_mono.pkl'
    stereo_calib_file = 'lr_calib_stereo.pkl'

    """
    # --- step 1. chessboard capture (mono) ---
    with oCamS_1CGN_U_ChessboardCapture() as ocams:
        ocams.configure(camera_source, exposure=exposure)
        ocams.set_chessboard(num_corners_col, num_corners_row)
        # left lense
        ocams.set_lenses(left=True, right=False)
        ocams.display('left', l_output_dir)
        # right lense
        ocams.set_lenses(left=False, right=True)
        ocams.display('right', r_output_dir)

    # --- step 2. calibration (mono) ---
    # left lense
    l_img_paths = glob.glob(f'{l_output_dir}/left_*.jpg')
    l_calib_data = calibrate_mono(chessboard, square_size, l_img_paths, img_width, img_height)
    save_data(l_calib_file, l_calib_data)
    print(l_calib_data['ret'])
    # right lense
    r_img_paths = glob.glob(f'{r_output_dir}/right_*.jpg')
    r_calib_data = calibrate_mono(chessboard, square_size, r_img_paths, img_width, img_height)
    save_data(r_calib_file, r_calib_data)
    print(r_calib_data['ret'])
    """

    # --- step 3. chessboard capture (stereo) ---
    with oCamS_1CGN_U_ChessboardCapture() as ocams:
        ocams.configure(camera_source, exposure=exposure)
        ocams.set_chessboard(num_corners_col, num_corners_row)
        # all lenses
        ocams.set_lenses(left=True, right=True)
        ocams.display('stereo', stereo_output_dir)

    # --- step 4. calibration (stereo) ---
    l_stereo_img_paths = glob.glob(f'{stereo_output_dir}/left_*.jpg')
    r_stereo_img_paths = glob.glob(f'{stereo_output_dir}/right_*.jpg')
    stereo_img_paths = [l_stereo_img_paths, r_stereo_img_paths]
    stereo_calib_data = calibrate_stereo(chessboard, square_size, stereo_img_paths, img_width, img_height)
    save_data(stereo_calib_file, stereo_calib_data)
    print(stereo_calib_data['ret'])