import os
import glob
import pickle
from datetime import datetime

import cv2
import numpy as np

from camera import oCamS_1CGN_U


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")


def save_data(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def mono_capture_chessboard(camera, chessboard, prefix, output_dir='out'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    i = 0
    while True:
        frame = camera.capture()
        frame_ = frame.copy()

        ret, corners = cv2.findChessboardCorners(frame_, chessboard)
        if ret:
            cv2.drawChessboardCorners(frame_, chessboard, corners, ret)

        cv2.imshow('preview', frame_)

        key = cv2.waitKey(10)
        if key == ord('s'):
            img_path = f'{prefix}_{str(i).zfill(3)}_{get_datetime()}.jpg'
            img_path = os.path.join(output_dir, img_path)
            cv2.imwrite(img_path, frame)
            i += 1
        elif key == ord('q'):
            break


def mono_calibrate(chessboard, square_size, img_paths, img_width, img_height):
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

# ----- -----

def stereo_capture_chessboard(camera, chessboard, output_dir='out'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    i = 0
    while True:
        frame = camera.capture()

        frame_left = frame[:, :640, :]
        frame_right = frame[:, 640:, :]

        frame_left_ = frame_left.copy()
        frame_right_ = frame_right.copy()

        ret_left, corners_left = cv2.findChessboardCorners(frame_left_, chessboard)
        ret_right, corners_right = cv2.findChessboardCorners(frame_right_, chessboard)

        if ret_left and ret_right:
            cv2.drawChessboardCorners(frame_left_, chessboard, corners_left, ret_left)
            cv2.drawChessboardCorners(frame_right_, chessboard, corners_right, ret_right)

        cv2.imshow('preview', cv2.hconcat([frame_left_, frame_right_]))

        key = cv2.waitKey(10)
        if key == ord('s'):
            tag = f'{str(i).zfill(3)}_{get_datetime()}.jpg'

            left_img_path = f'left_{tag}'
            left_img_path = os.path.join(output_dir, left_img_path)
            cv2.imwrite(left_img_path, frame_left)

            right_img_path = f'right_{tag}'
            right_img_path = os.path.join(output_dir, right_img_path)
            cv2.imwrite(right_img_path, frame_right)

            i += 1
        elif key == ord('q'):
            break


def stereo_calibrate(chessboard, square_size, img_paths, img_width, img_height):
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
    print(ret)

    calib_data = {
        'ret': ret, 'mtx_left': mtx_left, 'dist_left': dist_left,
        'mtx_right': mtx_right, 'dist_right': dist_right,
        'R': R, 'T': T, 'E': E, 'F': F}

    save_data('data/oCamS-1CGN-U_stereo_calib.pkl', calib_data)

# ----- -----

if __name__ == '__main__':

    chessboard = (10, 7)  # 11 cols 8 rows
    square_size = 23  # mm
    #camera_source = '/dev/camera/oCamS-1CGN-U'
    #lenses = ['left', 'right']

    #ocams = oCamS_1CGN_U()
    #ocams.connect(camera_source, exposure=400)
    #ocams.set_lenses(lenses)

    #prefix = '+'.join(ocams.lenses)
    output_dir = 'out_2023-10-10_00:48:23.972719'

    #stereo_capture_chessboard(ocams, chessboard, output_dir)
    #ocams.release()

    left_img_paths = glob.glob(f'{output_dir}/left*.jpg')
    left_img_paths.sort()

    right_img_paths = glob.glob(f'{output_dir}/right*.jpg')
    right_img_paths.sort()

    img_paths = [left_img_paths, right_img_paths]
    stereo_calibrate(chessboard, square_size, img_paths, 640, 480)


    '''
    mono_capture_chessboard(ocams, chessboard, prefix, output_dir)
    ocams.release()

    img_paths = glob.glob(f'{output_dir}/{prefix}*.jpg')
    calib_data = mono_calibrate(
        chessboard, square_size, img_paths,
        ocams.FRAME_WIDTH, ocams.FRAME_HEIGHT)
    save_data(
        f'data/{os.path.basename(camera_source)}_{prefix}_calib.pkl',
        calib_data)
    print(calib_data['ret'])
    '''