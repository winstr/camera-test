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


def calibrate(chessboard, square_size, img_paths, img_width, img_height):
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


def capture_chessboard(camera, chessboard, prefix, output_dir='out'):
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


if __name__ == '__main__':

    chessboard = (10, 7)  # 11 cols 8 rows
    square_size = 23  # mm
    camera_source = '/dev/camera/oCamS-1CGN-U'
    lenses = ['right']

    ocams = oCamS_1CGN_U()
    ocams.connect(camera_source, exposure=400)
    ocams.set_lenses(lenses)

    prefix = '+'.join(ocams.lenses)
    output_dir = f'out_{get_datetime()}'

    capture_chessboard(ocams, chessboard, prefix, output_dir)
    ocams.release()

    img_paths = glob.glob(f'{output_dir}/{prefix}*.jpg')
    calib_data = calibrate(
        chessboard, square_size, img_paths,
        ocams.FRAME_WIDTH, ocams.FRAME_HEIGHT)
    save_data(
        f'{os.path.basename(camera_source)}_{prefix}_calib.pkl',
        calib_data)
    print(calib_data['ret'])

    

    '''
    def calib_ocams(lenses):
        chessboard = (10, 7)  # 11 cols 8 rows
        square_size = 23  # mm

        camera_source = '/dev/camera/oCamS-1CGN-U'
        exposure = 400

        ocams = oCamS_1CGN_U()
        ocams.connect(camera_source, exposure)
        ocams.set_lenses(lenses)

        prefix = '+'.join(lenses)
        output_dir = f'out_{get_datetime()}'

        capture_chessboard(ocams, chessboard, prefix, output_dir)
        ocams.release()

        captured_img_paths = glob.glob(f'{output_dir}/{prefix}*.jpg')
        calib_data = calibrate(chessboard, square_size, captured_img_paths, ocams.FRAME_WIDTH, ocams.FRAME_HEIGHT)
        save_data(f'{os.path.basename(camera_source)}_{prefix}_calib.pkl', calib_data)
        print(calib_data['ret'])
    '''

'''
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


class Calibrator():

    def __init__(self, chessboard, square_size):
        self.chessboard = chessboard
        self.square_size = square_size
        self.calibration_data = {}

    def calibrate_with_images(self, img_dir, img_name_pattern):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(img_dir)
        
        img_paths = glob.glob(f"{img_dir}/{img_name_pattern}")
        img_height, img_width = cv2.imread(
            img_paths[0], cv2.IMREAD_GRAYSCALE).shape

        return calibrate_mono(
            self.chessboard, self.square_size, img_paths, img_width, img_height)


def __calibrate_oCamS_1CGN_U(chessboard, square_size):
    img_dir = "out"
    img_name_pattern_left = "*left.jpg"
    img_name_pattern_right = "*right.jpg"

    calibrator = Calibrator(chessboard, square_size)
    calib_left = calibrator.calibrate_with_images(
        img_dir, img_name_pattern_left)
    calib_right = calibrator.calibrate_with_images(
        img_dir, img_name_pattern_right)
    
    print(f"left: {calib_left['ret']}")
    print(f"right: {calib_right['ret']}")

    save_data("ocam_left_calib.pkl", calib_left)
    save_data("ocam_right_calib.pkl", calib_right)


#__calibrate_oCamS_1CGN_U((8, 6), 22.0)
#__validate_oCamS_1CGN_U()
'''