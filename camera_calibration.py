import os
import glob
import pickle

import cv2
import numpy as np


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


def __validate_oCamS_1CGN_U():
    from src.utils.camera import oCamS_1CGN_U
    cam = oCamS_1CGN_U(frame_exposure=400)

    left_calib_data = load_data("ocam_left_calib.pkl")
    left_mtx = left_calib_data["mtx"]
    left_dist = left_calib_data["dist"]

    right_calib_data = load_data("ocam_right_calib.pkl")
    right_mtx = right_calib_data["mtx"]
    right_dist = right_calib_data["dist"]

    while True:
        left_frame, right_frame = cam.get_frame(lens="all")
        corrected_left_frame = undistort_image(
            left_frame, left_mtx, left_dist)
        corrected_right_frame = undistort_image(
            right_frame, right_mtx, right_dist)
        
        concat = cv2.hconcat([
            corrected_left_frame,
            corrected_right_frame,
        ])
        cv2.imshow("corrected", concat)

        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    __calibrate_oCamS_1CGN_U((8, 6), 22.0)
    __validate_oCamS_1CGN_U()