import os
import time
import pickle
from tqdm import tqdm

import cv2
import numpy as np


class FailedToOpenError(RuntimeError):
    pass


class FailedToReadError(RuntimeError):
    pass


class Callibrator():

    def __init__(self, chessboard, square_size) -> None:
        self.chessboard = chessboard

        self.objp = np.zeros((1, chessboard[0]*chessboard[1], 3), np.float32)
        self.objp[0, :, :2] = \
            np.mgrid[0:chessboard[0],
                     0:chessboard[1]].T.reshape(-1, 2) * square_size
        self.objpoints = []
        self.imgpoints = []

        self.calibration_data = {
            "ret": None,
            "mtx": None,
            "dist": None,
            "rvecs": None,
            "tvecs": None}

    def calibrate(self, image_paths, image_width, image_height) -> None:
        for path in tqdm(image_paths, total=len(image_paths)):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(path)
            ret, corners = cv2.findChessboardCorners(
                image,
                self.chessboard,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (image_width, image_height), None, None)
        self.calibration_data["ret"] = ret
        self.calibration_data["mtx"] = mtx
        self.calibration_data["dist"] = dist
        self.calibration_data["rvecs"] = rvecs
        self.calibration_data["tvecs"] = tvecs

    def get_calibration_data(self) -> dict:
        return self.calibration_data


class VideoViewer():

    def __init__(self, source) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise FailedToOpenError(source)

        self.is_streaming_source = self.cap.get(
            cv2.CAP_PROP_FRAME_COUNT) <= 0

    def read(self) -> np.ndarray:
        retval, image = self.cap.read()
        if not retval:
            raise FailedToOpenError()
        return image

    def show(self) -> None:
        winname = f"{id(self.cap)}"
        if self.is_streaming_source:
            wait_ms = 1  # millisecond
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            wait_ms = int(1000.0 / fps)
        while True:
            image = self.read()
            cv2.imshow(winname, image)
            key = cv2.waitKey(wait_ms)
            if key == ord("q"):
                break


class CameraViewer(VideoViewer):

    def __init__(self, camera_source) -> None:
        super().__init__(camera_source)
        self.mtx = None
        self.dist = None

    def set_parameters(self, mtx, dist) -> None:
        self.mtx = mtx
        self.dist = dist

    def read(self, is_calibrated=False) -> np.ndarray:
        image = super().read()
        if is_calibrated:
            image = cv2.undistort(image, self.mtx, self.dist)


class oCamS_1CGN_U(VideoViewer):

    def __init__(self, device_file) -> None:
        super().__init__(device_file)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)


if __name__ == "__main__":
    stereo_cam = oCamS_1CGN_U()
    stereo_cam.show()
