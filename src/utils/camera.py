import os
import pickle
from datetime import datetime

import cv2
import numpy as np


class FailedToOpenError(RuntimeError): pass
class FailedToReadError(RuntimeError): pass
class InconsistentImageShapeError(RuntimeError): pass


def save_data(file_path, data) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_data(file_path) -> None:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")


class Callibrator():

    def __init__(self, chessboard, square_size_mm) -> None:
        self.chessboard = chessboard

        self.obj_points = np.zeros(
            shape=(1, chessboard[0] * chessboard[1], 3), dtype=np.float32)
        self.obj_points[0, :, :2] = np.mgrid[
            0:chessboard[0],
            0:chessboard[1]].T.reshape(-1, 2) * square_size_mm

        self.obj_points_buffer = []
        self.img_points_buffer = []
        self.calibration_params = {}

    def calibrate(self, img_paths) -> None:
        img_shape = None
        for img_path in img_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_shape is None:
                img_shape = img.shape[::-1]
            if not img_shape == img.shape[::-1]:
                raise InconsistentImageShapeError(img_path)

            ret, corners = cv2.findChessboardCorners(
                img,
                self.chessboard,
                (cv2.CALIB_CB_ADAPTIVE_THRESH
                 + cv2.CALIB_CB_FAST_CHECK
                 + cv2.CALIB_CB_NORMALIZE_IMAGE))
            if ret:
                self.obj_points_buffer.append(self.obj_points)
                self.img_points_buffer.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points_buffer,
            self.img_points_buffer,
            img_shape,
            None,
            None)

        self.calibration_params["ret"] = ret
        self.calibration_params["mtx"] = mtx
        self.calibration_params["dist"] = dist
        self.calibration_params["rvecs"] = rvecs
        self.calibration_params["tvecs"] = tvecs

    def get_parameters(self) -> dict:
        return self.calibration_params


class VideoViewer():

    def __init__(self, source, output_dir="out") -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise FailedToOpenError(source)
        self.output_dir = output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.is_streaming_source = \
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0

    def read(self) -> np.ndarray:
        ret, img = self.cap.read()
        if not ret:
            raise FailedToOpenError()
        return img

    def show(self, tag="snapshot") -> None:
        winname = f"{id(self.cap)}"
        if self.is_streaming_source:
            delay_ms = 1
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            delay_ms = int(1000.0 / fps)
        while True:
            img = self.read()
            cv2.imshow(winname, img)
            key = cv2.waitKey(delay_ms)
            if key == ord("q"):
                break
            if key == ord("s"):
                img_name = f"{get_datetime()}_{tag}.jpg"
                img_path = os.path.join(self.output_dir, img_name)
                cv2.imwrite(img_path, img)


class oCamS_1CGN_U(VideoViewer):

    def __init__(self, source, output_dir="out", params=None) -> None:
        super().__init__(source, output_dir)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 400)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        self.params = params
        self.target_lenses = ["left", "right"]

    def set_params(self, params):
        self.params = params

    def set_target_lenses(self, target_lenses):
        self.target_lenses = target_lenses
    
    def read(self) -> np.ndarray:
        img = super().read()
        img_R, img_L = cv2.split(img)

        if not params is None:
            img_L = cv2.undistort(img_L, self.params["mtx"][0], self.params["dist"][0])
            img_R = cv2.undistort(img_R, self.params["mtx"][1], self.params["dist"][1])

        target_imgs = []
        if "left" in self.target_lenses:
            target_imgs.append(img_L)
        if "right" in self.target_lenses:
            target_imgs.append(img_R)
        return cv2.hconcat(target_imgs)


if __name__ == "__main__":
    source = "/dev/video0"
    output_dir = "out"
    target_lenses = ["left"]
    img_tag = "left"

    left_params = load_data("ocam_left_calib.pkl")
    right_params = load_data("ocam_right_calib.pkl")

    params = {
        "mtx": (left_params["mtx"], right_params["mtx"]),
        "dist": (left_params["dist"], right_params["dist"])}

    stereo_cam = oCamS_1CGN_U(source, output_dir)
    stereo_cam.set_params(params)
    stereo_cam.set_target_lenses(target_lenses)
    stereo_cam.show(img_tag)
