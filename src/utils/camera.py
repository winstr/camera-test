import os
import time
import pickle

import cv2
import numpy as np


class FailedToOpenError(RuntimeError): pass
class FailedToReadError(RuntimeError): pass


def save_data(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_data(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


class CameraViewer():

    def __new__(cls, *args, **kwargs):
        raise TypeError("This class cannot be instantiated.")

    def __init__(self, device_file) -> None:
        if not os.path.isfile(device_file):
            raise FileNotFoundError(device_file)
        self.cap = cv2.VideoCapture(device_file)
        if not self.cap.isOpened():
            raise FailedToOpenError(device_file)
        self.is_calibrated = False

    def __del__(self) -> None:
        self.cap.release()

    def set_configs(self, width, height, exposure, cvt_rgb=False) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, cvt_rgb)

    def _undistort(self, image) -> np.ndarray:
        # When overriding this function in a subclass, ensure to set
        # `self.is_calibrated` to `True` if calibration is done.
        raise NotImplementedError()

    def _flatten(self, image) -> np.ndarray:
        raise NotImplementedError()

    def grab_image(self) -> np.ndarray:
        retval, image = self.cap.read()
        if not retval:
            raise FailedToReadError()
        if self.is_calibrated:
            image = self._undistort(image)
        return image

    def view_image(self) -> None:
        winname = f"[Preview] cap_id:{id(self.cap)}"
        image = self.grab_image()
        image = self._flatten(image)
        cv2.imshow(winname, image)
        cv2.waitKey(0)
        cv2.DestroyWindow(winname)

    def view_video(self) -> None:
        winname = f"[Preview] cap_id:{id(self.cap)}"
        prev_time = 0
        curr_time = 0
        while True:
            image = self.grab_image()
            image = self._flatten(image)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            cv2.putText(
                image=image,
                text=f"{str(fps):.2f}",
                org=(50, 100),
                font=cv2.FONT_HERSHEY_PLAIN,
                fontscale=1.0,
                color=(0, 255, 0),
                lineType=cv2.Line_AA)
            cv2.imshow(winname, image)
            if cv2.waitKey(1) == ord("q"):
                break
            prev_time = curr_time
        cv2.destroyWindow(winname)

    def save_image(self) -> None:
        raise NotImplementedError()
    
    def save_video(self) -> None:
        raise NotImplementedError()

    """
    def save_video(self, fps, fourcc, filename) -> None:
        if os.path.isfile(filename):
            raise FileExistsError(filename)
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        winname = f"[Preview] cap_id:{id(self.cap)}, fps:{fps}"
        while True:
            image = self.grab_image()
            cv2.imshow(winname, image)
            if cv2.waitKey(1) == ord("q"):
                break
            writer.write(image)
        cv2.destroyWindow(winname)
        writer.release()
    """


class oCamS_1CGN_U(CameraViewer):

    def __init__(self, device_file="/dev/cam/oCamS-1CGN-U") -> None:
        super().__init__(device_file)
        self.set_configs(
            width=640,
            height=480,
            exposure=200,
            cvt_rgb=False)
    
    def set_calibration(self, mtx_L, dist_L, mtx_R, dist_R):
        super().set_calibration(mtx=(mtx_L, mtx_R), dist=(dist_L, dist_R))

    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            raise CameraOpenError()
        if self.is_calibrated:
            frame[:,:,0] = cv2.undistort(frame[:,:,0], self.mtx[1], self.dist[1])  # right
            frame[:,:,1] = cv2.undistort(frame[:,:,1], self.mtx[0], self.dist[0])  # left
        return frame