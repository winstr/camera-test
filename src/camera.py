from dataclasses import dataclass, field
from typing import Tuple, List

import cv2
import numpy as np


class FailedOpenError(RuntimeError): pass
class FailedReadError(RuntimeError): pass
class ClassInitError(RuntimeError): pass
class UnsupportedFPSError(RuntimeError): pass


@dataclass
class CaptureMode():
    capture_width: int
    capture_height: int
    capture_fps: List[int] = field(default_factory=list)

    def values(self) -> Tuple[int, int, List[int]]:
        return self.capture_width, self.capture_height, self.capture_fps


class CameraCapture():

    def __init__(self):
        self._capture = None
        self._capture_source = None
        self._grabbed = False

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def configure(self, capture_source:str):
        self._capture_source = capture_source
        self._capture = cv2.VideoCapture(self._capture_source)
        if not self._capture.isOpened():
            raise FailedOpenError(self._capture_source)

    def release(self):
        if self._capture is not None and self._capture.isOpened():
            self._capture.release()
            self._capture = None
            self._capture_source = None
            self._grabbed = False

    def read(self) -> np.ndarray:
        ret, frame = self._capture.read()
        if not ret:
            raise FailedReadError(self._capture_source)
        return frame

    def grab(self):
        self._grabbed = self._capture.grab()

    def retrieve(self) -> np.ndarray:
        if not self._grabbed:
            raise FailedReadError(self._capture_source)
        self._grabbed = False
        ret, frame = self._capture.retrieve()
        if not ret:
            raise FailedReadError(self._capture_source)
        return frame


class Camera(CameraCapture):

    CAPTURE_MODES = {}

    def __new__(cls):
        if cls is Camera:
            raise ClassInitError()
        return super().__new__(cls)

    def __init__(self):
        self._capture_mode = None

    def configure(self, capture_source:str, mode:int, fps:int):
        super().configure(capture_source)
        capture_width, capture_height, capture_fps = self.CAPTURE_MODES[mode].values()
        if not fps in capture_fps:
            raise UnsupportedFPSError()
        self._capture_mode = CaptureMode(capture_width, capture_height, [fps])
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)


class oCamS1CGNU(Camera):

    # for USB 2.0 and 3.0
    CAPTURE_MODES = {0: CaptureMode(640, 480, [10, 15, 20, 25, 30, 35, 40, 45]),
                     1: CaptureMode(640, 360, [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]),
                     2: CaptureMode(320, 240, [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])}

    def configure(self, capture_source:str, mode:int, fps:int):
        super().configure(capture_source, mode, fps)
        self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 0.)
        self._capture.set(cv2.CAP_PROP_EXPOSURE, 150)

    def preprocess(self, frame:np.ndarray, dst_size:Tuple[int, int]=None) -> np.ndarray:
        r_frame, l_frame = cv2.split(frame)
        #r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BAYER_GB2BGR)
        l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BAYER_GB2BGR)
        if dst_size is not None:
            #r_frame = cv2.resize(r_frame, dst_size)
            l_frame = cv2.resize(l_frame, dst_size)
        return l_frame, r_frame


class ThermoCam160B(Camera):

    CAPTURE_MODES = {0: CaptureMode(160, 120, [9])}

    def configure(self, capture_source:str, mode:int, fps:int):
        super().configure(capture_source, mode, fps)
        self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 0.)

    def preprocess(self, frame:np.ndarray, dst_size:Tuple[int, int]=None) -> np.ndarray:
        frame = frame / 65535.0
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = (frame * 255.0).astype(np.uint8)
        if dst_size is not None:
            frame = cv2.resize(frame, dst_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame


if __name__ == '__main__':
    # Usage

    import traceback

    def display(cam, fn):
        try:
            while True:
                frame = cam.read()
                frame = fn(frame)  # optional
                cv2.imshow('dst', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        except:
            traceback.print_exc()
        cv2.destroyAllWindows()
        cam.release()

    dst_size = (640, 480)  # width and height of output image

    stereo_cam = oCamS1CGNU()
    stereo_cam.configure('/dev/camera/oCamS-1CGN-U', mode=0, fps=45)
    stereo_fn = lambda x: cv2.flip(stereo_cam.preprocess(x, dst_size)[0], 0)  # only left frame
    display(stereo_cam, stereo_fn)

    thermo_cam = ThermoCam160B()
    thermo_cam.configure('/dev/camera/ThermoCam160B', mode=0, fps=9)
    thermo_fn = lambda x: cv2.flip(thermo_cam.preprocess(x, dst_size), 0)
    display(thermo_cam, thermo_fn)
