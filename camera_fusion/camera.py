from typing import Tuple
from dataclasses import dataclass

import cv2
import numpy as np


class FailedOpenError(RuntimeError): pass
class FailedReadError(RuntimeError): pass
class UnsupportedMode(RuntimeError): pass
class InvalidFPSValue(RuntimeError): pass


@dataclass
class CaptureMode():

    cap_width: int
    cap_height: int
    cap_fps: int

    def values(self) -> Tuple[int, int, int]:
        return (self.cap_width, self.cap_height, self.cap_fps)


class CameraCapture():

    def __init__(self) -> None:
        self.cap = None           # A instance of cv2.VideoCapture(...)
        self.cap_source = None    # Capture source  e.g. /dev/video0

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def connect(self, cap_source:str) -> None:
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise FailedOpenError(cap_source)
        self.cap = cap
        self.cap_source = cap_source

    def disconnect(self) -> None:
        if self.is_connected():
            self.cap.release()
            self.cap = None
            self.cap_source = None

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise FailedReadError(self.cap_source)
        return frame
