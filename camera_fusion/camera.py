import logging
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
        self._grab = False
        logging.info(f"😊 CameraCapture Instance has been created! "
                     f"(class: {self.__class__.__name__}, "
                     f"id: {id(self)})")

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def connect(self, cap_source:str) -> None:
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            logging.info(f"😡 Failed to Open {self.cap_source}. "
                         f"Retry or check the camera. "
                         f"(class: {self.__class__.__name__}, "
                         f"id: {id(self)})")
            raise FailedOpenError(cap_source)
        self.cap = cap
        self.cap_source = cap_source
        logging.info(f"😊 CameraCapture Instance has been connected to "
                     f"{self.cap_source} successfully! "
                     f"(class: {self.__class__.__name__}, "
                     f"id: {id(self)})")

    def disconnect(self) -> None:
        if self.is_connected():
            self.cap.release()
            self.cap = None
            self.cap_source = None
            logging.info(f"😊 CameraCapture Instance has been closed. "
                         f"(class: {self.__class__.__name__}, "
                         f"id: {id(self)})")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            logging.info(f"😡 Failed to Read {self.cap_source}. "
                         f"Retry or check the camera. "
                         f"(class: {self.__class__.__name__}, "
                         f"id: {id(self)})")
            raise FailedReadError(self.cap_source)
        return frame

    def grab(self) -> None:
        self._grab = self.cap.grab()

    def retrieve(self) -> np.ndarray:
        if not self._grab:
            raise FailedReadError(self.cap_source)
        self._grab = False
        _, frame = self.cap.retrieve()
        return frame

    def preprocess(self, frame:np.ndarray) -> np.ndarray:
        # TODO
        return frame