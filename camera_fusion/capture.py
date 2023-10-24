import logging

import cv2
import numpy as np


class FailedOpenError(RuntimeError): pass
class FailedReadError(RuntimeError): pass


class CameraCapture():

    def __init__(self) -> None:
        self.cap = None           # A instance of cv2.VideoCapture(...)
        self.cap_source = None    # Capture source  e.g. /dev/video0
        self._grab = False
        logging.info(f"😊 CameraCapture has been created! : "
                     f"{self.__class__.__name__, id(self)}")

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def connect(self, cap_source:str, *args, **kwargs) -> None:
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            logging.info(f"😡 Failed to open {self.cap_source}! : "
                         f"{self.__class__.__name__, id(self)}")
            raise FailedOpenError(cap_source)
        self.cap = cap
        self.cap_source = cap_source
        logging.info(f"😊 CameraCapture has been connected to "
                     f"{self.cap_source} successfully! : "
                     f"{self.__class__.__name__, id(self)}")

    def disconnect(self) -> None:
        if self.is_connected():
            self.cap.release()
            self.cap = None
            self.cap_source = None
            logging.info(f"😊 CameraCapture has been closed! : "
                         f"{self.__class__.__name__, id(self)}")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            logging.info(f"😡 Failed to read {self.cap_source}! : "
                         f"{self.__class__.__name__, id(self)}")
            raise FailedReadError(self.cap_source)
        return frame

    def grab(self) -> None:
        self._grab = self.cap.grab()

    def retrieve(self) -> np.ndarray:
        if not self._grab:
            logging.info(f"😡 You must grab first ! : "
                         f"{self.__class__.__name__, id(self)}")
            raise FailedReadError(self.cap_source)
        self._grab = False
        ret, frame = self.cap.retrieve()
        if not ret:
            logging.info(f"😡 Failed to retrieve {self.cap_source}! : "
                         f"{self.__class__.__name__, id(self)}")
            raise FailedReadError(self.cap_source)
        return frame
