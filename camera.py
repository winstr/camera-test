import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from picamera2 import Picamera2
from matplotlib.pyplot import cm


class FailedOpenError(RuntimeError): pass
class FailedReadError(RuntimeError): pass
class NotExistModeError(RuntimeError): pass
class InvalidFPSError(RuntimeError): pass


@dataclass
class VideoMode():
    width: int
    height: int
    fps: int

    def values(self):
        return self.width, self.height, self.fps


class CameraCapture():

    VIDEO_MODE = {}

    def __init__(self) -> None:
        self.cap = None
        self.cap_source = None
        self.cap_width = None
        self.cap_height = None
        self.cap_fps = None

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def connect(self,
                cap_source:str,
                cap_width:int,
                cap_height:int,
                cap_fps:int,
                mode:int=None) -> None:

        if self.VIDEO_MODE:
            if not mode in self.VIDEO_MODE.keys():
                raise NotExistModeError(mode)
            if cap_fps > self.VIDEO_MODE[mode].fps:
                raise InvalidFPSError(cap_fps)

        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise FailedOpenError(cap_source)

        self.cap = cap
        self.cap_source = cap_source
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.cap_fps = cap_fps

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

    def postprocess(self, frame) -> np.ndarray:
        # must included resize logic.
        # e.g.  frame = cv2.resize(frame, dsize=(...))
        raise NotImplementedError()


class ThermoCam160B(CameraCapture):

    VENDOR_ID = "1209"
    PRODUCT_ID = "0160"
    MIN_TEMP = -10  # temperature
    MAX_TEMP = 140
    VIDEO_MODE = {0: VideoMode(160, 120, 9)}  # supported only one mode

    def connect(self,
                cap_source:str,
                cap_width:int,
                cap_height:int,
                cap_fps:int) -> None:

        super().connect(cap_source, cap_width, cap_height, cap_fps, mode=0)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('Y', '1', '6', ' '))

    def postprocess(self, frame) -> np.ndarray:
        frame = frame / 65535.0  # rescale 0~65535(16bit img) to 0~1
        frame = frame * (self.MAX_TEMP - self.MIN_TEMP)
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = cm.plasma(frame)
        frame = frame[:, :, :3]
        frame = frame[:, :, ::-1]
        frame = (frame * 255).astype(np.uint8)  # rescale to 0~255(8bit img)
        frame = cv2.resize(frame, (self.cap_width, self.cap_height))
        return frame


class RaspPiCamera3(CameraCapture):

    VIDEO_MODE = {0: VideoMode(1920, 1080, 50),
                  1: VideoMode(1280, 720, 100),
                  2: VideoMode(640, 480, 120)}

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.is_open

    def connect(self,
                csi_id:int,
                cap_width:int,
                cap_height:int,
                cap_fps:int,
                mode:int) -> None:

        if self.VIDEO_MODE:
            if not mode in self.VIDEO_MODE.keys():
                raise NotExistModeError(mode)
            if cap_fps > self.VIDEO_MODE[mode].fps:
                raise InvalidFPSError(cap_fps)

        try:
            mode = self.VIDEO_MODE

            cap = Picamera2(csi_id)
            cap.configure(
                cap.create_preview_configuration(
                    main={"size": (),
                          "format": "RGB888"}))
            cap.start()
        except:
            traceback.print_exc()
            raise FailedOpenError(f"CSI{csi_port}")

        self._cap = cap
        self._cap_source = f"CSI{csi_port}"
        self._display_width = display_width
        self._display_height = display_height

    def disconnect(self) -> None:
        if self.is_connected():
            self._cap.close()
            self._cap = None

    def read(self) -> np.ndarray:
        try:
            frame = self._cap.capture_array()
        except:
            traceback.print_exc()
            raise FailedReadError(self.csi_port)
        frame = cv2.resize(frame,
                           (self._display_width, self._display_height))
        return frame
