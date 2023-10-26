"""
-------------------------------
Author: winstr
-------------------------------

Functions:
    get_gstreamer_pipeline(int, int, int, int, int, int) -> str

Classes:
    Camera()
    OCamS1CGNU(Camera)
    ThermoCam160B(Camera)
    RaspberryPiCamera2(Camera)
"""

import traceback
from typing import Union, Dict

import cv2
import numpy as np


def get_gstreamer_pipeline(sensor_id: int=0,
                           src_width: int=3624,
                           src_height: int=2464,
                           fps: int=21,
                           dst_width: int=640,
                           dst_height: int=480) -> str:

    # for NVIDIA Jetson
    pipeline = (
        f'nvarguscamerasrc sensor-id={sensor_id} '
        f'wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 '
        f'ee-strength=1 ! video/x-raw(memory:NVMM), '
        f'width={src_width}, height={src_height}, '
        f'format=NV12, framerate={fps}/1 ! nvvidconv flip-method=0 ! '
        f'video/x-raw, width={dst_width}, height={dst_height}, '
        f'format=BGRx ! videoconvert ! video/x-raw, format=BGR ! '
        f'videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink')

    return pipeline


class Camera():
    """ An wrapping class of cv2.VideoCapture """

    def __init__(self):
        self._source = None   # e.g. /dev/video0 ...
        self._cap = None      # video capture instance
        self._grab = False    # grab state

    def __del__(self):
        self.release()

    def initialize(self,
                   source: Union[int, str],
                   src_width: int,
                   src_height: int,
                   src_fps: int):

        print(
        '[ WARNING ] '
        'During the camera initialization phase, src_width, src_height, '
        'and fps should be initialized with camera mode values provided '
        'by the camera manufacturer. DO NOT PUT ARBITRARY VALUES !!')

        self.release()
        self._source = source
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f'Failed to open {self._source}')
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, src_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_height)
        self._cap.set(cv2.CAP_PROP_FPS, src_fps)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f'Failed to read {self._source}')
        return frame

    def grab(self):
        self._grab = self._cap.grab()

    def retrieve(self) -> np.ndarray:
        if not self._grab:
            raise RuntimeError(f'Failed to grab {self._source}')
        self._grab = False
        ret, frame = self._cap.retrieve()
        if not ret:
            raise RuntimeError(f'Failed to read {self._source}')
        return frame


class OCamS1CGNU(Camera):
    """ A Type of Stereo Camera """

    @staticmethod
    def split(frame: np.ndarray, to_bgr: bool) -> Dict[np.ndarray, np.ndarray]:
        r_frame, l_frame = cv2.split(frame)
        if to_bgr:
            r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BAYER_GB2BGR)
            l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BAYER_GB2BGR)
        return {'right': r_frame, 'left': l_frame}

    def initialize(self,
                   source: Union[int, str],
                   width: int,
                   height: int,
                   fps: int):
        super().initialize(source, width, height, fps)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Bayer

    def set_exposure(self, exposure: int):
        self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


class ThermoCam160B(Camera):
    """ A Type of Infrared Thermal Camera """

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        """ 0 ~ 65535 -> 0 ~ 1 -> min-max normalization -> 0 ~ 255 """
        frame = frame / 65535  # 16-bit image(0 ~ 65535) -> 0 ~ 1
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = (frame * 255).astype(np.uint8)  # -> 8-bit image(0 ~ 255)
        return frame

    def initialize(self,
                   source: Union[int, str],
                   width: int,
                   height: int,
                   fps: int):
        super().initialize(source, width, height, fps)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Bayer
        self._cap.set(cv2.CAP_PROP_FOURCC,
                      cv2.VideoWriter.fourcc('Y', '1', '6', ' '))


class RaspberryPiCamera2(Camera):
    """ A Type of Visible or NoIR Camera """

    def initialize(self, sensor_id:int, gstreamer_pipeline:str):
        self.release()
        self._cap = cv2.VideoCapture(gstreamer_pipeline)
        if not self._cap.isOpened():
            raise RuntimeError(f'Failed to open {self._source}')
        self._source = sensor_id


if __name__ == '__main__':
    # Usage

    def get_ocams():
        ocams = OCamS1CGNU()
        ocams.initialize('/dev/cam/oCamS-1CGN-U', 640, 480, 45)
        ocams.set_exposure(200)
        return ocams
    
    def get_picam():
        sensor_id = 0
        picam = RaspberryPiCamera2()
        picam.initialize(sensor_id, get_gstreamer_pipeline(sensor_id))
        return picam
    
    def get_ircam():
        ircam = ThermoCam160B()
        ircam.initialize('/dev/cam/ThermoCam160B', 160, 120, 9)
        return ircam
    
    """
    # oCamS-1CGN-U
    cam = get_ocams()
    fn = lambda frame: OCamS1CGNU.split(frame).left
    """
    """
    # RaspberryPiCamera2
    cam = get_picam()
    fn = lambda frame: frame
    """
    #"""
    # ThermoCam160B
    cam = get_ircam()
    fn = lambda frame: cv2.resize(ThermoCam160B.normalize(frame), (640, 480))
    #"""

    try:
        while True:
            frame = cam.read()
            frame = fn(frame)
            cv2.imshow('display', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    except:
        traceback.print_exc()

    cv2.destroyAllWindows()
    cam.release()
