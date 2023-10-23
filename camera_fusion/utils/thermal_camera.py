import cv2
import numpy as np
from matplotlib.pyplot import cm

from overrides import overrides

from camera_fusion.camera import CameraCapture


class UnsupportedMode(RuntimeError): pass
class InvalidFPS(RuntimeError): pass


class ThermoCam160B(CameraCapture):

    """ ThermoEye Infrared Thermal Camera """

    min_temp = -10  # temperature
    max_temp = 140

    def __init__(self) -> None:
        super().__init__()
        """ The supported capture mode is limited to:
            - width: 160 / height: 120 / fps: 9 """
        self.dst_width = 160
        self.dst_height = 120

    def config_frame_resize(self, dst_width:int, dst_height:int) -> None:
        self.dst_width = dst_width
        self.dst_height = dst_height

    @overrides
    def connect(self, cap_source:str) -> None:
        super().connect(cap_source)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('Y', '1', '6', ' '))

    """
    @overrides
    def preprocess(self, frame:np.ndarray) -> np.ndarray:
        frame = frame / 65535.0  # rescale 0~65535(16bit img) to 0~1
        frame = frame * (self.max_temp - self.min_temp)
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = cm.plasma(frame)
        frame = frame[:, :, :3]
        frame = frame[:, :, ::-1]
        frame = (frame * 255).astype(np.uint8)  # rescale to 0~255(8bit img)
        frame = cv2.resize(frame, (self.dst_width, self.dst_height))
        return frame
    """

    @overrides
    def preprocess(self, frame:np.ndarray) -> np.ndarray:
        frame = frame / 65535.0
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = (frame * 255).astype(np.uint8)  # rescale to 0~255(8bit img)
        frame = cv2.resize(frame, (self.dst_width, self.dst_height))
        frame = cv2.equalizeHist(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame
