from typing import Tuple

import cv2
import numpy as np
from overrides import overrides

from camera_fusion.capture import CameraCapture


class oCamS1CGNU(CameraCapture):
    """ WithRobot Stereo Camera Module

    capture_modes = {0: CaptureMode(1280, 720, 60),
                     1: CaptureMode(1280, 720, 30),
                     2: CaptureMode(640, 480, 45),
                     ... etc.}
    """

    @staticmethod
    def split(frame:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r_frame, l_frame = cv2.split(frame)  # right, left
        for frame in [r_frame, l_frame]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)
        return l_frame, r_frame

    @overrides
    def connect(self,
                cap_source:str,
                cap_width:int,
                cap_height:int,
                cap_fps:int,
                cvt_rgb:float=0.) -> None:

        super().connect(cap_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
        self.cap.set(cv2.CAP_PROP_FPS, cap_fps)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, cvt_rgb)
