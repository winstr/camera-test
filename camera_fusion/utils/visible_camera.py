import cv2
import numpy as np

from overrides import overrides

from camera_fusion.camera import CaptureMode
from camera_fusion.camera import CameraCapture
from camera_fusion.camera import FailedOpenError
from camera_fusion.camera import UnsupportedMode
from camera_fusion.camera import InvalidFPSValue


class Picam2Gstreamer(CameraCapture):

    """ Raspberry Pi Camera Module v.2 with GStreamer """

    capture_modes = {0: CaptureMode(3264, 2464, 21),  # default
                     1: CaptureMode(3264, 1848, 28),
                     2: CaptureMode(1920, 1080, 30),
                     3: CaptureMode(1280, 720, 60),
                     4: CaptureMode(1280, 720, 120)}

    def __init__(self) -> None:
        super().__init__()
        self.capture_mode = self.capture_modes[0]
        self.dst_width = self.capture_mode.cap_width
        self.dst_height = self.capture_mode.cap_height

    def config_capture_mode(self, capture_mode:int, fps:int=None) -> None:
        if not capture_mode in self.capture_modes.keys():
            raise UnsupportedMode(capture_mode)
        mode = self.capture_modes[capture_mode]
        if fps is None:
            fps = mode.cap_fps
        elif fps > mode.cap_fps:
            raise InvalidFPSValue(fps)
        self.capture_mode = CaptureMode(mode.cap_width, mode.cap_height, fps)

    def config_frame_resize(self, dst_width:int, dst_height:int) -> None:
        self.dst_width = dst_width
        self.dst_height = dst_height

    @overrides
    def connect(self, cap_source:str) -> None:
        cap_w, cap_h, cap_fps = self.capture_mode.values()
        dst_w, dst_h = self.dst_width, self.dst_height

        gstreamer_pipeline = (
            f'nvarguscamerasrc sensor-id={cap_source} wbmode=3 tnr-mode=2 '
            f'tnr-strength=1 ee-mode=2 ee-strength=1 ! '
            f'video/x-raw(memory:NVMM), width={cap_w}, height={cap_h}, '
            f'format=NV12, '
            f'framerate={cap_fps}/1 ! nvvidconv flip-method=0 ! '
            f'video/x-raw, width={dst_w}, height={dst_h}, format=BGRx ! '
            f'videoconvert ! '
            f'video/x-raw, format=BGR ! videobalance contrast=1.5 '
            f'brightness=-.2 saturation=1.2 ! appsink')

        # need sensor-id extraction...
        super().connect(gstreamer_pipeline)

    @overrides
    def preprocess(self, frame) -> np.ndarray:
        return frame


class oCamS1CGNU(CameraCapture):

    """ WithRobot Stereo Camera Module """

    capture_modes = {0: CaptureMode(1280, 720, 60),
                     1: CaptureMode(1280, 720, 30),
                     2: CaptureMode(640, 480, 45)}

    def __init__(self) -> None:
        super().__init__()
        self.capture_mode = self.capture_modes[0]
        self.dst_width = self.capture_mode.cap_width
        self.dst_height = self.capture_mode.cap_height

    def config_capture_mode(self, capture_mode:int, fps:int=None) -> None:
        if not capture_mode in self.capture_modes.keys():
            raise UnsupportedMode(capture_mode)
        mode = self.capture_modes[capture_mode]
        if fps is None:
            fps = mode.cap_fps
        elif fps > mode.cap_fps:
            raise InvalidFPSValue(fps)
        self.capture_mode = CaptureMode(mode.cap_width, mode.cap_height, fps)

    def config_frame_resize(self, dst_width:int, dst_height:int) -> None:
        self.dst_width = dst_width
        self.dst_height = dst_height

    @overrides
    def connect(self, cap_source: str) -> None:
        super().connect(cap_source)
        cap_w, cap_h, cap_fps = self.capture_mode.values()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
        self.cap.set(cv2.CAP_PROP_FPS, cap_fps)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    @overrides
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        dst_w, dst_h = self.dst_width, self.dst_height
        r_frame, l_frame = cv2.split(frame)
        l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BAYER_GB2BGR)
        r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BAYER_GB2BGR)
        l_frame = cv2.resize(l_frame, (dst_w, dst_h))
        r_frame = cv2.resize(r_frame, (dst_w, dst_h))
        frame = l_frame # temporary
        #frame = cv2.hconcat([l_frame, r_frame])
        return frame

    def preprocess_(self, frame: np.ndarray) -> np.ndarray:
        frame = self.preprocess(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (0, 0), 2)
        frame = cv2.equalizeHist(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame