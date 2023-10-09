import cv2
import numpy as np


class FailedToConnectError(RuntimeError):
    def __init__(self, camera_source) -> None:
        msg = f'Failed to connect: {camera_source}.'
        super().__init__(msg)


class FailedToReadError(RuntimeError):
    def __init__(self, camera_source) -> None:
        msg = f'Failed to read: {camera_source}.'
        super().__init__(msg)


class oCamS_1CGN_U():
    ''' stereo camera '''

    def __init__(self,
                 camera_source:str,
                 frame_width:int=640,
                 frame_height:int=480,
                 fps:int=30,
                 exposure:int=200,
                 target_lenses:list=['left', 'right']):

        self.camera_source = camera_source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.exposure = exposure
        self.target_lenses = target_lenses
        self.cap = None

    def connect_camera(self):
        self.cap = cv2.VideoCapture(self.camera_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

        if not self.cap.isOpened():
            self.disconnect_camera()
            raise FailedToConnectError(self.camera_source)

    def disconnect_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def grab_frames(self):
        if self.cap is None:
            self.connect_camera()

        ret, buffer = self.cap.read()
        if not ret:
            self.disconnect_camera()
            raise FailedToReadError(self.camera_source)

        r_frame, l_frame = cv2.split(buffer)  # right, left
        l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BAYER_GB2BGR)
        r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BAYER_GB2BGR)
        
        frame = {}
        frame['left'] = l_frame if 'left' in self.target_lenses else None
        frame['right'] = r_frame if 'right' in self.target_lenses else None

        return frame


def capture_chessboard(ocams:oCamS_1CGN_U,
                       chessboard:tuple,
                       output_dir:str) -> None:

    i = 0
    while True:
        frame = ocams.grab_frames()
        l_frame, r_frame = None, None
        if frame['left'] and frame['right']:
            l_frame, r_frame = frame['left'].copy(), frame['right'].copy()
            l_ret, l_corners = cv2.findChessboardCorners(l_frame, chessboard)
            r_ret, r_corners = cv2.findChessboardCorners(r_frame, chessboard)
            if l_ret and r_ret:
                cv2.drawChessboardCorners(l_frame, chessboard, l_corners, l_ret)
                cv2.drawChessboardCorners(r_frame, chessboard, r_corners, r_ret)
        elif frame['left']:
            l_frame, r_frame = frame['left'].copy()
            l_ret, l_corners = cv2.findChessboardCorners(l_frame, chessboard)
            if l_ret:
                cv2.drawChessboardCorners(l_frame, chessboard, l_corners, l_ret)
        elif frame['right']:
            r_frame = frame['right'].copy()
            r_ret, r_corners = cv2.findChessboardCorners(r_frame, chessboard)
            if r_ret:
                cv2.drawChessboardCorners(r_frame, chessboard, r_corners, r_ret)
        