import os
import time
import pickle
from typing import Iterator

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist)
    return dst


def gstreamer_pipeline(
        csi_device,
        capture_width=640,
        capture_height=480,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0):
    '''
    usage:
        cap = cv2.VideoCapture(gstreamer_pipeline(device=0), cv2.CAP_GSTREAMER)
        ...

    params:
        - csi_device(int): csi camera port number. ex) 0 -> /dev/video0
        - capture_width(int): ...
        - capture_height(int): ...
        - display_width(int): ...
        - display_height(int): ...
        - framerate(int): ...
        - flip_method(int):
            0: No rotation/flip (default).
            1: Rotate clockwise by 90 degrees.
            2: Flip horizontally.
            3: Rotate clockwise by 180 degrees.
            4: Rotate clockwise by 90 degrees then flip horizontally.
            5: Rotate clockwise by 270 degrees.
            6: Rotate clockwise by 270 degrees then flip horizontally.
            7: Rotate clockwise by 90 degrees.
    '''
    return (
        f'nvarguscamerasrc sensor-id={csi_device} ! video/x-raw(memory:NVMM), '
        f'width=(int){capture_width}, '
        f'height=(int){capture_height}, '
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! video/x-raw, '
        f'width=(int){display_width}, '
        f'height=(int){display_height}, '
        'format=(string)BGRx ! videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink')


class Camera():

    def __init__(self):
        self.camera = None
        self.camera_source = None
        self.is_connected = False

    def connect(self,
                camera_source,
                fps=None,
                width=640,
                height=480,
                exposure=200,
                cvt_rgb=0.,
                fourcc=None,
                max_retries=5,
                delay=2) -> None:
        self.release()
        retry_count = 0
        while retry_count < max_retries:
            camera = cv2.VideoCapture(camera_source)
            if camera.isOpened():
                print(f'Connected to camera {camera_source}'
                      ' successfully.')
                if fps is not None:
                    camera.set(cv2.CAP_PROP_FPS, fps)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
                camera.set(cv2.CAP_PROP_CONVERT_RGB, cvt_rgb)
                if fourcc is not None:
                    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                self.camera = camera
                self.is_connected = True
                return
            print(f'Failed to connect to camera {camera_source}. '
                  f'Retrying in {delay} seconds ...')
            retry_count += 1
            time.sleep(delay)
        print(f'Failed to connect to camera {camera_source} '
              f'after {max_retries} attempts.')

    def release(self) -> None:
        if self.is_connected:
            self.camera.release()
            self.camera = None
            self.is_connected = False

    def preproc(self, frame) -> np.ndarray:
        # TODO
        return frame

    def capture(self) -> np.ndarray:
        if self.camera is None or not self.camera.isOpened():
            self.connect(self.camera_source)
        ret, frame = self.camera.read()
        if not ret:
            print(f'Failed to read frame of {self.camera_source}.')
            frame = None
        frame = self.preproc(frame)
        return frame

    def gen_frames(self, dsize=None) -> Iterator[bytes]:
        # dsize(tuple[int, int]): destination size (width, height).
        while self.is_connected:
            frame = self.capture()
            if dsize is not None:
                frame = cv2.resize(frame, dsize=dsize)
            ret, jpeg_frame = cv2.imencode('.jpg', frame)
            if not ret:
                print(f'Failed to generate jpg frame of {self.camera_source}')
                break
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')


class oCamS_1CGN_U(Camera):

    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30

    CALIB_LEFT_PKL = 'data/oCamS-1CGN-U_left_calib.pkl'
    CALIB_RIGHT_PKL = 'data/oCamS-1CGN-U_right_calib.pkl'

    def __init__(self):
        super().__init__()
        self.lenses = ['left', 'right']
        self._calib_left = None
        self._calib_right = None
        self._set_calibration()

    def set_lenses(self, lenses):
        # possible: ['left', 'right'] or ['left'] or ['right']
        self.lenses = lenses

    def _set_calibration(self):
        if os.path.isfile(self.CALIB_LEFT_PKL):
            self._calib_left = load_data(self.CALIB_LEFT_PKL)
        if os.path.isfile(self.CALIB_RIGHT_PKL):
            self._calib_right = load_data(self.CALIB_RIGHT_PKL)
    
    def connect(self,
                camera_source,
                fps=FPS,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                exposure=0,
                cvt_rgb=0,
                fourcc=None,
                max_retries=5,
                delay=2):
        return super().connect(
            camera_source, fps, width, height, exposure, cvt_rgb,
            fourcc, max_retries, delay)

    def preproc(self, frame):
        frame_right, frame_left = cv2.split(frame)

        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BAYER_GB2BGR)
        if self._calib_left is not None:
            frame_left = undistort_image(
                frame_left, self._calib_left['mtx'], self._calib_left['dist'])
            
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BAYER_GB2BGR)
        if self._calib_right is not None:
            frame_right = undistort_image(
                frame_right, self._calib_right['mtx'], self._calib_right['dist'])

        return frame_left, frame_right

    def capture(self) -> np.ndarray:
        if self.camera is None or not self.camera.isOpened():
            self.connect(self.camera_source)
        ret, frame = self.camera.read()
        if not ret:
            print(f'Failed to read frame of {self.camera_source}.')
            frame = None

        frame_left, frame_right = self.preproc(frame)
        frames = []
        if 'left' in self.lenses:
            frames.append(frame_left)
        if 'right' in self.lenses:
            frames.append(frame_right)
        if not frames:
            raise ValueError(self.lenses)
        frames = cv2.hconcat(frames)
        return frames


class ThermoCam160B(Camera):

    FRAME_WIDTH = 160
    FRAME_HEIGHT = 120
    FPS = 9

    TEMP_MIN = -10
    TEMP_MAX = 140

    def __init__(self):
        super().__init__()

    def connect(self,
                camera_source,
                fps=FPS,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                exposure=0,
                cvt_rgb=0,
                max_retries=5,
                delay=2):
        fourcc = cv2.VideoWriter.fourcc('Y', '1', '6', ' ')
        return super().connect(
            camera_source, fps, width, height, exposure, cvt_rgb,
            fourcc, max_retries, delay)

    def preproc(self, frame):
        frame = (frame / 65535.) * (self.TEMP_MAX - self.TEMP_MIN) + self.TEMP_MIN
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = plt.cm.plasma(frame)
        frame = frame[:, :, :3]
        frame = frame[:, :, ::-1]
        frame = (frame * 255).astype(np.uint8)
        return frame


'''
import cv2

MIN_DEG = 20000
MAX_DEG = 65535

cap = cv2.VideoCapture('/dev/video2')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.normalize(frame, frame, MIN_DEG, MAX_DEG, cv2.NORM_MINMAX)
    frame = cv2.resize(frame, dsize=(640, 480))
    cv2.imshow("", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
'''