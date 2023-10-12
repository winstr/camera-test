import os
import pickle
import traceback
from datetime import datetime

import cv2
import numpy as np


class NotConfiguredError(RuntimeError):
    def __init__(self, cap_src) -> None:
        msg = f'Not configured: {cap_src}'
        super().__init__(msg)


class FailedToConnectError(RuntimeError):
    def __init__(self, cap_src) -> None:
        msg = f'Failed to connect: {cap_src}.'
        super().__init__(msg)


class FailedToReadError(RuntimeError):
    def __init__(self, cap_src) -> None:
        msg = f'Failed to read_frame: {cap_src}.'
        super().__init__(msg)


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")


def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist)
    return dst


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_data(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


class Camera():

    def __init__(self):
        self._cap_src = None
        self._img_w = None
        self._img_h = None
        self._fps = None
        self._exp = None
        self._cvt_rgb = None
        self._cap = None

        self._is_connected = False
        self._is_configured = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def configure(self,
                         cap_src:str,
                         img_w:int=640,
                         img_h:int=480,
                         fps:float=30.0,
                         exp:int=200,
                         cvt_rgb:float=0.) -> None:
        '''
        params:
            cap_src(str): Video capture source
                - local device, e.g. /dev/video0
                - online video URL, e.g. https://...
            img_w(int): Num of pixels of frame width, e.g. 640
            img_h(int): Num of pixels of frame height, e.g. 480
            fps(float): Frame rate or frame per sec.  e.g. 30
            exp(int): exposure. e.g. 200
            cvt_rgb(float): Convert frame to RGB channel
                - 0: False
                - 1: True
        '''
        self._cap_src = cap_src
        self._img_w = img_w
        self._img_h = img_h
        self._fps = fps
        self._exp = exp
        self._cvt_rgb = cvt_rgb
        self._is_configured = True

    def connect(self) -> None:
        if not self._is_configured:
            raise NotConfiguredError(self._cap_src)
        self.disconnect()
        self._cap = cv2.VideoCapture(self._cap_src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._img_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._img_h)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exp)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, self._cvt_rgb)
        self._is_connected = True
        if not self._cap.isOpened():
            self.disconnect()
            raise FailedToConnectError(self._cap_src)

    def disconnect(self) -> None:
        if self._is_connected:
            self._cap.release()
            self._cap = None
            self._is_connected = False

    def read_frame(self) -> np.ndarray:
        if not self._is_connected:
            self.connect()
        ret, frame = self._cap.read()
        if not ret:
            self.disconnect()
            raise FailedToReadError(self._cap_src)
        frame = self._preprocess(frame)
        return frame

    def _preprocess(self, frame:np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def grab_frame(self, prefix:str='', output_dir:str=None) -> None:
        if output_dir is None:
            output_dir = f'out_{get_datetime()}'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        i = 0
        while True:
            try:
                frame = self.read_frame()
                image = self._postprocess(frame)
                cv2.imshow(f'{self._cap_src}: {prefix}', image)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    image_file = f"{'_'.join([prefix, str(i).zfill(5)])}.jpg"
                    image_file = os.path.join(output_dir, image_file)
                    cv2.imwrite(image_file, image)
                    i += 1
                if key == ord('q'):
                    break
            except:
                print(traceback.format_exc())
                break
        cv2.destroyAllWindows()
        self.disconnect()
    
    def _postprocess(self, frame:np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class oCamS1CGNU(Camera):
    ''' a type of stereo camera '''
    def __init__(self) -> None:
        super().__init__()
        self._lenses = {'left':False, 'right':False}

        self._l_calib_data_mono = None  # left
        self._r_calib_data_mono = None  # right

        self._calib_data_stereo = None
        self._l_map = None
        self._r_map = None

    def set_lenses(self, left:bool, right:bool) -> None:
        self._lenses['left'] = left
        self._lenses['right'] = right

    def set_mono_calibration_params(self, l_calib_file, r_calib_file) -> None:
        if self._calib_data_stereo is not None:
            self._calib_data_mono_left = None
            self._calib_data_mono_right = None
            print('stereo calibration is already applied.')
            return
        self._l_calib_data_mono = load_data(l_calib_file)
        self._r_calib_data_mono = load_data(r_calib_file)

    def set_stereo_calibration_params(self, calib_file) -> None:
        if self._l_calib_data_mono is not None and self._r_calib_data_mono is not None:
            self._l_calib_data_mono = None
            self._r_calib_data_mono = None
            print('mono calibrations are disabled.')
        self._calib_data_stereo = load_data(calib_file)

        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            self._calib_data_stereo['l_mtx'],
            self._calib_data_stereo['l_dist'],
            self._calib_data_stereo['r_mtx'],
            self._calib_data_stereo['r_dist'],
            (self._img_w, self._img_h),
            self._calib_data_stereo['R'],
            self._calib_data_stereo['T'])
        
        l_map_1st, l_map_2nd = cv2.initUndistortRectifyMap(
            self._calib_data_stereo['l_mtx'],
            self._calib_data_stereo['l_dist'],
            R1,
            P1,
            (self._img_w, self._img_h),
            cv2.CV_16SC2)

        r_map_1st, r_map_2nd = cv2.initUndistortRectifyMap(
            self._calib_data_stereo['r_mtx'],
            self._calib_data_stereo['r_dist'],
            R2,
            P2,
            (self._img_w, self._img_h),
            cv2.CV_16SC2)

        self._l_map = (l_map_1st, l_map_2nd)
        self._r_map = (r_map_1st, r_map_2nd)

    def empty_all_calibration_params(self) -> None:
        self._l_calib_data_mono = None
        self._r_calib_data_mono = None
        self._calib_data_stereo = None
        self._l_map = None
        self._r_map = None

    def _preprocess(self, frame) -> np.ndarray:
        r_frame, l_frame = cv2.split(frame)  # right, left
        l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BAYER_GB2BGR)
        r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BAYER_GB2BGR)

        if self._l_calib_data_mono is not None and self._r_calib_data_mono is not None:
            l_frame = undistort_image(
                l_frame, self._l_calib_data_mono['mtx'], self._l_calib_data_mono['dist'])
            r_frame = undistort_image(
                r_frame, self._r_calib_data_mono['mtx'], self._r_calib_data_mono['dist'])

        if self._calib_data_stereo is not None:
            l_frame = cv2.remap(l_frame, self._l_map[0], self._l_map[1], cv2.INTER_LINEAR)
            r_frame = cv2.remap(r_frame, self._r_map[0], self._r_map[1], cv2.INTER_LINEAR)

        frames = {'left': None, 'right': None}
        if self._lenses['left']:
            frames['left'] = l_frame
        if self._lenses['right']:
            frames['right'] = r_frame

        return frames
    
    def _postprocess(self, frames) -> np.ndarray:
        frame = []
        if frames['left'] is not None:
            frame.append(frames['left'])
        if frames['right'] is not None:
            frame.append(frames['right'])
        image = cv2.hconcat(frame)
        return image


class ChessboardCapture_oCamS1CGNU(oCamS1CGNU):
    ''' for calibration '''
    def __init__(self) -> None:
        super().__init__()
        self._chessboard = None

    def set_chessboard(self, num_corners_column:int, num_corners_row:int) -> None:
        if not isinstance(num_corners_column, int):
            raise ValueError(num_corners_column)
        if not isinstance(num_corners_row, int):
            raise ValueError(num_corners_row)
        self._chessboard = (num_corners_column, num_corners_row)

    def check_chessboard(self) -> None:
        if self._chessboard is None:
            raise ValueError(self._chessboard)

    def grab_frame(self, prefix:str='', output_dir:str=None) -> None:
        self.check_chessboard()

        if output_dir is None:
            output_dir = f'out_{get_datetime()}'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        i = 0
        while True:
            try:
                frames = self.read_frame()
                image = self._postprocess(frames)
                cv2.imshow(f'{self._cap_src}: {self._lenses}', image)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    if self._lenses['left']:
                        image_file = f"left_{'_'.join([prefix, str(i).zfill(5)])}.jpg"
                        image_file = os.path.join(output_dir, image_file)
                        cv2.imwrite(image_file, frames['left'])
                    if self._lenses['right']:
                        image_file = f"right_{'_'.join([prefix, str(i).zfill(5)])}.jpg"
                        image_file = os.path.join(output_dir, image_file)
                        cv2.imwrite(image_file, frames['right'])
                    i += 1
                if key == ord('q'):
                    break
            except:
                print(traceback.format_exc())
                break
        cv2.destroyAllWindows()
        self.disconnect()

    def _postprocess(self, frames) -> np.ndarray:
        if isinstance(frames['left'], np.ndarray) and isinstance(frames['right'], np.ndarray):
            l_frame, r_frame = frames['left'].copy(), frames['right'].copy()
            l_ret, l_corners = cv2.findChessboardCorners(l_frame, self._chessboard)
            r_ret, r_corners = cv2.findChessboardCorners(r_frame, self._chessboard)
            if l_ret and r_ret:
                cv2.drawChessboardCorners(l_frame, self._chessboard, l_corners, l_ret)
                cv2.drawChessboardCorners(r_frame, self._chessboard, r_corners, r_ret)
            image = cv2.hconcat([l_frame, r_frame])
        elif isinstance(frames['left'], np.ndarray):
            l_frame = frames['left'].copy()
            l_ret, l_corners = cv2.findChessboardCorners(l_frame, self._chessboard)
            if l_ret:
                cv2.drawChessboardCorners(l_frame, self._chessboard, l_corners, l_ret)
            image = l_frame
        elif isinstance(frames['right'], np.ndarray):
            r_frame = frames['right'].copy()
            r_ret, r_corners = cv2.findChessboardCorners(r_frame, self._chessboard)
            if r_ret:
                cv2.drawChessboardCorners(r_frame, self._chessboard, r_corners, r_ret)
            image = r_frame
        else:
            print(f'Warning: All lenses are disabled: {self._lenses}')
            image = np.zeros((self._img_h, self._img_w, 3))
        return image

"""
class ThermoCam160B(Camera):

    FRAME_WIDTH = 160
    FRAME_HEIGHT = 120
    FPS = 9

    TEMP_MIN = -10
    TEMP_MAX = 140

    def __init__(self):
        super().__init__()

    def connect(self,
                cap_src,
                fps=FPS,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                exp=0,
                cvt_rgb=0,
                max_retries=5,
                delay=2):
        fourcc = cv2.VideoWriter.fourcc('Y', '1', '6', ' ')
        return super().connect(
            cap_src, fps, width, height, exp, cvt_rgb,
            fourcc, max_retries, delay)

    def preproc(self, frame):
        frame = (frame / 65535.) * (self.TEMP_MAX - self.TEMP_MIN) + self.TEMP_MIN
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = plt.cm.plasma(frame)
        frame = frame[:, :, :3]
        frame = frame[:, :, ::-1]
        frame = (frame * 255).astype(np.uint8)
        return frame

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
"""