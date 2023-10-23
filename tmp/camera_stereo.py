'''
@ author: Seunghyeon Kim (winstr)
@ date:
'''

import os
import pickle
import traceback
from datetime import datetime
from collections import namedtuple

import cv2
import numpy as np


class NotConfiguredError(RuntimeError):
    def __init__(self, camera_source) -> None:
        msg = f'Not configured: {camera_source}'
        super().__init__(msg)


class FailedToConnectError(RuntimeError):
    def __init__(self, camera_source) -> None:
        msg = f'Failed to connect: {camera_source}.'
        super().__init__(msg)


class FailedToReadError(RuntimeError):
    def __init__(self, camera_source) -> None:
        msg = f'Failed to read_frame: {camera_source}.'
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
    '''
    A Camera class to manage the video capturing through various sources
    like local device or online URL. It provides functionalities such as
    connecting to a source, configuring, reading frames, and saving
    grabbed frames.
    
    Attributes:
        _camera_source (str): The video capture source, either a local device path or an online video URL.
        _frame_width (int): Width of the video frame.
        _frame_height (int): Height of the video frame.
        _fps (float): Frames per second.
        _exposure (int): Exposure setting.
        _cvt_rgb (float): Option to convert frame to RGB channel, 0 for False, 1 for True.
        _camera: An OpenCV VideoCapture object.
        _is_connected (bool): A flag indicating whether the camera is currently connected.
        _is_configured (bool): A flag indicating whether the camera is configured.
    
    Methods:
        configure(): Configure camera settings.
        connect(): Connect to the camera.
        disconnect(): Disconnect the camera.
        read_frame(): Read a frame from the camera.
        _preprocess(): A method to be implemented for preprocessing frames.
        grab_frame(): Grab and save frames from the camera.
        _postprocess(): A method to be implemented for postprocessing frames.
    '''

    def __init__(self):
        self._camera_source = None
        self._frame_width = None
        self._frame_height = None
        self._fps = None
        self._exposure = None
        self._cvt_rgb = None
        self._camera = None

        self._is_connected = False
        self._is_configured = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def configure(self,
                         camera_source:str,
                         frame_width:int=640,
                         frame_height:int=480,
                         fps:float=30.0,
                         exposure:int=200,
                         cvt_rgb:float=0.) -> None:
        self._camera_source = camera_source
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._fps = fps
        self._exposure = exposure
        self._cvt_rgb = cvt_rgb

        self._is_configured = True

    def connect(self) -> None:
        if not self._is_configured:
            raise NotConfiguredError(self._camera_source)

        self.disconnect()
        self._camera = cv2.VideoCapture(self._camera_source)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._camera.set(cv2.CAP_PROP_FPS, self._fps)
        self._camera.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
        self._camera.set(cv2.CAP_PROP_CONVERT_RGB, self._cvt_rgb)

        self._is_connected = True

        if not self._camera.isOpened():
            self.disconnect()
            raise FailedToConnectError(self._camera_source)

    def disconnect(self) -> None:
        if self._is_connected:
            self._camera.release()
            self._camera = None

            self._is_connected = False

    def read_frame(self) -> np.ndarray:
        if not self._is_connected:
            self.connect()

        ret, frame = self._camera.read()
        if not ret:
            self.disconnect()
            raise FailedToReadError(self._camera_source)

        frame = self._preprocess(frame)
        return frame

    def _preprocess(self, frame:np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def grab_frame(self,
                   prefix:str='img',
                   output_dir:str='out',
                   exist_dir_ok:bool=True) -> None:
        '''
        Parameters:
            prefix(str): Prefix for the filename of the saved image.
            output_dir(str):
                Directory where the grabbed images will be saved. Creates the directory if it
                doesn't exist, unless exist_dir_ok is False.
            exist_dir_ok(bool):
                If True, the method will use an existing directory, or create it if it doesn't
                exist. If False, an error will be raised if the directory already exists.
        '''
        if os.path.isdir(output_dir) and not exist_dir_ok:
            raise FileExistsError(output_dir)
        elif not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        while True:
            try:
                frame = self.read_frame()
                image = self._postprocess(frame)
                cv2.imshow(prefix, image)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    image_file = f"{'_'.join([prefix, get_datetime()])}.jpg"
                    image_file = os.path.join(output_dir, image_file)
                    cv2.imwrite(image_file, image)
                if key == ord('q'):
                    break

            except:
                print(traceback.format_exc())
                break

        cv2.destroyAllWindows()
        self.disconnect()
    
    def _postprocess(self, frame:np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class StereoCamera(Camera):
    # oCamS-1CGN-U

    __UndistortMono = namedtuple('__UndistortMono', ['mtx', 'dist'])
    __UndistortStereo = namedtuple('__UndistortStereo', ['map_1', 'map_2'])

    def __init__(self) -> None:
        super().__init__()
        self._lens_selection = {
            'left':True,
            'right':True}
        self._undistort_mono = {
            'left': self.__UndistortMono(mtx=None, dist=None),
            'right': self.__UndistortMono(mtx=None, dist=None)}
        self._undistort_stereo = {
            'left': self.__UndistortStereo(map_1=None, map_2=None),
            'right': self.__UndistortStereo(map_1=None, map_2=None)}

    def select_lens(self, left:bool, right:bool) -> None:
        self._lens_selection['left'] = left
        self._lens_selection['right'] = right

    def is_initialized_mono_calibration_data(self) -> bool:
        for lens in self._lens_selection.keys():
            if not self._lens_selection[lens]:
                continue
            calib_data = self._undistort_mono[lens]
            if calib_data.mtx is None or calib_data.dist is None:
                return False
        return True

    def is_initialized_stereo_calibration_data(self) -> bool:
        for lens in self._lens_selection.keys():
            if not self._lens_selection[lens]:
                continue
            calib_data = self._undistort_stereo[lens]
            if calib_data.map_1 is None or calib_data.map_2 is None:
                return False
        return True

    def empty_mono_calibration_data(self) -> None:
        for lens in self._lens_selection.keys():
            calib_data = self._undistort_mono[lens]
            calib_data.mtx = None
            calib_data.dist = None

    def empty_stereo_calibration_data(self) -> None:
        for lens in self._lens_selection.keys():
            calib_data = self._undistort_stereo[lens]
            calib_data.map_1 = None
            calib_data.map_2 = None

    def load_mono_calibration_data(self,
                                   left_calib_file:str=None,
                                   right_calib_file:str=None) -> None:
        if self.is_initialized_stereo_calibration_data():
            self.empty_mono_calibration_data()
            print('Cannot load mono calibration files because stereo calibration'
                  ' is already applied.')
            return

        calib_files = {'left': left_calib_file, 'right': right_calib_file}
        for lens in self._lens_selection.keys():
            if not os.path.isfile(calib_files[lens]):
                raise FileNotFoundError(calib_files[lens])
            if not self._lens_selection[lens] and calib_files[lens] is not None:
                msg = (f'Cannot load the {calib_files[lens]} becuase the '
                       f'{lens}-side lens is disabled. : {self._lens_selection}')
                raise ValueError(msg)
            calib = load_data(calib_files[lens])
            calib_data = self._undistort_mono[lens]
            calib_data.mtx = calib.mtx
            calib_data.dist = calib.dist

    def load_stereo_calibration_data(self, calib_file:str) -> None:
        if self.is_initialized_mono_calibration_data():
            self.empty_mono_calibration_data()
            print('Mono calibrations are disabled.')

        if not os.path.isfile(calib_file):
            raise FileNotFoundError(calib_file)
        calib = load_data(calib_file)

        w = self._frame_width
        h = self._frame_height

        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            calib.l_mtx, calib.l_dist, calib.r_mtx, calib.r_dist, (w, h), calib.R, calib.T)

        (self._undistort_stereo['left'].map_1,
         self._undistort_stereo['left'].map_2) = cv2.initUndistortRectifyMap(
             calib.l_mtx, calib.l_dist, R1, P1, (w, h), cv2.CV_16SC2)

        (self._undistort_stereo['right'].map_1,
         self._undistort_stereo['right'].map_2) = cv2.initUndistortRectifyMap(
             calib.r_mtx, calib.r_dist, R2, P2, (w, h), cv2.CV_16SC2)

    def _preprocess(self, frame) -> np.ndarray:
        r_frame, l_frame = cv2.split(frame)
        frames = {'left': l_frame, 'right': r_frame}

        # Mono Calibration
        for lens in self._lens_selection.keys():
            if not self._lens_selection[lens]:
                continue
            frames[lens] = cv2.cvtColor(frames[lens], cv2.COLOR_BAYER_GB2BGR)
            if self.is_initialized_mono_calibration_data():
                calib_data = self._undistort_mono[lens]
                frames[lens] = undistort_image(frames[lens], calib_data.mtx, calib_data.dist)

        # Stereo Calibration
        if self.is_initialized_stereo_calibration_data():
            for lens in self._lens_selection.keys():
                calib_data = self._undistort_stereo[lens]
                frames[lens] = cv2.remap(
                    frames[lens], calib_data.map_1, calib_data.map_2, cv2.INTER_LINEAR)

        return frames
    
    def _postprocess(self, frames) -> np.ndarray:
        buffer = []
        for lens in self._lens_selection.keys():
            if self._lens_selection[lens]:
                buffer.append(frames[lens])
        
        image = cv2.hconcat(buffer)
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
                cv2.imshow(f'{self._camera_source}: {self._lenses}', image)
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
            image = np.zeros((self._frame_height, self._frame_width, 3))
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