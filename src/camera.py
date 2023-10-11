import os
import pickle
import traceback
from datetime import datetime

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
        msg = f'Failed to read: {camera_source}.'
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
        self._camera_source = None
        self._frame_width = None
        self._frame_height = None
        self._fps = None
        self._exposure = None
        self._cap = None
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
                  cvt_rgb:float=0.0):
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
        self._cap = cv2.VideoCapture(self._camera_source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, self._cvt_rgb)
        self._is_connected = True

        if not self._cap.isOpened():
            self.disconnect()
            raise FailedToConnectError(self._camera_source)

    def disconnect(self) -> None:
        if self._is_connected:
            self._cap.release()
            self._cap = None
            self._is_connected = False

    def read(self) -> np.ndarray:
        if not self._is_connected:
            self.connect()

        ret, buffer = self._cap.read()
        if not ret:
            self.disconnect()
            raise FailedToReadError(self._camera_source)

        frame = self._preproc_buffer(buffer)
        return frame

    def display(self, prefix:str='', output_dir:str=None) -> None:
        if output_dir is None:
            output_dir = f'out_{get_datetime()}'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        i = 0
        while True:
            try:
                frame = self.read()
                image = self._prepare_display(frame)
                cv2.imshow(f'{self._camera_source}: {prefix}', image)
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

    def _preproc_buffer(self, buffer) -> np.ndarray:
        raise NotImplementedError()

    def _prepare_display(self, frame) -> np.ndarray:
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
            (self._frame_width, self._frame_height),
            self._calib_data_stereo['R'],
            self._calib_data_stereo['T'])
        
        l_map_1st, l_map_2nd = cv2.initUndistortRectifyMap(
            self._calib_data_stereo['l_mtx'],
            self._calib_data_stereo['l_dist'],
            R1,
            P1,
            (self._frame_width, self._frame_height),
            cv2.CV_16SC2)

        r_map_1st, r_map_2nd = cv2.initUndistortRectifyMap(
            self._calib_data_stereo['r_mtx'],
            self._calib_data_stereo['r_dist'],
            R2,
            P2,
            (self._frame_width, self._frame_height),
            cv2.CV_16SC2)

        self._l_map = (l_map_1st, l_map_2nd)
        self._r_map = (r_map_1st, r_map_2nd)

    def empty_all_calibration_params(self) -> None:
        self._l_calib_data_mono = None
        self._r_calib_data_mono = None
        self._calib_data_stereo = None
        self._l_map = None
        self._r_map = None

    def _preproc_buffer(self, buffer) -> np.ndarray:
        r_frame, l_frame = cv2.split(buffer)  # right, left
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
    
    def _prepare_display(self, frames) -> np.ndarray:
        buffer = []
        if frames['left'] is not None:
            buffer.append(frames['left'])
        if frames['right'] is not None:
            buffer.append(frames['right'])
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

    def display(self, prefix:str='', output_dir:str=None) -> None:
        self.check_chessboard()

        if output_dir is None:
            output_dir = f'out_{get_datetime()}'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        i = 0
        while True:
            try:
                frames = self.read()
                image = self._prepare_display(frames)
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

    def _prepare_display(self, frames) -> np.ndarray:
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
