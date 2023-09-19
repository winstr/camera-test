import pickle

import cv2


class CameraOpenError(RuntimeError): pass
class CameraReadError(RuntimeError): pass


def save_data(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_data(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


class Camera():

    def __init__(self, device_file, frame_width=640, frame_height=480, frame_exposure=200,):
        self.cam = cv2.VideoCapture(device_file)
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, False)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, frame_exposure)
        self.is_calibrated = False

    def set_calibration(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.is_calibrated = True

    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            raise CameraOpenError()
        if self.is_calibrated:
            frame = cv2.undistort(frame, self.mtx, self.dist)
        return frame

    def __del__(self):
        self.cam.release()


class oCamS_1CGN_U(Camera):
    """ oCamS-1CGN-U is a stereo camera. """

    def __init__(self, device_file, frame_width=640, frame_height=480, frame_exposure=200,):
        super().__init__(device_file, frame_width, frame_height, frame_exposure)

    def set_calibration(self, mtx_L, dist_L, mtx_R, dist_R):
        super().set_calibration(mtx=(mtx_L, mtx_R), dist=(dist_L, dist_R))

    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            raise CameraOpenError()
        if self.is_calibrated:
            frame[:,:,0] = cv2.undistort(frame[:,:,0], self.mtx[1], self.dist[1])  # right
            frame[:,:,1] = cv2.undistort(frame[:,:,1], self.mtx[0], self.dist[0])  # left
        return frame