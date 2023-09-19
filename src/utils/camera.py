import cv2


class CameraOpenError(RuntimeError):
    pass


class CameraReadError(RuntimeError):
    pass


class Camera():

    def __init__(self,
                 device_file,
                 frame_width=640,
                 frame_height=480,
                 frame_exposure=200,):
        self.cam = cv2.VideoCapture(device_file)
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, frame_exposure)
        if not self.cam.isOpened():
            raise CameraOpenError()

    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            raise CameraOpenError()
        return frame

    def show_frame(self, frame, title=None):
        if not title:
            title = f"{id(self.cam)}"
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    
    def __del__(self):
        self.cam.release()


class oCamS_1CGN_U(Camera):
    """ oCamS-1CGN-U is a stereo camera. """

    def __init__(self,
                 device_file="/dev/cam/oCamS-1CGN-U",
                 frame_width=640,
                 frame_height=480,
                 frame_exposure=200,):
        super().__init__(device_file,
                         frame_width,
                         frame_height,
                         frame_exposure)

    def get_frame(self, lens):
        if not lens in ["left", "right", "all"]:
            raise ValueError()

        frame = super().get_frame()
        right_frame, left_frame = cv2.split(frame)

        if lens == "left":
            return left_frame
        if lens == "right":
            return right_frame
        if lens == "all":
            return frame
