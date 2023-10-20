import cv2

from camera_fusion.camera import CameraCapture
from camera_fusion.utils.visible_camera import Picam2Gstreamer
from camera_fusion.utils.thermal_camera import ThermoCam160B


def get_picam() -> CameraCapture:
    cam = Picam2Gstreamer()
    cam.config_capture_mode(capture_mode=0)
    cam.config_frame_resize(640, 480)
    cam.connect(cap_source='0')
    return cam


def get_ircam() -> CameraCapture:
    cam = ThermoCam160B()
    cam.config_frame_resize(640, 480)
    cam.connect(cap_source='/dev/cam/ThermoCam160B')
    return cam


if __name__ == "__main__":
    #cam = get_picam()
    cam = get_ircam()
    while True:
        frame = cam.read()
        cv2.imshow('res', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.disconnect()
