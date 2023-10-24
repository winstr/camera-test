import logging
import traceback

import cv2

from camera_fusion.camera_fusion.capture import CameraCapture
from camera_fusion.utils.visible_camera import Picam2Gstreamer
from camera_fusion.utils.visible_camera import oCamS1CGNU
from camera_fusion.utils.thermal_camera import ThermoCam160B


logging.basicConfig(format="[ %(asctime)s, %(levelname)s ] %(message)s",
                    datefmt="%Y.%m.%d %H:%M:%S",
                    level=logging.INFO)


def get_picam(cap_source:str) -> CameraCapture:
    cam = Picam2Gstreamer()
    cam.config_capture_mode(capture_mode=0)
    cam.config_frame_resize(640, 480)
    cam.connect(cap_source=cap_source)
    return cam


def get_ircam(cap_source:str) -> CameraCapture:
    cam = ThermoCam160B()
    cam.config_frame_resize(640, 480)
    cam.connect(cap_source=cap_source)
    return cam

def get_stereo(cap_source:str) -> CameraCapture:
    cam = oCamS1CGNU()
    cam.config_capture_mode(1)
    cam.config_frame_resize(640, 480)
    cam.connect(cap_source=cap_source)
    return cam


def display_cam(cam:CameraCapture):
    try:
        while True:
            cam.grab()
            frame = cam.retrieve()
            frame = cam.preprocess(frame)

            cv2.imshow("res", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except:
        traceback.print_exc()

    cv2.destroyAllWindows()
    cam.disconnect()


if __name__ == "__main__":
    #cam = get_picam('0')
    #cam = get_picam('1')
    cam = get_ircam('/dev/camera/ThermoCam160B')
    #cam = get_stereo('/dev/camera/oCamS-1CGN-U')
    display_cam(cam)
