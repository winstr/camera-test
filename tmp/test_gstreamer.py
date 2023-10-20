import cv2

from camera.visible_camera import Picam2Gstreamer
from camera.thermal_camera import ThermoCam160B


if __name__ == "__main__":
    cam = Picam2Gstreamer()
    cam.config_capture_mode(capture_mode=0)
    cam.config_frame_resize(640, 480)
    cam.connect(cap_source='0')

    #cam = ThermoCam160B()
    #cam.config_frame_resize(640, 480)
    #cam.connect(cap_source='/dev/cam/ThermoCam160B')

    while True:
        frame = cam.read()
        cv2.imshow('res', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.disconnect()