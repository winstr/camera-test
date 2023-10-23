import time
import logging
import traceback

import cv2
import numpy as np

from camera_fusion.utils.visible_camera import oCamS1CGNU
from camera_fusion.utils.thermal_camera import ThermoCam160B


logging.basicConfig(format="[ %(asctime)s, %(levelname)s ] %(message)s",
                    datefmt="%Y.%m.%d %H:%M:%S",
                    level=logging.INFO)


def detect_edge_canny(img:np.ndarray, min_thresh:int=60, max_thresh:int=200):
    img_canny = cv2.Canny(img, min_thresh, max_thresh)
    return img_canny


def main():
    frame_width = 640
    frame_height = 480

    min_thresh = 50
    max_thresh = 100

    stcam = oCamS1CGNU()
    stcam.config_capture_mode(2, 45)
    stcam.config_frame_resize(frame_width, frame_height)
    stcam.connect("/dev/camera/oCamS-1CGN-U")

    ircam = ThermoCam160B()
    ircam.config_frame_resize(frame_width, frame_height)
    ircam.connect("/dev/camera/ThermoCam160B")

    print(f"🐞 oCamS FPS: {stcam.cap.get(cv2.CAP_PROP_FPS)}")
    print(f"🐞 Thermo FPS: {ircam.cap.get(cv2.CAP_PROP_FPS)}")

    try:
        while True:
            stcam.grab()
            ircam.grab()
            stimg = stcam.preprocess_(stcam.retrieve())
            irimg = ircam.preprocess(ircam.retrieve())

            stimg = cv2.flip(stimg, 0)
            irimg = cv2.flip(irimg, 0)

            concat1 = cv2.hconcat([stimg, irimg])

            stimg = detect_edge_canny(stimg, min_thresh, max_thresh)
            irimg = detect_edge_canny(irimg, min_thresh, max_thresh)

            concat2 = cv2.hconcat([stimg, irimg])
            concat2 = cv2.cvtColor(concat2, cv2.COLOR_GRAY2BGR)

            cv2.imshow('res', cv2.vconcat([concat1, concat2]))
            if cv2.waitKey(200) == ord('q'):
                break
    except:
        traceback.print_exc()
    
    cv2.destroyAllWindows()
    stcam.disconnect()
    ircam.disconnect()


if __name__ == "__main__":
    main()
