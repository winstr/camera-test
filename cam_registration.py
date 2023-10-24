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


SHARPENING_MASK = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
CANNY_MAX_THRES = 200
CANNY_MIN_THERS = 60


def crop(img:np.ndarray):
    def helper(x:int):
        x_adj = int(x * 0.75)
        gap = int((x - x_adj) / 2)
        return x_adj, gap
    img_width, img_height = img.shape[:2][::-1]
    img_width_adj, w_gap = helper(img_width)
    img_height_adj, h_gap = helper(img_height)
    img_crop = img[h_gap:h_gap+img_height_adj, w_gap:w_gap+img_width_adj, :]
    return img_crop


def main():
    frame_width = 640
    frame_height = 480

    stcam = oCamS1CGNU()
    stcam.config_capture_mode(2)
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

            stimg = cv2.flip(stcam.preprocess(stcam.retrieve()), 0)
            stimg = crop(stimg)
            irimg = cv2.flip(ircam.preprocess(ircam.retrieve()), 0)

            print(stimg.shape, irimg.shape)

            if cv2.waitKey(200) == ord('q'):
                break
    except:
        traceback.print_exc()
    
    cv2.destroyAllWindows()
    stcam.disconnect()
    ircam.disconnect()


if __name__ == "__main__":
    main()
