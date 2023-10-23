import time
import logging
import traceback

import cv2

from camera_fusion.utils.visible_camera import oCamS1CGNU
from camera_fusion.utils.thermal_camera import ThermoCam160B


logging.basicConfig(format="[ %(asctime)s, %(levelname)s ] %(message)s",
                    datefmt="%Y.%m.%d %H:%M:%S",
                    level=logging.INFO)


def main():
    frame_width = 640
    frame_height = 480

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
        #count = 0
        while True:
            stcam.grab()
            #count += 1
            #if count % 3:
            #    continue
            ircam.grab()
            #count = 0
            stimg = stcam.preprocess(stcam.retrieve())
            irimg = ircam.preprocess(ircam.retrieve())

            stimg = cv2.flip(stimg, 0)
            irimg = cv2.flip(irimg, 0)

            frame = cv2.hconcat([stimg, irimg])
            cv2.imshow('res', frame)
            if cv2.waitKey(200) == ord('q'):
                break
    except:
        traceback.print_exc()
    
    cv2.destroyAllWindows()
    stcam.disconnect()
    ircam.disconnect()


if __name__ == "__main__":
    main()
