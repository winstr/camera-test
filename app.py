from typing import Generator

import cv2
#import numpy as np
from flask import Flask, Response
#from flask import redirect, url_for

from camera_fusion.camera import CameraCapture
from camera_fusion.utils.visible_camera import Picam2Gstreamer
from camera_fusion.utils.thermal_camera import ThermoCam160B


PICAM = Picam2Gstreamer()
PICAM.config_capture_mode(capture_mode=0, fps=9)
PICAM.config_frame_resize(640, 480)
PICAM.connect(cap_source='0')

THCAM = ThermoCam160B()
THCAM.config_frame_resize(640, 480)
THCAM.connect(cap_source='/dev/cam/ThermoCam160B')


def gen_frames(cam: CameraCapture) -> Generator[bytes, None, None]:
    while True:
        frame = cam.read()
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


APP = Flask(__name__)


@APP.route('/video_feed_picam')
def video_feed_stereo():
    return Response(gen_frames(PICAM),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@APP.route('/video_feed_ircam')
def video_feed_thermo():
    return Response(gen_frames(THCAM),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@APP.route('/')
def index():
    return '''
    <html>
    <body>
        <img src="/video_feed_picam">
        <img src="/video_feed_ircam">
    </body>
    </html>
    '''


if __name__ == '__main__':
    try:
        APP.run(host='0.0.0.0', port=5000)
    except:
        pass
    finally:
        PICAM.disconnect()
        THCAM.disconnect()
