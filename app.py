from flask import Flask, Response
from flask import redirect, url_for

from camera import oCamS_1CGN_U, ThermoCam160B


STEREO_CAM = oCamS_1CGN_U()
STEREO_CAM.connect('/dev/camera/oCamS-1CGN-U',
                   fps=9,
                   exposure=400)

THERMO_CAM = ThermoCam160B()
THERMO_CAM.connect('/dev/camera/ThermoCam160B')

APP = Flask(__name__)


@APP.route('/video_feed_stereo')
def video_feed_stereo():
    return Response(STEREO_CAM.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@APP.route('/video_feed_thermo')
def video_feed_thermo():
    return Response(THERMO_CAM.gen_frames(dsize=(640, 480)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@APP.route('/')
def index():
    return '''
    <html>
    <body>
        <img src="/video_feed_stereo">
        <img src="/video_feed_thermo">
    </body>
    </html>
    '''


if __name__ == '__main__':
    try:
        APP.run(host='0.0.0.0', port=5000)
    except:
        pass
    finally:
        STEREO_CAM.release()
        THERMO_CAM.release()
