import cv2
from flask import Flask, Response
from flask import redirect, url_for

from csi_camera import gstreamer_pipeline


APP = Flask(__name__)
CAP = cv2.VideoCapture(
    gstreamer_pipeline(csi_device=0),
    cv2.CAP_GSTREAMER)


def gen_frames():
    if not CAP.isOpened():
        print('Failed to open camera.')
        exit()

    while True:
        ret, frame = CAP.read()
        if not ret:
            print('Failed to retrieve frame')
            break

        ret, jpeg_frame = cv2.imencode('.jpg', frame)

        yield(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')


@APP.route('/video_feed0')
def video_feed0():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@APP.route('/')
def index():
    return redirect(url_for('video_feed0'))


if __name__ == '__main__':
    try:
        APP.run(host='0.0.0.0', port=5000)
    except:
        pass
    finally:
        CAP.release()
