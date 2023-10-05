import cv2
from flask import Flask, Response
from flask import redirect, url_for

#MIN_DEG = 20000
#MAX_DEG = 65535

app = Flask(__name__)


stereo = cv2.VideoCapture('/dev/video0')
stereo.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
stereo.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
stereo.set(cv2.CAP_PROP_EXPOSURE, 100)

thermo = cv2.VideoCapture('/dev/video2')
thermo.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
thermo.set(cv2.CAP_PROP_CONVERT_RGB, 0)

#cams = [stereo, thermo]
#
#fpss = [cam.get(cv2.CAP_PROP_FPS) for cam in cams]
#fps = min(fpss)
#
#for cam in cams:
#    cam.set(cv2.CAP_PROP_FPS, fps)
#    print(cam.get(cv2.CAP_PROP_FPS))


def gen_frames():
    while True:
        ret_stereo, frame_stereo = stereo.read()
        ret_thermo, frame_thermo = thermo.read()
        if not ret_stereo and ret_thermo:
            break
        
        # stereo
        # frame_stereo = cv2.cvtColor(frame_stereo, cv2.COLOR_BAYER_BG2GRAY)

        # thermo
        cv2.normalize(frame_thermo, frame_thermo, 0, 255, cv2.NORM_MINMAX)
        frame_thermo = cv2.convertScaleAbs(frame_thermo)
        frame_thermo = cv2.resize(frame_thermo, dsize=(640, 480))
        frame_thermo = cv2.cvtColor(frame_thermo, cv2.COLOR_GRAY2RGB)

        concat = cv2.hconcat([frame_stereo, frame_thermo])
        ret, buffer = cv2.imencode('.jpg', concat)

        concat = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + concat + b'\r\n')


@app.route('/video_feed0')
def video_feed0():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return redirect(url_for('video_feed0'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)