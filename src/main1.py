import traceback

import cv2
import flask
import numpy as np


def convert_to_jpeg(frame: np.ndarray) -> bytes:
    is_encoded, jpeg = cv2.imencode('.jpeg', frame)
    if not is_encoded:
        err = 'Failed to encode the frame.'
        raise RuntimeError(err)
    return jpeg.tobytes()


def convert_to_http_multipart(jpeg: bytes) -> bytes:
    return (
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n'
        + jpeg +
        b'\r\n'
    )


def main():
    camera_source = 'rtsp://192.168.1.101:554/profile2/media.smp'
    target_size = (640, 360)
    skip_interval = 3

    try:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            err = f"The camera is not opened."
            raise RuntimeError(err)
        
        frame_index = -1
        
        while True:
            cap.grab()

            frame_index += 1
            if frame_index > skip_interval - 1:
                frame_index = 0
            if frame_index % skip_interval:
                continue

            ret, frame = cap.retrieve()
            if not ret:
                err = f"Faild to read the frame."
                raise RuntimeError(err)
            
            frame = cv2.resize(frame, target_size)
            fjpeg = convert_to_jpeg(frame)
            fdata = convert_to_http_multipart(fjpeg)

            yield fdata
    
    except:
        traceback.print_exc()
    
    finally:
        cap.release()


app = flask.Flask(__name__)

@app.route('/video')
def video():
    return flask.Response(
        main(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
