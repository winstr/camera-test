import json
import base64
import traceback

import cv2
import numpy as np
from ultralytics import YOLO


def convert_to_b64str(frame: np.ndarray) -> str:
    frame_bytes = frame.tobytes()
    frame_base64 = base64.b64encode(frame_bytes)
    frame_base64_string = frame_base64.decode("utf-8")
    return frame_base64_string


def main(environ, start_response):
    status = "200 OK"
    response_headers = [("Content-type", "application/octet-stream")]

    camera_source = 'rtsp://192.168.1.101:554/profile2/media.smp'
    target_size = (640, 360)
    skip_interval = 3

    gst_pipe1 = (
        "rtspsrc location=rtsp://192.168.1.101:554/profile2/media.smp latency=0 ! "
        "rtph264depay ! h264parse ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "avdec_h264 ! videoconvert ! appsink"
    )

    gst_pipe2 = (
        f"rtspsrc location={camera_source} latency=0 ! "
        "rtph264depay ! h264parse ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "nvv4l2decoder ! nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

    model = YOLO('yolov8n.pt')
    names: dict = model.names

    try:
        start_response(status, response_headers)

        cap = cv2.VideoCapture(gst_pipe1, cv2.CAP_GSTREAMER)
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
            preds = model.track(frame, persist=True, verbose=False)
            preds = None

            frame_b64str = convert_to_b64str(frame)
            if preds is not None:
                boxes_list = preds[0].boxes.data.cpu().numpy().tolist()
            else:
                boxes_list = []
            
            data = {
                "frame": frame_b64str,
                "names": names,
                "boxes": boxes_list
            }

            yield (json.dumps(data) + '\n').encode()
    
    except:
        traceback.print_exc()
    
    finally:
        cap.release()
