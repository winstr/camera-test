import json
import base64
import traceback
import threading
from queue import Queue
from dataclasses import dataclass

import cv2
from ultralytics import YOLO


@dataclass
class Size:
    width: int
    height: int

    def tuple(self):
        return self.width, self.height


CAMERA_SOURCE = "rtsp://192.168.1.101:554/profile2/media.smp"
SKIPPING_STEP = 5
FRAME_QUEUE = Queue(maxsize=5)
JDATA_QUEUE = Queue(maxsize=5)
TARGET_SIZE = Size(width=640, height=360)


def capture_frame() -> None:
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        err = f"Not opened the source: {CAMERA_SOURCE}"
        raise RuntimeError(err)

    try:
        frame_index = -1
        while True:
            cap.grab()

            frame_index += 1
            if frame_index > SKIPPING_STEP - 1:
                frame_index = 0
            if frame_index % SKIPPING_STEP:
                continue

            ret, frame = cap.read()
            if not ret:
                err = "Failed to read the next frame."
                raise RuntimeError(err)

            FRAME_QUEUE.put(frame)

    except:
        traceback.print_exc()

    finally:
        cap.release()


def process_frame() -> None:
    model = YOLO('yolov8n.pt')
    names = model.names

    while True:
        frame = FRAME_QUEUE.get()
        frame = cv2.resize(frame, TARGET_SIZE.tuple())
        preds = model.track(frame, persist=True, verbose=False)

        frame_b64 = base64.b64encode(frame.tobytes())
        frame_b64str = frame_b64.decode("utf-8")

        if preds is None:
            boxes = []
        else:
            boxes = preds[0].boxes.data.cpu().numpy().tolist()
        
        data = {
            "frame": frame_b64str,
            "names": names,
            "boxes": boxes,
        }

        JDATA_QUEUE.put(json.dumps(data) + '\n')


def main(environ, start_response):
    status = "200 OK"
    response_headers = [("Content-type", "application/octet-stream")]

    start_response(status, response_headers)

    capture_thread = threading.Thread(target=capture_frame, args=())
    process_thread = threading.Thread(target=process_frame, args=())

    capture_thread.start()
    process_thread.start()

    try:
        while True:
            data = JDATA_QUEUE.get()
            if data is None:
                break
            yield data.encode()

    except:
        traceback.print_exc()

    finally:
        capture_thread.join()
        process_thread.join()
