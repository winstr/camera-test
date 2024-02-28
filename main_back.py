import json
import base64
import traceback
import threading
from queue import Queue
from typing import Tuple, Callable, Union, Any, Dict, List
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# gunicorn --workers=2 --bind=0.0.0.0:8000 --timeout=300 pose_estimation:main

@dataclass
class Resolution():

    width: int
    height: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.width, self.height


def capture_livestream(
        source: str,
        frame_dsize: Resolution,
        frame_queue: Queue,
        stop_event: threading.Event
    ) -> None:

    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        error_message = "livestream source is not open."
        raise RuntimeError(error_message)
    
    try:
        while not stop_event.is_set():
            video_capture.grab()
            if frame_queue.full():
                continue
            is_captured, frame = video_capture.retrieve()
            if not is_captured:
                error_message = "frame is not captured."
                RuntimeError(error_message)
            frame = cv2.resize(frame, frame_dsize.to_tuple())
            frame_queue.put(frame)
    except:
        traceback.print_exc()
    finally:
        video_capture.release()
        stop_event.set()


def predict_livestream(
        frame_queue: Queue,
        preds_queue: Queue,
        stop_event: threading.Event,
        predict: Callable[[np.ndarray], Any],
        postproc: Callable[[Any], Dict[str, List[Any]]],
    ) -> None:

    try:
        while not stop_event.is_set():
            frame = frame_queue.get()
            if preds_queue.full():
                continue
            preds = postproc(predict(frame))

            frame = base64.b64encode(frame.tobytes())
            frame = frame.decode("utf-8")
            preds["frame"] = frame

            preds_queue.put(preds)
    except:
        traceback.print_exc()
    finally:
        stop_event.set()


class PoseEstimator(YOLO):

    def __init__(self, model: str="yolov8n-pose.pt"):
        super().__init__(model=model)

    def estimate(self, frame: np.ndarray) -> List[Results]:
        return self.track(frame, persist=True, verbose=False)

    @staticmethod
    def postproc(preds: Union[List[Results], None]) -> Dict[str, List[Any]]:
        boxes, kptss = [], []
        if preds is not None:
            boxes = preds[0].boxes.data.cpu().numpy().tolist()
            kptss = preds[0].keypoints.data.cpu().numpy().tolist()
        return { "boxes": boxes, "kptss": kptss }


def main(environ, start_response):
    status = "200 OK"
    response_headers = [("Content-type", "application/octet-stream")]
    start_response(status, response_headers)

    source = "rtsp://192.168.1.101:554/profile2/media.smp"
    frame_dsize = Resolution(width=640, height=360)
    
    buffer_size = 1
    frame_queue = Queue(buffer_size)
    preds_queue = Queue(buffer_size)
    stop_event = threading.Event()

    model = PoseEstimator()

    capture_thread = threading.Thread(
        target=capture_livestream, args=(source,
                                         frame_dsize,
                                         frame_queue,
                                         stop_event))
    predict_thread = threading.Thread(
        target=predict_livestream, args=(frame_queue,
                                         preds_queue,
                                         stop_event,
                                         model.estimate,
                                         PoseEstimator.postproc))
    
    capture_thread.start()
    predict_thread.start()

    try:
        while not stop_event.is_set():
            data = preds_queue.get()
            if data is None:
                break
            data = json.dumps(data) + "\n"
            yield data.encode()
    except:
        traceback.print_exc()
    finally:
        stop_event.set()
        capture_thread.join()
        predict_thread.join()
