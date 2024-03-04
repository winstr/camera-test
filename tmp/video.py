import threading
import traceback
from queue import Queue
from typing import Union

import cv2
import numpy as np


# --- Exceptions ---
class ConnectionError(RuntimeError):
    def __init__(self, source: str):
        super().__init__(f'Failed to open {source}.')


class GrabError(RuntimeError):
    def __init__(self):
        super().__init__('Failed to grab the next frame.')


class RetrieveError(RuntimeError):
    def __init__(self):
        super().__init__('Failed to retrieve the next frame.')
# ---


class VideoCaptureThread(threading.Thread):

    def __init__(self, video_source: str):
        super().__init__()
        self._video_source = video_source
        self._frame_buffer = Queue(maxsize=1)

        self._capture_event = threading.Event()
        self._stop_capturing = False

    def run(self):
        cap = cv2.VideoCapture(self._video_source)
        if not cap.isOpened():
            raise ConnectionError(self._video_source)
        self._capture_event.set()

        try:
            while not self._stop_capturing:
                self._capture_event.wait()

                is_grabbed = cap.grab()
                if not is_grabbed:
                    raise GrabError()

                is_captured, frame = cap.retrieve()
                if not is_captured:
                    raise RetrieveError()

                if self._frame_buffer.full():
                    self._frame_buffer.get()
                self._frame_buffer.put(frame)
        except:
            traceback.print_exc()
        finally:
            cap.release()

    def pause(self):
        self._capture_event.clear()

    def resume(self):
        self._capture_event.set()

    def stop(self):
        self._stop_capturing = True

    def read(self) -> Union[np.ndarray, None]:
        frame = self._frame_buffer.get()
        return frame
