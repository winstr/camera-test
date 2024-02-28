import logging
import threading
import traceback
from typing import Tuple, Union
from dataclasses import dataclass
from queue import Queue, Full, Empty

import cv2
import numpy as np


class ConnectionError(Exception):
    def __init__(self):
        super().__init__('Failed to open the video source.')


class GrabError(Exception):
    def __init__(self):
        super().__init__('Failed to grab the next frame.')


class RetrieveError(Exception):
    def __init__(self):
        super().__init__('Failed to retrieve the next frame.')


@dataclass
class Resolution():
    width: int
    height : int

    def to_tuple(self) -> Tuple[int, int]:
        return self.width, self.height


class VideoCaptureThread(threading.Thread):

    def __init__(self, video_source: str, target_size: Resolution):
        super().__init__()
        self._video_source = video_source
        self._frame_buffer = Queue(maxsize=1)
        self._target_size = target_size
        self._is_ready = threading.Event()
        self._should_stop = False

    def read(self) -> Union[np.ndarray, None]:
        try:
            frame = self._frame_buffer.get_nowait()
        except Empty:
            frame = None
        return frame

    def run(self):
        cap = cv2.VideoCapture(self._video_source)
        if not cap.isOpened():
            raise ConnectionError
        self._is_ready.set()
        logging.info(f'Connection success: {self._video_source}')

        try:
            logging.info('Run capture thread')
            while not self._should_stop:
                self._is_ready.wait()

                is_grabbed = cap.grab()
                if not is_grabbed:
                    raise GrabError

                is_captured, frame = cap.retrieve()
                if not is_captured:
                    raise RetrieveError

                frame = cv2.resize(frame, self._target_size.to_tuple())
                try:
                    self._frame_buffer.put_nowait(frame)
                except Full:
                    self._frame_buffer.get()
                    self._frame_buffer.put(frame)
        except:
            traceback.print_exc()
        finally:
            cap.release()

        logging.info(f'Connection closed: {self._video_source}')

    def pause(self):
        logging.info(f'Pause capture thread')
        self._is_ready.clear()

    def resume(self):
        logging.info(f'Resume capture thread')
        self._is_ready.set()

    def stop(self):
        logging.info(f'Stop capture thread')
        self._should_stop = True
