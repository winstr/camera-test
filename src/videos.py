import logging
import threading
import traceback
from queue import Queue
from typing import Union

import cv2
import numpy as np


class VideoCaptureOpenError(RuntimeError):

    def __init__(self, source: str):
        err = f'Failed to open the video source: {source}'
        super().__init__(err)


class VideoCaptureReadError(RuntimeError):

    def __init__(self):
        err = 'Failed to read the next frame.'
        super().__init__(err)


class VideoCaptureThread(threading.Thread):

    def __init__(self, video_source: Union[str, int]):
        super().__init__()
        self._source = video_source
        self._buffer = Queue(maxsize=1)

        self._stop = False
        self._continue = threading.Event()
        self._is_capturing = False

    def run(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise VideoCaptureOpenError(self._source)
        logging.info(f'Video source opened. {self._source}')

        self._stop = False
        self._continue.set()
        self._is_capturing = True
        logging.info('Start capturing.')

        try:
            while True:
                self._continue.wait()
                if self._stop:
                    break

                is_captured, frame = cap.read()
                if not is_captured:
                    raise VideoCaptureReadError()

                if self._buffer.full():
                    self._buffer.get()
                self._buffer.put(frame)

        except:
            traceback.print_exc()

        finally:
            logging.info('Video source released.')
            cap.release()
            logging.info('Stopped capturing.')
            self._stop = True
            self._continue.clear()
            self._is_capturing = False

    def pause(self):
        self._continue.clear()
        self._is_capturing = False
        logging.info('Paused capturing.')

    def resume(self):
        self._continue.set()
        self._is_capturing = True
        logging.info('Resumed capturing.')

    def stop(self):
        self._stop = True
        if not self._continue.is_set():
            self._continue.set()
        logging.info('Stopping capture process...')

    def read(self) -> np.ndarray:
        frame = self._buffer.get()
        return frame

    def is_capturing(self) -> bool:
        return self._is_capturing
