import logging
import threading
import traceback
from queue import Queue

import cv2
import numpy as np


class VideoCaptureConnectionError(RuntimeError):

    def __init__(self, source: str):
        super().__init__(f'Failed to open the source {source}.')


class VideoCaptureGrabError(RuntimeError):

    def __init__(self):
        super().__init__('Failed to grab the next frame.')


class VideoCaptureRetrieveError(RuntimeError):

    def __init__(self):
        super().__init__('Failed to retrieve the next frame.')


class VideoCaptureThread(threading.Thread):

    def __init__(self, video_source: str):
        super().__init__()
        self._video_source = video_source
        self._frame_buffer = Queue(maxsize=1)
        self._event = threading.Event()
        self._stop_capturing = False

    def run(self):
        cap = cv2.VideoCapture(self._video_source)
        if not cap.isOpened():
            raise VideoCaptureConnectionError(self._video_source)
        logging.info(f'Video source opened. {self._video_source}')

        self._event.set()

        try:
            logging.info('Start capturing.')
            while not self._stop_capturing:
                self._event.wait()

                is_grabbed = cap.grab()
                if not is_grabbed:
                    raise VideoCaptureGrabError()

                is_captured, frame = cap.retrieve()
                if not is_captured:
                    raise VideoCaptureRetrieveError()

                if self._frame_buffer.full():
                    self._frame_buffer.get()
                self._frame_buffer.put(frame)

        except:
            traceback.print_exc()

        finally:
            logging.info('Stopped capturing.')
            cap.release()
            logging.info('Video source released.')

    def pause(self):
        self._event.clear()
        logging.info('Paused capturing.')

    def resume(self):
        self._event.set()
        logging.info('Resumed capturing.')

    def stop(self):
        if not self._event.is_set():
            self._event.set()
        self._stop_capturing = True
        logging.info('Stopping capture process...')

    def read(self) -> np.ndarray:
        frame = self._frame_buffer.get()
        return frame
