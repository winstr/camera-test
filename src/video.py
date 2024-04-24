import time
import asyncio
import threading
import websockets
from dataclasses import dataclass
from typing import Any, Union, Callable

import cv2
import numpy as np

from src.logger import get_logger


class FailedOpenError(Exception):
    def __init__(self, video_uri: Any):
        msg = f"Failed to open {video_uri!r}."
        super().__init__(msg)


class FailedReadError(Exception):
    def __init__(self, video_uri: Any):
        msg = f"Failed to read frame from {video_uri!r}."
        super().__init__(msg)


class VideoCapture():
    def __init__(self, video_uri: Union[int, str]=None):
        self._cap = cv2.VideoCapture()
        self._lock = threading.Lock()
        self._video_uri = None
        if video_uri is not None:
            self.open(video_uri)

    @property
    def video_uri(self) -> Union[int, str]:
        with self._lock:
            return self._video_uri

    def open(self, video_uri: Union[int, str]):
        with self._lock:
            self._cap.open(video_uri)
            if not self._cap.isOpened():
                self._video_uri = None
                raise FailedOpenError(video_uri)
            else:
                self._video_uri = video_uri

    def read(self) -> np.ndarray:
        with self._lock:
            ret, frame = self._cap.read()
            if not ret:
                raise FailedReadError(self._video_uri)
        return frame

    def release(self):
        with self._lock:
            self._cap.release()
            self._video_uri = None

    def __repr__(self) -> str:
        with self._lock:
            return f"VideoCapture({self._video_uri!r})"


class VideoBuffer():
    def __init__(self, size: int=1):
        self._frames = []
        self._size = self._inspect(size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    @size.setter
    def size(self, value: int):
        new_size = self._inspect(value)
        with self._lock:
            if new_size < self._size:
                self._frames = self._frames[-new_size:]
            self._size = new_size

    def put(self, frame: np.ndarray):
        with self._lock:
            if len(self._frames) >= self._size:
                self._frames.pop(0)
            self._frames.append(frame)
            self._not_empty.notify()

    def get(self, timeout: float=None) -> np.ndarray:
        with self._not_empty:
            if timeout is None:
                while not self._frames:
                    self._not_empty.wait()
            else:
                t0 = time.time()
                while not self._frames:
                    t1 = time.time()
                    remainder = timeout - (t1 - t0)
                    if remainder <= 0:
                        raise TimeoutError(
                            "Timeout within recheck.")
                    if not self._not_empty.wait(remainder):
                        raise TimeoutError(
                            "Timeout within wait.")
            frame = self._frames.pop(0)
            return frame

    @staticmethod
    def _inspect(size: int) -> int:
        if isinstance(size, int) and size > 0:
            return size
        raise ValueError(f"Size must be a positive intger.")

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def __repr__(self) -> str:
        return f"VideoBuffer({self._size!r})"


@dataclass
class Order:
    video_uri: Union[int, str]
    wsock_uri: str
    buffer_size: int
    encode_func: Callable[[np.ndarray], bytes]
    timeout: int


class VideoStreamer():
    def __init__(self):
        self._logger = get_logger(self)
        self._cap = VideoCapture()
        self._buf = VideoBuffer(size=1)
        self._task = None
        self._lock = threading.Lock()
        self._stop_thread = threading.Event()

    async def start(self, order: Order):
        if self._task is not None:
            await self.stop()
        self._task = asyncio.create_task(self._streaming(order))
        self._logger.info("Started a new task.")

    async def stop(self):
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            self._logger.info("Stopped an existing task.")
        except Exception as e:
            msg = f"An unexpected error has occurred. {e}"
            self._logger.exception(msg)
        finally:
            self._task = None

    async def _streaming(self, order: Order):
        async with websockets.connect(order.wsock_uri) as wscp:
            # Prepare VideoCapture and VideoBuffer.
            try:
                self._cap.open(order.video_uri)
                self._buf.size = order.buffer_size
            except FailedOpenError as e:
                msg = f"URI is invalid or unavailable. {e}"
                self._logger.exception(msg)
                return
            except ValueError as e:
                msg = f"Invalid buffer size. {e}"
                self._logger.exception(msg)
                return
            except Exception as e:
                msg = f"An unexpected error has occurred. {e}"
                self._logger.exception(msg)
                return
            # Stream frames to target websocket server.
            try:
                cap_th = threading.Thread(target=self._capture)
                cap_th.start()
                while True:
                    frame = self._buf.get(order.timeout)
                    stream = order.encode_func(frame)
                    await wscp.send(stream)
                    await asyncio.sleep(0)
            except FailedReadError as e:
                msg = f"Cannot update frame. {e}"
                self._logger.exception(msg)
            except TimeoutError as e:
                msg = f"No frames available. {e}"
                self._logger.exception(msg)
            except Exception as e:
                msg = f"An unexpected error has occurred. {e}"
                self._logger.exception(msg)
            finally:
                self._stop_thread.set()
                cap_th.join()
                self._stop_thread.clear()
                self._cap.release()

    def _capture(self):
        while not self._stop_thread.is_set():
            frame = self._cap.read()
            self._buf.put(frame)
