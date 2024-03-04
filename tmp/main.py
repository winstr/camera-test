import json
import logging
import asyncio
import traceback

import cv2
import websockets

from edge.video import VideoCaptureThread
from edge.models import BaseModel, YoloPoseModel


logging.basicConfig(
    format='[%(levelname)s:%(filename)s:%(funcName)s] %(message)s',
    level=logging.DEBUG
)


class WebsocketClient():

    def __init__(self, server_uri: str):
        # --- configs ---
        self._server_uri = server_uri
        self._reconnection_delay_seconds = 5
        self._reconnection_maximum_count = 10
        # --- components ---
        self._cap_thread = None
        self._pred_model = None
        # --- flags ---
        self._stop_sending = False


    def run(self):
        asyncio.run(self._run())


    async def _run(self):
        reconnection_count = 0
        while not self._stop_sending:
            try:
                async with websockets.connect(self._server_uri) as ws:
                    reconnection_count = 0
                    text = 'hello, world! hi, daphne.'
                    await ws.send(text)
            except:
                traceback.print_exc()
                await asyncio.sleep(self._reconnection_delay_seconds)
                reconnection_count = reconnection_count + 1
                if reconnection_count > self._reconnection_maximum_count:
                    logging.info('Exceeded the maximum reconnection count.')
                    break


'''
def load_model(tag: str):
    if tag == '0':
        model = YoloPoseModel()
    else:
        raise RuntimeError
    return model


async def send_frame_async(
        django_asgi_url: str,
        stream_source: str,
        delimiter: str=b'\xFF\xFE\xFF\xFE'
    ) -> None:

    model = YoloPoseModel()
    cap_thread = VideoCaptureThread(stream_source)
    cap_thread.start()

    async with websockets.connect(django_asgi_url) as ws:
        #await ws.send('init')
        #recv = await ws.recv()
        #if not recv == 'OK':
        #    raise RuntimeError

        try:
            while True:
                frame = cap_thread.read()
                if frame is None:
                    continue
                frame = cv2.resize(frame, (640, 360))
                preds = model.predict(frame)
                preds = json.dumps(preds).encode('utf-8')

                is_encoded, frame = cv2.imencode('.jpeg', frame)
                if not is_encoded:
                    raise RuntimeError('JPEG encoding error.')
                frame = frame.tobytes()

                combined = frame + delimiter + preds
                await ws.send(combined)

        except Exception as e:
            traceback.print_exc()
            cap_thread.stop()


if __name__ == "__main__":
    django_asgi_uri = "ws://172.27.1.14:8000/ws/stream/"
    stream_source = "rtsp://192.168.1.101:554/profile2/media.smp"

    asyncio.run(
        send_frame_async(
            django_asgi_uri,
            stream_source,
        )
    )


class WebsocketClient(ABC):

    def __init__(self, asgi_uri: str):
        self._asgi_url = asgi_uri
        self._reconnect_delay = 1
        self._stop_sending = False

    @abstractmethod
    def generate_context_data(self) -> bytes:
        pass

    @abstractmethod
    def generate_sending_data(self) -> bytes:
        pass

    @abstractmethod
    def process_received_data(self, recv: Any):
        pass

    def run(self):
        asyncio.run(self._send())

    def stop(self):
        self._stop_sending = True

    async def _send(self):
        while not self._stop_sending:
            try:
                async with websockets.connect(self._asgi_url) as ws:
                    self._reconnect_delay = 1
                    context = self.generate_context_data()
                    await ws.send(context)
                    while not self._stop_sending:
                        data = self.generate_sending_data()
                        await ws.send(data)
                        recv = await ws.recv()
                        self.process_received_data(recv)
            except:
                traceback.print_exc()
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay *= 2
                if self._reconnect_delay > 300:
                    break
'''