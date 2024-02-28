import json
import logging
import asyncio
import traceback

import cv2
import websockets

from src.video import VideoCaptureThread, Resolution
from src.model import PoseEstimator


logging.basicConfig(
    format='[%(levelname)s:%(filename)s:%(funcName)s] %(message)s',
    level=logging.DEBUG
)


async def send_frame_async(
        django_asgi_url: str,
        stream_source: str,
        target_size: Resolution,
        delimiter: str=b'\xFF\xFE\xFF\xFE'
    ) -> None:

    model = PoseEstimator('yolov8n-pose.pt')
    cap_thread = VideoCaptureThread(stream_source, target_size)
    cap_thread.start()

    async with websockets.connect(django_asgi_url) as websocket:
        try:
            while True:
                frame = cap_thread.read()
                if frame is None:
                    continue
                preds = model.predict_(frame)
                preds = json.dumps(preds).encode('utf-8')

                is_encoded, frame = cv2.imencode('.jpeg', frame)
                if not is_encoded:
                    raise RuntimeError('JPEG encoding error.')
                frame = frame.tobytes()

                combined = frame + delimiter + preds
                await websocket.send(combined)

        except Exception as e:
            traceback.print_exc()
            cap_thread.stop()


if __name__ == "__main__":
    django_asgi_uri = "ws://172.27.1.14:8000/ws/stream/"
    stream_source = "rtsp://192.168.1.101:554/profile2/media.smp"
    target_size = Resolution(width=640, height=360)

    asyncio.run(
        send_frame_async(
            django_asgi_uri,
            stream_source,
            target_size
        )
    )

    # server-side
    # daphne -b 172.27.0.11 -p 8000 config.asgi:application --verbosity 2
    '''
    while True:
        try:
            async with websockets.connect(
                django_asgi_uri, ping_interval=60, ping_timeout=50) as websocket:
                while True:
                    frame = cap_thread.read()
                    if frame is None:
                        continue
                    is_encoded, jpeg = cv2.imencode('.jpg', frame)
                    if not is_encoded:
                        raise RuntimeError('jpg encode error.')
                    await websocket.send(jpeg.tobytes())
        except websockets.exceptions.ConnectionClosed:
            logging.info('raised connection closed exception.')
            logging.info('attempting to reconnect in 5 seconds ...')
            await asyncio.sleep(5)
        except Exception as e:
            logging.info(f'some exception occured: {e}')
            break
    cap_thread.stop()
    '''