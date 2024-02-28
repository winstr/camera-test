import logging
import asyncio

import cv2
import websockets

from src.video import VideoCaptureThread, Resolution

logging.basicConfig(
    format='[%(levelname)s:%(filename)s:%(funcName)s] %(message)s',
    level=logging.DEBUG
)

async def transmit_data():
    django_server = 'ws://172.27.0.10:8888/ws/path/'
    video_source = 'rtsp://192.168.1.101:554/profile2/media.smp'
    target_size = Resolution(width=640, height=360)
    cap_thread = VideoCaptureThread(video_source, target_size)

    async with websockets.connect(django_server) as websocket:
        logging.info(f'connected to websocket server: {django_server}')
        logging.info('started capture thread.')
        cap_thread.start()

        try:
            while True:
                frame = cap_thread.read()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                _, buffer = cv2.imencode('.jpeg', frame)
                await websocket.send(buffer.tobytes())
        except Exception as e:
            logging.error(f'Exception occurred: {e}')
        finally:
            cap_thread.stop()

if __name__ == '__main__':
    asyncio.run(transmit_data())