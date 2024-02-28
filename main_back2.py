import logging
import asyncio
import traceback
import websockets

#import cv2

from src.video import VideoCaptureThread, Resolution


logging.basicConfig(
    format='[%(levelname)s:%(filename)s:%(funcName)s] %(message)s',
    level=logging.DEBUG
)


async def transmit_frame():
    server = '172.27.1.21:8765'
    async with websockets.connect(server) as websocket:
        logging.info(f'connected to websocket server: {server}')
        video_source = 'rtsp://192.168.1.101:554/profile2/media.smp'
        target_size = Resolution(width=640, height=360)
        cap_thread = VideoCaptureThread(video_source, target_size)

        logging.info('started capture thread.')
        cap_thread.start()

        try:
            while True:
                frame = cap_thread.read()
                if frame is None:
                    continue
                await websocket.send(frame)
        except:
            traceback.print_exc()
        finally:
            cap_thread.stop()


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(transmit_frame())