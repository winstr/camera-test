import gc
import json
import logging
import asyncio
import traceback
from typing import Type, Dict

import torch
import websockets
from websockets import WebSocketServerProtocol

from src.model import BaseModel
from src.video import VideoCaptureThread


class StreamSender():

    def __init__(self):
        self._is_streaming = False
        self._streaming = asyncio.Event()

    async def start(self, source_uri: str, target_uri: str, model: BaseModel):
        cap_thread = VideoCaptureThread(source_uri)
        cap_thread.start()

        async with websockets.connect(target_uri) as websocket:
            logging.info(f'Websocket established: {target_uri}')

            try:
                self._streaming.set()
                self._is_streaming = True
                logging.info('Start streaming.')

                while self._streaming.is_set():
                    frame = cap_thread.read()
                    frame = model.predict(frame)
                    frame = model.to_bytes(frame)
                    await websocket.send(frame)
                    await asyncio.sleep(0)

            except:
                traceback.print_exc()
                self._streaming.clear()

            finally:
                logging.info('Websocket terminated.')
                cap_thread.stop()
                self._is_streaming = False

    def stop(self):
        self._streaming.clear()

    def is_streaming(self) -> bool:
        return self._is_streaming
    
class StreamManager():

    def __init__(self, model_types: Dict[str, Type]):
        self._model_types = model_types

        self._stream_sender = StreamSender()
        self._model = None

    def run(self, host: str, port: int):
        start_server = websockets.serve(self._echo, host, port)
        logging.info(f'Started server at {host}:{port}.')
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    async def _echo(self, websocket: WebSocketServerProtocol, path: str):
        async for recv in websocket:
            try:
                # --- TEST
                logging.info(f'received: {recv}')

                source_uri = 'rtsp://192.168.1.101:554/profile2/media.smp'
                target_uri = 'ws://172.27.1.14/ws/stream/'
                if recv == '0':
                    model_type = 'yolov8n'
                else:
                    model_type = 'yolov8n-pose'
                await websocket.send(recv)
                # ---
                if not model_type in self._model_types.keys():
                    await websocket.send('Unknown model type.')
                else:
                    if self._stream_sender.is_streaming():
                        self._stream_sender.stop()
                        while self._stream_sender.is_streaming():
                            await asyncio.sleep(1)
                    if self._model is None:
                        self._model = self._model_types[model_type]()
                    elif self._model.__class__.__name__ != model_type:
                        self._model.to_cpu()
                        del self._model
                        gc.collect()
                        torch.cuda.empty_cache()
                        self._model = self._model_types[model_type]()
                    else:
                        pass
                    asyncio.create_task(self._stream_sender.start(source_uri, target_uri, self._model))
            except:
                traceback.print_exc()