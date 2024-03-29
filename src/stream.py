import json
import logging
import asyncio
import traceback
from typing import Union, Type, Tuple, Iterable, Any, Dict

import websockets
from numpy import ndarray
from cv2 import imencode, resize

from src.models import BaseModel
from src.videos import VideoCaptureThread
import src.streamdata as streamdata


def to_json_bytes(data: Dict[str, Any]) -> bytes:
    response = json.dumps(data)
    response = response.encode('utf-8')
    return response


class Request():
    GET_SUPPORTED_MODELS = '0'
    GET_STREAMER_STATES = '1'
    SET_DSIZE = '2'
    STREAM_VIDEO = '3'


class EdgeVisionAIStreamer():

    @staticmethod
    def serialize(frame: ndarray, preds: Dict[str, ndarray]) -> bytes:
        data = streamdata.StreamData()
        is_encoded, frame = imencode('.jpeg', frame)
        if not is_encoded:
            raise RuntimeError('unknown encoding error.')
        frame = frame.tobytes()
        data.frame = frame
        for name, array in preds.items():
            data.preds[name].shape.extend(array.shape)
            data.preds[name].data.extend(array.flatten())
        data = data.SerializeToString()
        return data

    def __init__(self):
        self._video_source = None
        self._receiver_uri = None
        self._model_name = None
        self._dsize = (640,360)

        self._stop = False
        self._continue = asyncio.Event()
        self._is_streaming = False
    
    @property
    def dsize(self) -> Tuple[int, int]:
        return self._dsize
    
    @dsize.setter
    def dsize(self, size: Tuple[int, int]):
        if not isinstance(size, Iterable) or not len(size) == 2:
            logging.warn('size must be a iterable of length 2.')
        elif not all([isinstance(value, int) for value in size]):
            logging.info('all values of size must be integers.')
        elif not all([value > 0 for value in size]):
            logging.info('all values of size must be positive.')
        else:
            self._dsize = size

    def get_states(self) -> Dict[str, Any]:
        return {'video_source': self._video_source,
                'receiver_uri': self._receiver_uri,
                'model_name': self._model_name,
                'dsize': {'width': self._dsize[0],
                          'height': self._dsize[1]},
                'is_streaming': self._is_streaming}

    async def start(
        self,
        video_source: Union[int, str],
        receiver_uri: str,
        model_type: Type[BaseModel]
    ):
        if not issubclass(model_type, BaseModel):
            logging.info('model type must be BaseModel.')
            return

        self._video_source = video_source
        self._receiver_uri = receiver_uri

        async with websockets.connect(receiver_uri) as websocket:
            model = model_type()
            self._model_name = str(model)

            cap_thread = VideoCaptureThread(video_source)
            cap_thread.start()

            self._stop = False
            self._continue.set()
            self._is_streaming = True
            try:
                while True:
                    await self._continue.wait()
                    if self._stop:
                        break
                    frame = cap_thread.read()
                    frame = resize(frame, self._dsize)
                    preds = model.predict(frame)
                    bytes_data = self.serialize(frame, preds)
                    await websocket.send(bytes_data)
                    await asyncio.sleep(0)
            except:
                traceback.print_exc()
            finally:
                cap_thread.stop()
                while cap_thread.is_capturing():
                    await asyncio.sleep(0.1)
                model.release()
                self._stop = True
                self._continue.clear()
                self._is_streaming = False

    def pause(self):
        self._continue.clear()
        self._is_streaming = False

    def resume(self):
        self._continue.set()
        self._is_streaming = True

    def stop(self):
        self._stop = True
        if not self._continue.is_set():
            self._continue.set()

    def is_streaming(self) -> bool:
        return self._is_streaming


class EdgeVisionAIServer():

    def __init__(self, supported_models: Dict[str, Type[BaseModel]]):
        self._streamer = EdgeVisionAIStreamer()
        self._supported_models = supported_models

    def run(self, host: str, port: int):
        start_server = websockets.serve(self._echo, host, port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    async def _echo(self, websocket, path):
        async for recv in websocket:
            if not isinstance(recv, bytes):
                await websocket.send('invalid request.')
                continue

            recv = recv.decode('utf-8')
            recv = json.loads(recv)
            request = recv['request']

            if request == Request.GET_SUPPORTED_MODELS:
                data = self._get_supported_models()
                await websocket.send(str(data))
                await websocket.send(to_json_bytes(data))
            elif request == Request.GET_STREAMER_STATES:
                data = self._get_streamer_states()
                await websocket.send(str(data))
                await websocket.send(to_json_bytes(data))
            elif request == Request.SET_DSIZE:
                self._set_dsize(recv)
            elif request == Request.STREAM_VIDEO:
                await self._stream_video(recv)
            else:
                await websocket.send('Bad request type.')

    def _get_supported_models(self):
        return list(self._supported_models.keys())

    def _get_streamer_states(self):
        return self._streamer.get_states()

    def _set_dsize(self, recv):
        width = int(recv['dsize']['width'])
        height = int(recv['dsize']['height'])
        self._streamer.dsize = (width, height)
        logging.info(self._streamer.dsize)

    async def _stream_video(self, recv):
        model_type = recv['model_type']
        if not model_type in self._supported_models:
            logging.info('invalid model type.')
        else:
            video_source = recv['video_source']
            receiver_uri = recv['receiver_uri']

            if self._streamer.is_streaming():
                self._streamer.stop()
                while self._streamer.is_streaming():
                    await asyncio.sleep(0.1)

            model_type = self._supported_models[model_type]
            asyncio.create_task(
                self._streamer.start(
                    video_source,
                    receiver_uri,
                    model_type
            ))
