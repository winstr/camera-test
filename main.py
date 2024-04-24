import json
import logging
import asyncio

import websockets

from src.logger import get_logger
from src.video import Order, VideoStreamer
from config.settings import MODEL_TYPES


class Server():
    def __init__(self):
        self.logger = get_logger(self)
        self.streamer = VideoStreamer()
        self.model = None

    def run(self, host: str, port: int):
        start = websockets.serve(self.echo, host, port)
        self.logger.info(f"Listen {host}:{port} ...")
        asyncio.get_event_loop().run_until_complete(start)
        asyncio.get_event_loop().run_forever()

    async def echo(self, wssp, path):
        self.logger.info(f"OK.")
        async for recv in wssp:
            if not isinstance(recv, bytes):
                await wssp.send("Bad message type.")
                continue
            recv = recv.decode("utf-8")
            recv = json.loads(recv)
            try:
                video_uri = recv["video_uri"]
                wsock_uri = recv["wsock_uri"]
                buffer_size = 1
                model_name = recv["model_name"]
                timeout = None
            except KeyError as key:
                msg = f"Unknown key: {key}"
                await wssp.send(msg)
            else:
                if not model_name in MODEL_TYPES.keys():
                    msg = f"Unknown model name: {model_name}"
                    await wssp.send(msg)
                else:
                    await self.streamer.stop()
                    if self.model is not None:
                        self.model.release()
                    self.model = MODEL_TYPES[model_name]()
                    await self.streamer.start(
                        Order(video_uri,
                              wsock_uri,
                              buffer_size,
                              self.model.encode,
                              timeout))

if __name__ == '__main__':
    try:
        Server().run('172.27.1.21', 8000)
    except Exception as e:
        logging.exception(e)
