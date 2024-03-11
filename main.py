import logging

from src.model import Yolo, YoloPose
from src.streaming import StreamManager


logging.basicConfig(
    format='[%(levelname)s:%(filename)s:%(funcName)s] %(message)s',
    level=logging.DEBUG
)


class Yolov8n(Yolo):
    def __init__(self):
        super().__init__(pt='yolov8n.pt')


class Yolov8nPose(YoloPose):
    def __init__(self):
        super().__init__(pt='yolov8n-pose.pt')


if __name__ == '__main__':
    model_types = {
        'yolov8n': Yolov8n,
        'yolov8n-pose': Yolov8nPose,
    }

    stream_manager = StreamManager(model_types)
    stream_manager.run('172.27.1.12', 8000)
