import logging

from src.stream import EdgeVisionAIServer

from configs.models import Identity, Yolov8n, Yolov8nPose


logging.basicConfig(
    format='[%(levelname)s:%(filename)s:%(funcName)s] %(message)s',
    level=logging.INFO
)


if __name__ == '__main__':
    supported_models = {
        'Identity': Identity,
        'Yolov8n': Yolov8n,
        'Yolov8nPose': Yolov8nPose,
    }
    server = EdgeVisionAIServer(supported_models)
    server.run('172.27.1.12', 8000)
