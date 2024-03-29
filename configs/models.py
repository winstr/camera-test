from typing import Any, Dict

import numpy as np
from ultralytics import YOLO
from src.models import BaseModel


class Identity(BaseModel):

    def __init__(self, source_model: Any=None):
        pass

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        empty = np.array([])
        return {'empty': empty}

    def is_cuda(self) -> bool:
        return False

    def release(self):
        pass


class Yolov8n(BaseModel):

    def __init__(self):
        super().__init__(YOLO('yolov8n.pt'))

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        results = self._source_model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
        else:
            # shape of each box: (7,)
            # - x_min, y_min ,x_max, y_max, box_id, box_conf, box_category_id
            boxes = results[0].boxes.data.cpu().numpy()
        return {'boxes': boxes}


class Yolov8nPose(BaseModel):

    def __init__(self):
        super().__init__(YOLO('yolov8n-pose.pt'))

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        results = self._source_model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            # shape of each box: (7,)
            # - x_min, y_min ,x_max, y_max, box_id, box_conf, box_category_id
            boxes = results[0].boxes.data.cpu().numpy()
            # shape of each kpts: (17, 3)
            # - rows: body parts (e.g. nose, eye, ear, shoulder, knee, ...)
            # - cols: x, y, conf
            kptss = results[0].keypoints.data.cpu().numpy()
        return {'boxes': boxes, 'kptss': kptss}
