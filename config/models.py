from typing import Dict

import numpy as np
from ultralytics import YOLO

from src.model import BaseModel


class Identity(BaseModel):
    def __init__(self, model=None):
        pass

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        return {"empty": np.ndarray([])}

    def release(self):
        pass


class Yolov8n(BaseModel):
    """
        box: x_min, y_min ,x_max, y_max, box_id, box_conf, box_category_id
        single box shape: [7,]
    """
    def __init__(self):
        super().__init__(YOLO("yolov8n.pt"))

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
        return {"boxes": boxes}


class Yolov8nPose(BaseModel):
    """
        box: x_min, y_min ,x_max, y_max, box_id, box_conf, box_category_id
        single box shape: [7,]

        kpts: rows: nose, eye, ear, shoulder, knee, ... / cols: x, y, conf
        single kpts shape: [17, 3]
    """
    def __init__(self):
        super().__init__(YOLO('yolov8n-pose.pt'))

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
            kptss = results[0].keypoints.data.cpu().numpy()
        return {"boxes": boxes, "kptss": kptss}
