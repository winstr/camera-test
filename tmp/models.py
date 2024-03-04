from typing import Any, Dict
from abc import ABC, abstractmethod

import numpy as np
from ultralytics import YOLO


# --- Exceptions ---
class NotNumpyArrayError(TypeError):

    def __init__(self, array_name: str, array: Any):
        msg = f'The {array_name} must be a numpy array not {type(array)}.'
        super().__init__(msg)
# ---


class BaseModel(ABC):

    def __init__(self, model: Any):
        self._model = model

    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        pass


class YoloPoseModel(BaseModel):

    def __init__(self, pt: str='yolov8n-pose.pt'):
        super().__init__(YOLO(pt))

    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        if not isinstance(input_data, np.ndarray):
            raise NotNumpyArrayError('input_data', input_data)
        res = self._model.track(input_data, persist=True, verbose=False)[0]
        if res is None:
            boxes = []
            kptss = []
        else:
            boxes = res.boxes.data.cpu().numpy().tolist()
            kptss = res.keypoints.data.cpu().numpy().tolist()
        return { 'boxes': boxes, 'kptss': kptss }
