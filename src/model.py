import logging
from typing import Any, Dict
from abc import ABC, abstractmethod

import cv2
import numpy as np
from ultralytics import YOLO

from src.plotting import plot_bounding_box, plot_keypoints


class BaseModel(ABC):

    @abstractmethod
    def predict_(self, frame: np.ndarray) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_cpu(self):
        pass

    def to_bytes(self, frame: np.ndarray) -> bytes:
        is_encoded, frame = cv2.imencode('.jpg', frame)
        if not is_encoded:
            raise RuntimeError('Encoding error.')
        frame = frame.tobytes()
        return frame


class Yolo(BaseModel):

    def __init__(
            self,
            pt: str,
            box_conf_thres: float=0.5
        ):
        self._model = YOLO(pt)
        self._box_conf_thres = box_conf_thres
    
    def predict_(self, frame: np.ndarray) -> Dict[str, Any]:
        results = self._model.track(frame,
                                    persist=True,
                                    verbose=False)
        if results is None:
            boxes = []
        else:
            boxes = results[0].boxes.data.cpu().numpy().tolist()
        return { 'boxes': boxes }

    def predict(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (640, 360))
        boxes = self.predict_(frame)['boxes']

        if boxes:
            boxes = np.array(boxes)
            boxes = boxes[boxes[:, 5] >= self._box_conf_thres]

            for box in boxes:
                obj_id = int(box[4])
                xyxy = box[:4].astype(int)
                pt1 = tuple(xyxy[:2])
                pt2 = tuple(xyxy[2:])
                conf = f'{box[5]:.3f}'
                category = self._model.names[int(box[6])]
                label = f'{category}: {conf}'
                plot_bounding_box(frame,
                                  pt1,
                                  pt2,
                                  obj_id,
                                  label=label)
        return frame
    
    def to_cpu(self):
        self._model.cpu()


class YoloPose(BaseModel):

    schema = {
        0: [1, 2,],
        1: [3,],
        2: [4,],
        3: [],
        4: [],
        5: [6, 7, 11,],
        6: [8, 12,],
        7: [9,],
        8: [10,],
        9: [],
        10: [],
        11: [12, 13,],
        12: [14,],
        13: [15,],
        14: [16,],
        15: [],
        16: [],
    }

    def __init__(
            self,
            pt: str,
            box_conf_thres: float=0.5,
            kpt_conf_thres: float=0.5,
        ):
        self._model = YOLO(pt)
        self._box_conf_thres = box_conf_thres
        self._kpt_conf_thres = kpt_conf_thres

    def predict_(self, frame: np.ndarray) -> Dict[str, Any]:
        results = self._model.track(frame,
                                    persist=True,
                                    verbose=False)
        if results is None:
            boxes = []
            kptss = []
        else:
            boxes = results[0].boxes.data.cpu().numpy().tolist()
            kptss = results[0].keypoints.data.cpu().numpy().tolist()
        return { 'boxes': boxes, 'kptss': kptss }

    def predict(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (640, 360))

        preds = self.predict_(frame)
        boxes = preds['boxes']
        kptss = preds['kptss']

        if boxes and kptss:
            boxes = np.array(boxes)
            kptss = np.array(kptss)

            indices = boxes[:, 5] >= self._box_conf_thres
            boxes = boxes[indices]
            kptss = kptss[indices]

            for box, kpts in zip(boxes, kptss):
                obj_id = int(box[4])
                xyxy = box[:4].astype(int)
                pt1 = tuple(xyxy[:2])
                pt2 = tuple(xyxy[2:])
                conf = f'{box[5]:.3f}'
                category = self._model.names[int(box[6])]
                label = f'{category}: {conf}'
                plot_bounding_box(frame,
                                  pt1,
                                  pt2,
                                  obj_id,
                                  label=label)
                plot_keypoints(frame,
                               kpts,
                               obj_id,
                               self.schema,
                               conf_thres=self._kpt_conf_thres)
        return frame

    def to_cpu(self):
        self._model.cpu()