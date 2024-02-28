import logging
from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO


class PoseEstimator(YOLO):

    def __init__(self, model: str='yolov8n-pose.pt'):
        logging.info(f"Load {model} ...")
        super().__init__(model=model)
        logging.info(f'Done')

    def predict_(self, frame: np.ndarray) -> Dict[str, List[Any]]:
        results = self.track(frame, persist=True, verbose=False)
        boxes, kptss = [], []
        if results is not None:
            boxes = results[0].boxes.data.cpu().numpy().tolist()
            kptss = results[0].keypoints.data.cpu().numpy().tolist()
        return { 'boxes': boxes, 'kptss': kptss }
