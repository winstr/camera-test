from typing import Tuple, Dict, List

import cv2
import numpy as np

from src.color import ALL_COLORS, hex2bgr


COLORS = [hex2bgr(c[500]) for c in ALL_COLORS]


def plot_text(
        image: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        font_color: Tuple[int, int, int],
        font_style: int=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float=0.5,
        font_thickness: int=1,
        background_color: Tuple[int, int, int]=None,
    ):

    if background_color:
        w, h = cv2.getTextSize(text,
                               font_style,
                               font_scale,
                               font_thickness)[0]
        x, y = origin
        pt1 = (x, y - h)
        pt2 = (x + w, y)
        cv2.rectangle(image,
                      pt1,
                      pt2,
                      background_color,
                      cv2.FILLED)

    cv2.putText(image,
                text,
                origin,
                font_style,
                font_scale,
                font_color,
                font_thickness)


def plot_bounding_box(
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        box_id: int,
        border_thickness: int=1,
        label: str=None,
    ):

    color = COLORS[box_id % len(COLORS)]

    cv2.rectangle(image,
                  pt1,
                  pt2,
                  color,
                  border_thickness)

    if label:
        plot_text(image,
                  label,
                  pt1,
                  (255, 255, 255),
                  background_color=color)


def plot_keypoints(
        image: np.ndarray,
        kpts: np.ndarray,
        kpts_id: int,
        schema: Dict[int, List[int]],
        thickness: int=1,
        conf_thres: float=0.5,
    ):

    color = COLORS[kpts_id % len(COLORS)]

    pts = kpts[:, :2].astype(int)
    confs = kpts[:, 2]

    for i in range(len(kpts)):
        if confs[i] < conf_thres:
            continue
        pt1 = tuple(pts[i])

        for j in schema[i]:
            if confs[j] < conf_thres:
                continue
            pt2 = tuple(pts[j])
            cv2.line(image,
                     pt1,
                     pt2,
                     color,
                     thickness)
        
        cv2.circle(image,
                   pt1,
                   thickness+1,
                   color,
                   thickness)
