from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def detect_lanes(frame_bgr: np.ndarray) -> List[Dict[str, float]]:
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * 0.5):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=30,
    )

    lane_lines: List[Dict[str, float]] = []
    if lines is None:
        return lane_lines

    for line in lines[:12]:
        x1, y1, x2, y2 = line[0]
        lane_lines.append(
            {
                'x1': float(x1),
                'y1': float(y1 + int(h * 0.5)),
                'x2': float(x2),
                'y2': float(y2 + int(h * 0.5)),
                'score': 0.5,
            }
        )
    return lane_lines
