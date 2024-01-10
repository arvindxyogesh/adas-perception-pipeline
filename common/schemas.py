from __future__ import annotations

from typing import Any, Dict, List


def frame_message(
    frame_id: str,
    camera_id: str,
    frame_ts_ms: int,
    image_b64: str,
    source: str,
) -> Dict[str, Any]:
    return {
        "frame_id": frame_id,
        "camera_id": camera_id,
        "frame_ts_ms": frame_ts_ms,
        "source": source,
        "image_b64": image_b64,
    }


def detection_message(
    frame_id: str,
    camera_id: str,
    frame_ts_ms: int,
    processing_ts_ms: int,
    detections: List[Dict[str, Any]],
    lanes: List[Dict[str, Any]],
    inference_ms: float,
    pipeline_latency_ms: float,
) -> Dict[str, Any]:
    return {
        "frame_id": frame_id,
        "camera_id": camera_id,
        "frame_ts_ms": frame_ts_ms,
        "processing_ts_ms": processing_ts_ms,
        "inference_ms": inference_ms,
        "pipeline_latency_ms": pipeline_latency_ms,
        "detections": detections,
        "lanes": lanes,
    }
