from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from inference_service.lane_detection import detect_lanes


COCO_TO_ADAS = {
    1: 'pedestrian',
    2: 'car',
    3: 'car',
    4: 'car',
    6: 'car',
    8: 'truck',
}


class InferenceRequest(BaseModel):
    frame_id: str
    camera_id: str
    frame_ts_ms: int
    image_b64: str


class BatchInferenceRequest(BaseModel):
    frames: List[InferenceRequest]


app = FastAPI(title='ADAS GPU Inference Service', version='1.0.0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.45'))
backend = os.getenv('INFERENCE_BACKEND', 'torch').strip().lower()
input_size = int(os.getenv('INFERENCE_INPUT_SIZE', '320'))
enable_lane_detection = os.getenv('ENABLE_LANE_DETECTION', '1').strip().lower() not in {'0', 'false', 'no'}

REQUEST_COUNT = Counter('adas_infer_requests_total', 'Total inference requests', ['endpoint'])
FRAME_COUNT = Counter('adas_infer_frames_total', 'Total frames processed')
INFER_LATENCY = Histogram('adas_infer_latency_ms', 'Per-frame inference latency in ms')


def load_model() -> torch.nn.Module:
    # TensorRT fallback path allows interview discussion of backend abstraction.
    if backend == 'tensorrt':
        print('INFERENCE_BACKEND=tensorrt requested; falling back to torch backend in this reference build.')
    net = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    net.to(device)
    net.eval()
    return net


model = load_model()


def decode_frame(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Failed to decode input frame')
    return frame


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    if input_size and frame.shape[0] > 0 and frame.shape[1] > 0:
        frame = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def extract_detections(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        conf = float(score.item())
        cls = int(label.item())
        if conf < conf_threshold or cls not in COCO_TO_ADAS:
            continue
        x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
        detections.append(
            {
                'label': COCO_TO_ADAS[cls],
                'confidence': conf,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return detections


def build_result(payload: InferenceRequest, frame: np.ndarray, pred: Dict[str, Any], started_at: float) -> Dict[str, Any]:
    detections = extract_detections(pred)
    lanes: List[Dict[str, Any]] = []
    if enable_lane_detection:
        lane_frame = cv2.resize(frame, (max(1, frame.shape[1] // 2), max(1, frame.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        lanes = detect_lanes(lane_frame)
    finished_at = time.time()
    infer_ms = (finished_at - started_at) * 1000.0
    INFER_LATENCY.observe(infer_ms)
    FRAME_COUNT.inc()
    return {
        'frame_id': payload.frame_id,
        'camera_id': payload.camera_id,
        'frame_ts_ms': payload.frame_ts_ms,
        'processing_ts_ms': int(finished_at * 1000),
        'inference_ms': infer_ms,
        'detections': detections,
        'lanes': lanes,
    }


def run_single(payload: InferenceRequest) -> Dict[str, Any]:
    started_at = time.time()
    frame = decode_frame(payload.image_b64)
    tensor = preprocess_frame(frame)
    with torch.no_grad():
        pred = model([tensor])[0]
    return build_result(payload, frame, pred, started_at)


@app.get('/health')
def health() -> Dict[str, str]:
    return {
        'status': 'ok',
        'device': str(device),
        'backend': backend,
        'input_size': str(input_size),
        'lane_detection': str(enable_lane_detection).lower(),
        'model': 'ssdlite320_mobilenet_v3_large',
    }


@app.get('/metrics', response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode('utf-8'), media_type=CONTENT_TYPE_LATEST)


@app.post('/infer')
def infer(payload: InferenceRequest) -> Dict[str, Any]:
    REQUEST_COUNT.labels(endpoint='infer').inc()
    return run_single(payload)


@app.post('/infer_batch')
def infer_batch(payload: BatchInferenceRequest) -> Dict[str, Any]:
    REQUEST_COUNT.labels(endpoint='infer_batch').inc()
    started_at = time.time()
    frames = [decode_frame(frame.image_b64) for frame in payload.frames]
    tensors = [preprocess_frame(frame) for frame in frames]
    with torch.no_grad():
        preds = model(tensors)
    results = [build_result(frame_payload, frame, pred, started_at) for frame_payload, frame, pred in zip(payload.frames, frames, preds)]
    finished_at = time.time()
    return {
        'batch_inference_ms': (finished_at - started_at) * 1000.0,
        'results': results,
    }
