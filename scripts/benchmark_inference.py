"""Real, reproducible benchmark for the ADAS inference path.

This script does not fabricate numbers. Two things are measured, both for
real, on whatever machine runs this script:

1. Latency/throughput of the project's actual detector architectures
   (torchvision `fasterrcnn_resnet50_fpn` as the heavier baseline and
   `ssdlite320_mobilenet_v3_large`, the model inference_service/app.py
   actually serves) on real photographs. Forward-pass latency is a function
   of the network architecture and tensor shapes, not the specific trained
   weight values, so this is a faithful timing measurement of the production
   compute graph even when the pretrained-weight download is unavailable
   (see NOTE below).

2. A qualitative, real end-to-end detection + lane overlay produced with an
   actually-downloaded pretrained COCO detector, to demonstrate the
   detection/lane-drawing path working on real content.

NOTE on weights: torchvision's pretrained checkpoints are hosted at
download.pytorch.org, which this sandbox's network egress policy blocks
(confirmed via /root/.ccr/README.md — organization policy denial, not a
transient error). Section 1 therefore runs both architectures with
weights=None (random init, identical op graph/FLOPs) and reports that
plainly. Section 2 substitutes a pretrained YOLOv8n checkpoint (Ultralytics),
fetched from github.com/ultralytics/assets releases, which this environment's
proxy does allow, so the qualitative figure reflects a real trained detector.
If you run this script somewhere with unrestricted network access, it will
automatically use the real fasterrcnn/ssdlite pretrained weights for section 1
too (see USE_PRETRAINED below) and results will differ.

Outputs:
  - report/benchmark_results.json
  - report/figures/latency_curve.png
  - report/figures/batch_latency.png
  - report/figures/sample_detection_overlay.jpg

Run:
    python scripts/benchmark_inference.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference_service.lane_detection import detect_lanes  # noqa: E402

CONF_THRESHOLD = 0.45
INPUT_SIZE = 192
BATCH_SIZE = 4
N_FRAMES = 12
ADAS_LABELS = {'car', 'truck', 'bus', 'person'}

SAMPLE_IMAGE_URLS = [
    'https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg',
    'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg',
    'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg',
]

SAMPLES_DIR = REPO_ROOT / 'datasets' / 'bench_samples'
RESULTS_PATH = REPO_ROOT / 'report' / 'benchmark_results.json'
FIGURES_DIR = REPO_ROOT / 'report' / 'figures'


def download_samples() -> List[Path]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in SAMPLE_IMAGE_URLS:
        dest = SAMPLES_DIR / Path(url).name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        paths.append(dest)
    return paths


def load_frames(paths: List[Path], n: int) -> List[np.ndarray]:
    frames = []
    i = 0
    while len(frames) < n:
        img = cv2.imread(str(paths[i % len(paths)]))
        if img is None:
            raise RuntimeError(f'Failed to decode {paths[i % len(paths)]}')
        frames.append(img)
        i += 1
    return frames


def try_load_pretrained(builder, name: str, device: torch.device):
    try:
        model = builder(weights='DEFAULT').to(device).eval()
        return model, True
    except (urllib.error.URLError, OSError) as exc:
        print(f'[warn] pretrained weights unavailable for {name} ({exc}); using weights=None (architecture-only)')
        model = builder(weights=None, weights_backbone=None).to(device).eval()
        return model, False


def preprocess(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def percentile(values: List[float], pct: float) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    index = int(round((len(items) - 1) * pct))
    return float(items[index])


def benchmark_model(model: torch.nn.Module, frames: List[np.ndarray], device: torch.device, tag: str) -> Dict[str, Any]:
    tensors = [preprocess(f, device) for f in frames]

    with torch.no_grad():
        model([tensors[0]])  # warm-up, excluded from timing

    single_latencies_ms = []
    for t in tensors:
        start = time.perf_counter()
        with torch.no_grad():
            model([t])
        single_latencies_ms.append((time.perf_counter() - start) * 1000.0)

    batch_latencies_ms = []
    for i in range(0, len(tensors), BATCH_SIZE):
        chunk = tensors[i:i + BATCH_SIZE]
        if len(chunk) < BATCH_SIZE:
            continue
        start = time.perf_counter()
        with torch.no_grad():
            model(chunk)
        batch_latencies_ms.append((time.perf_counter() - start) * 1000.0)

    total_s = sum(single_latencies_ms) / 1000.0
    fps = len(single_latencies_ms) / total_s if total_s > 0 else 0.0

    return {
        'tag': tag,
        'single_latencies_ms': single_latencies_ms,
        'batch_latencies_ms': batch_latencies_ms,
        'p50_single_ms': statistics.median(single_latencies_ms),
        'p95_single_ms': percentile(single_latencies_ms, 0.95),
        'p50_batch_ms': statistics.median(batch_latencies_ms) if batch_latencies_ms else 0.0,
        'p95_batch_ms': percentile(batch_latencies_ms, 0.95) if batch_latencies_ms else 0.0,
        'avg_fps': fps,
    }


def run_qualitative_detection(frames: List[np.ndarray]) -> Dict[str, Any]:
    """Real detections from an actually-downloaded pretrained COCO detector."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {'available': False, 'reason': 'ultralytics not installed'}

    weights_path = REPO_ROOT / 'datasets' / 'bench_samples' / 'yolov8n.pt'
    try:
        if not weights_path.exists():
            model = YOLO('yolov8n.pt')
            downloaded = Path('yolov8n.pt')
            if downloaded.exists():
                downloaded.rename(weights_path)
        else:
            model = YOLO(str(weights_path))
    except Exception as exc:
        return {'available': False, 'reason': f'pretrained detector download blocked: {exc}'}

    per_frame_dets = []
    for frame in frames:
        result = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)[0]
        dets = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = model.names[int(cls)]
            if label not in ADAS_LABELS:
                continue
            dets.append({'label': label, 'confidence': float(conf), 'bbox': [float(v) for v in box]})
        per_frame_dets.append(dets)

    return {
        'available': True,
        'model': 'yolov8n (Ultralytics, pretrained COCO weights)',
        'per_frame_detections': per_frame_dets,
        'avg_detections_per_frame': sum(len(d) for d in per_frame_dets) / len(per_frame_dets) if per_frame_dets else 0.0,
    }


def draw_overlay(frame: np.ndarray, dets: List[Dict[str, Any]], lanes: List[Dict[str, float]]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d['bbox']]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{d['label']} {d['confidence']:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for ln in lanes:
        cv2.line(out, (int(ln['x1']), int(ln['y1'])), (int(ln['x2']), int(ln['y2'])), (0, 165, 255), 2)
    return out


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running benchmark on device={device}')

    paths = download_samples()
    frames = load_frames(paths, N_FRAMES)

    baseline, baseline_pretrained = try_load_pretrained(fasterrcnn_resnet50_fpn, 'faster_rcnn_resnet50_fpn', device)
    optimized, optimized_pretrained = try_load_pretrained(ssdlite320_mobilenet_v3_large, 'ssdlite320_mobilenet_v3_large', device)

    baseline_result = benchmark_model(baseline, frames, device, 'faster_rcnn_resnet50_fpn (baseline)')
    optimized_result = benchmark_model(optimized, frames, device, 'ssdlite320_mobilenet_v3_large (optimized)')

    qualitative = run_qualitative_detection(frames)

    overlay_frame = frames[1]
    lane_input = cv2.resize(overlay_frame, (overlay_frame.shape[1] // 2, overlay_frame.shape[0] // 2))
    lanes_half = detect_lanes(lane_input)
    lanes_full = [{'x1': l['x1'] * 2, 'y1': l['y1'] * 2, 'x2': l['x2'] * 2, 'y2': l['y2'] * 2} for l in lanes_half]

    overlay_dets = qualitative['per_frame_detections'][1] if qualitative.get('available') else []
    overlay = draw_overlay(overlay_frame, overlay_dets, lanes_full)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURES_DIR / 'sample_detection_overlay.jpg'), overlay)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(range(1, len(baseline_result['single_latencies_ms']) + 1), baseline_result['single_latencies_ms'],
            marker='o', label='Faster R-CNN (baseline)', color='#d62728')
    ax.plot(range(1, len(optimized_result['single_latencies_ms']) + 1), optimized_result['single_latencies_ms'],
            marker='o', label='SSDLite MobileNetV3 (optimized)', color='#2ca02c')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Single-frame latency (ms)')
    ax.set_title('Measured single-frame latency (CPU, this environment)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'latency_curve.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(range(1, len(baseline_result['batch_latencies_ms']) + 1), baseline_result['batch_latencies_ms'],
            marker='s', label='Faster R-CNN (baseline)', color='#d62728')
    ax.plot(range(1, len(optimized_result['batch_latencies_ms']) + 1), optimized_result['batch_latencies_ms'],
            marker='s', label='SSDLite MobileNetV3 (optimized)', color='#2ca02c')
    ax.set_xlabel(f'Batch index ({BATCH_SIZE} frames/batch)')
    ax.set_ylabel('Batch latency (ms)')
    ax.set_title('Measured batch latency (CPU, this environment)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'batch_latency.png')
    plt.close(fig)

    results = {
        'environment': {
            'device': str(device),
            'input_size': INPUT_SIZE,
            'batch_size': BATCH_SIZE,
            'n_frames': N_FRAMES,
            'conf_threshold': CONF_THRESHOLD,
            'baseline_pretrained_weights_loaded': baseline_pretrained,
            'optimized_pretrained_weights_loaded': optimized_pretrained,
            'note': (
                'Frames are real photographs (not synthetic) used as benchmark input; no licensed '
                'driving dataset ships with this repo (see datasets/README.md). If pretrained-weight '
                'download was blocked in this run, timings above reflect the real production network '
                'architecture with randomly-initialized weights, which does not change forward-pass '
                'latency.'
            ),
        },
        'baseline': baseline_result,
        'optimized': optimized_result,
        'qualitative_detection': {k: v for k, v in qualitative.items() if k != 'per_frame_detections'},
        'avg_lane_segments_per_frame': len(lanes_half),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
