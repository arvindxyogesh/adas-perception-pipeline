# ADAS Real-Time Perception Pipeline (Kafka + Spark + CUDA)

Production-style streaming perception pipeline for ADAS workloads.

## Architecture

1. `producer/producer.py`
- Streams frames from webcam or video to Kafka topic `raw_frames`.

2. `spark_app/streaming_job.py`
- Spark Structured Streaming job reads `raw_frames`.
- Sends batched frames to GPU inference microservice.
- Publishes enriched results to Kafka topic `detections`.

3. `inference_service/app.py`
- CUDA-enabled object detection with PyTorch (`fasterrcnn_resnet50_fpn`).
- Lane detection via OpenCV Canny + Hough transform.
- Batch API (`/infer_batch`) for higher throughput.
- Prometheus metrics endpoint (`/metrics`).
- Backend abstraction with `INFERENCE_BACKEND` env hook (Torch default, TensorRT fallback path).

4. `dashboard/app.py`
- Streamlit dashboard displaying FPS, p50/p95 latency, inference time, and detection volume.

5. Reliability and Product Features
- Retry with exponential backoff between Spark and inference service.
- Service healthchecks in Docker Compose.
- Micro-batch inference from Spark for better GPU utilization.

## Repo Structure

```text
adas-perception-pipeline/
  common/
    metrics.py
    schemas.py
  config/
    settings.yaml
  dashboard/
    app.py
    Dockerfile
    requirements.txt
  datasets/
    README.md
  inference_service/
    app.py
    lane_detection.py
    Dockerfile
    requirements.txt
  producer/
    producer.py
    Dockerfile
    requirements.txt
  scripts/
    download_datasets.sh
    evaluate_precision_recall.py
  spark_app/
    streaming_job.py
    Dockerfile
    requirements.txt
  docker-compose.yml
  README.md
```

## Topics
- `raw_frames`: video frame messages
- `detections`: inference + lane + latency output

## Message Contracts

### raw_frames
```json
{
  "frame_id": "uuid",
  "camera_id": "front_center",
  "frame_ts_ms": 1713950000000,
  "source": "video_stream",
  "image_b64": "..."
}
```

### detections
```json
{
  "frame_id": "uuid",
  "camera_id": "front_center",
  "frame_ts_ms": 1713950000000,
  "processing_ts_ms": 1713950000085,
  "inference_ms": 46.2,
  "pipeline_latency_ms": 92.0,
  "detections": [{"label": "car", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}],
  "lanes": [{"x1": 10, "y1": 600, "x2": 400, "y2": 400, "score": 0.5}]
}
```

## Quickstart

### 1) Add sample video
Place a driving clip at:
- `datasets/sample_drive.mp4`

Or run automated fetch:

```bash
bash scripts/download_datasets.sh
```

The script attempts:
- sample video download
- KITTI devkit download
- nuScenes mini download + integrity check
- BDD100K labels download (official URL, then Kaggle fallback)

### 2) Start pipeline
```bash
docker compose up --build
```

### 3) Open dashboards
- Streamlit metrics UI: `http://localhost:8501`
- Spark Master UI: `http://localhost:8080`
- Inference health: `http://localhost:8000/health`
- Inference Prometheus metrics: `http://localhost:8000/metrics`

## Metrics

Pipeline logs and dashboard show:
- FPS (windowed)
- End-to-end latency p50 and p95
- Inference time
- Throughput (frames/sec)

Inference service Prometheus metrics:
- `adas_infer_requests_total`
- `adas_infer_frames_total`
- `adas_infer_latency_ms`

Target profile:
- ~20 FPS
- p95 latency < 150 ms (simulated environment acceptable)

## Benchmark (real, reproducible numbers)

`scripts/benchmark_inference.py` is a self-contained, reproducible benchmark that measures the
project's two detector architectures (Faster R-CNN baseline vs. the SSDLite MobileNetV3 model
`inference_service/app.py` actually serves) on real photographs, and writes the raw numbers to
`report/benchmark_results.json` plus the figures used in `report/report.tex`. It requires
`torch`, `torchvision`, `opencv-python-headless`, `matplotlib`, and (for the qualitative overlay
only) `ultralytics`.

```bash
pip install torch torchvision opencv-python-headless matplotlib ultralytics
python scripts/benchmark_inference.py
```

Measured on a 4-vCPU CPU-only container (no GPU, `download.pytorch.org` blocked by network
policy, so both architectures ran with `weights=None` — see the script docstring and
`report/report.tex` Section 3 for why that does not affect latency validity):

| Metric | Faster R-CNN (baseline) | SSDLite MobileNetV3 (optimized) |
|---|---|---|
| Single-frame p50 latency | 1260.1 ms | 63.5 ms |
| Single-frame p95 latency | 1284.2 ms | 67.4 ms |
| Average throughput | 0.79 FPS | 15.8 FPS |
| Batch p50 latency (4 frames) | 6359.6 ms | 235.8 ms |
| Batch p95 latency (4 frames) | 7693.1 ms | 271.3 ms |

On this CPU-only host, single-frame latency clears the 150 ms target comfortably, but batch p95
and average throughput fall short of the ~20 FPS / <150 ms targets — the gap is explained in the
report's Discussion section and is expected to close on the CUDA path this service is written for.
Re-run the script on a GPU host to validate that directly; don't take the CPU numbers as a GPU
proxy.

## Precision / Recall Evaluation

1. Export predictions from `detections` Kafka stream to `predictions.jsonl`.
2. Prepare frame-level ground truth at `datasets/ground_truth.jsonl`.
3. Run:
```bash
python scripts/evaluate_precision_recall.py \
  --predictions predictions.jsonl \
  --ground-truth datasets/ground_truth.jsonl
```

## Datasets Used In This Project

This project used the following datasets and sample media:

- Sample driving video clip (`datasets/sample_drive.mp4`) for local streaming and latency benchmarking.
- KITTI devkit artifacts for ADAS/detection evaluation workflow setup.
- nuScenes mini split for autonomous driving dataset integration tests.
- BDD100K labels/archive (downloaded via official source or Kaggle fallback in the downloader).

Dataset sources:

- KITTI: https://www.cvlibs.net/datasets/kitti/
- BDD100K: https://bdd-data.berkeley.edu/
- nuScenes: https://www.nuscenes.org/

Git policy for datasets:

- Large dataset files and extracted assets under `datasets/` are excluded from git via `.gitignore`.
- Only dataset documentation files are tracked in git (for reproducible setup instructions).

See `datasets/README.md` for preparation notes.

## Production Hardening (next steps)

1. Connect true TensorRT engine loading via ONNX export + TRT runtime.
2. Add Kafka schema registry (Avro/Protobuf) for strict event contracts.
3. Add model versioning and canary routing.
4. Add distributed tracing (OpenTelemetry).
5. Add autoscaling for inference workers.

## Interview Demo Script

1. Start stack and open Streamlit + Spark UI.
2. Show Kafka ingest by tailing producer logs.
3. Show Spark micro-batch logs with processed frame counts.
4. Show `inference_ms`, `pipeline_latency_ms`, and FPS trends in dashboard.
5. Show backend abstraction:
  - `INFERENCE_BACKEND=torch` (working path)
  - `INFERENCE_BACKEND=tensorrt` (documented fallback path in reference build)
6. Discuss scaling plan:
  - increase Kafka partitions
  - increase Spark parallelism
  - scale inference replicas behind load balancer

## Resume-Ready Achievement Statement

Built a production-style ADAS perception pipeline using Kafka, Spark Structured Streaming, and CUDA-ready inference; benchmarked two detector architectures end-to-end and measured a ~20x latency/throughput improvement (1260 ms to 63.5 ms median, 0.8 to 15.8 FPS) from switching to SSDLite MobileNetV3, with real-time dashboard observability. See `report/report.pdf` for the full reproducible benchmark methodology and results.

## Copyright

Copyright (c) 2026 Arvind Yogesh. All rights reserved.

See the COPYRIGHT file for additional terms.
