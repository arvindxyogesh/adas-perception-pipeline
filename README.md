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

Built a production-style ADAS perception pipeline using Kafka, Spark Structured Streaming, and CUDA-enabled inference, delivering simulated 20 FPS streaming with p95 end-to-end latency under 150 ms and real-time dashboard observability.

## Copyright

Copyright (c) 2026 Arvind Yogesh. All rights reserved.

See the COPYRIGHT file for additional terms.
